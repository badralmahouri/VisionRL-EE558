"""Compute ToolVQA RL accuracy from VeRL rollout JSONL files.

This is the ToolVQA analogue of compute_line_drawing_rl_accuracy.py. It reads
rollout rows dumped by VeRL validation, extracts the final answer from the model
output, compares it against the reward_model ground truth, and writes
predictions.jsonl plus accuracy.json.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

TOOLS_BLOCK = re.compile(r"<\|tools_prefix\|>(\[.*?\])<\|tools_suffix\|>", re.DOTALL)


def normalize(text: Any) -> str:
    text = re.sub(r"<\|[^|]+?\|>", " ", str(text))
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9.%-]+", " ", text)
    return " ".join(text.split()).strip(" .,!?:;\t\n")


def extract_numbers(text: str) -> list[str]:
    return re.findall(r"[-+]?\d+(?:\.\d+)?%?", text)


def exact_match(pred: str, gt: str) -> bool:
    return normalize(pred) == normalize(gt)


def soft_match(pred: str, gt: str) -> bool:
    pred_norm = normalize(pred)
    gt_norm = normalize(gt)
    if not pred_norm or not gt_norm:
        return False
    if pred_norm == gt_norm:
        return True

    gt_nums = extract_numbers(gt_norm)
    if gt_nums:
        pred_nums = extract_numbers(pred_norm)
        if any(num in pred_nums for num in gt_nums):
            return True

    # Useful for short open-ended answers, e.g. "top trenz" inside a sentence.
    gt_words = gt_norm.split()
    if 1 <= len(gt_words) <= 5:
        return f" {gt_norm} " in f" {pred_norm} "
    return False


def extract_display_answers(text: str) -> list[str] | None:
    if not text:
        return None
    blocks = TOOLS_BLOCK.findall(text)
    if not blocks:
        return None
    for block in reversed(blocks):
        try:
            calls = json.loads(block)
        except json.JSONDecodeError:
            continue
        if not isinstance(calls, list):
            continue
        for call in reversed(calls):
            if not isinstance(call, dict):
                continue
            args = call.get("display_answers")
            if isinstance(args, dict) and isinstance(args.get("answers"), list) and args["answers"]:
                return [str(a) for a in args["answers"]]
    return None


def strip_special(text: str) -> str:
    text = re.sub(r"<\|tools_prefix\|>.*?<\|tools_suffix\|>", " ", text, flags=re.DOTALL)
    text = re.sub(r"<\|[^|]+?\|>", " ", text)
    return " ".join(text.split()).strip()


def extract_prediction(output: str) -> str:
    answers = extract_display_answers(output)
    if answers:
        return answers[-1].strip()

    clean = strip_special(output)
    patterns = [
        r"(?:final\s+answer|answer)\s*(?:is|:|=)?\s*([^\.\n]+)",
        r"^([^\.\n]+)$",
    ]
    for pattern in patterns:
        m = re.search(pattern, clean, flags=re.IGNORECASE)
        if m:
            return m.group(1).strip().strip('"\'')
    return clean


def iter_rollout_rows(rollout_dir: Path) -> list[dict[str, Any]]:
    files = sorted(rollout_dir.glob("*.jsonl"))
    if not files:
        raise FileNotFoundError(f"no rollout JSONL files found in {rollout_dir}")
    rows: list[dict[str, Any]] = []
    for path in files:
        with path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    row["_rollout_file"] = str(path)
                    rows.append(row)
    return rows


def get_output(row: dict[str, Any]) -> str:
    for key in ("output", "response", "responses", "generated_text", "text"):
        val = row.get(key)
        if isinstance(val, str):
            return val
        if isinstance(val, list) and val and isinstance(val[-1], str):
            return val[-1]
    return ""


def get_ground_truth(row: dict[str, Any]) -> str:
    for key in ("gts", "ground_truth", "answer"):
        val = row.get(key)
        if isinstance(val, str):
            return val
        if isinstance(val, list) and val:
            return str(val[0])
    reward_model = row.get("reward_model")
    if isinstance(reward_model, dict) and "ground_truth" in reward_model:
        return str(reward_model["ground_truth"])
    extra_info = row.get("extra_info")
    if isinstance(extra_info, dict) and "answer" in extra_info:
        return str(extra_info["answer"])
    return ""


def get_question_id(row: dict[str, Any], idx: int) -> Any:
    for key in ("question_id", "id", "uid"):
        if key in row:
            return row[key]
    extra_info = row.get("extra_info")
    if isinstance(extra_info, dict):
        for key in ("index", "question_id", "id"):
            if key in extra_info:
                return extra_info[key]
    return idx



def load_prediction_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def score_rows(rows: list[dict[str, Any]], source: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    predictions = []
    exact_correct = 0
    soft_correct = 0
    for i, row in enumerate(rows):
        if "prediction" in row and "answers" in row:
            pred = str(row.get("prediction", ""))
            answers = row.get("answers") or []
            gt = str(answers[0]) if answers else ""
            output = str(row.get("raw_prediction", pred))
        else:
            output = get_output(row)
            gt = get_ground_truth(row)
            pred = extract_prediction(output)
        is_exact = exact_match(pred, gt)
        is_soft = soft_match(pred, gt)
        exact_correct += int(is_exact)
        soft_correct += int(is_soft)
        predictions.append({
            "question_id": get_question_id(row, i),
            "question": row.get("question"),
            "prediction": pred,
            "answers": [gt],
            "exact_correct": is_exact,
            "soft_correct": is_soft,
            "correct": is_soft,
            "raw_prediction": output,
            "rollout_file": row.get("_rollout_file"),
        })

    exact_accuracy = exact_correct / len(rows) if rows else 0.0
    soft_accuracy = soft_correct / len(rows) if rows else 0.0
    acc = {
        "accuracy": soft_accuracy,
        "soft_accuracy": soft_accuracy,
        "exact_accuracy": exact_accuracy,
        "num_samples": len(rows),
        "num_correct": soft_correct,
        "num_soft_correct": soft_correct,
        "num_exact_correct": exact_correct,
        "source": source,
    }
    return predictions, acc


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute ToolVQA soft/exact accuracy")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--rollout-dir", default=None, help="Directory containing VeRL rollout JSONL files.")
    group.add_argument("--predictions", default=None, help="Baseline predictions JSONL file.")
    parser.add_argument("--output-dir", default=None, help="Directory for normalized predictions.jsonl and accuracy.json.")
    parser.add_argument("--output", default=None, help="Optional explicit accuracy JSON path.")
    args = parser.parse_args()

    if args.rollout_dir:
        input_path = Path(args.rollout_dir)
        rows = iter_rollout_rows(input_path)
        default_output_dir = input_path.parent
        source = str(input_path)
        print(f"Loaded {len(rows)} rollout rows")
    else:
        input_path = Path(args.predictions)
        rows = load_prediction_rows(input_path)
        default_output_dir = input_path.parent
        source = str(input_path)
        print(f"Loaded {len(rows)} prediction rows")

    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions, acc = score_rows(rows, source=source)

    pred_path = output_dir / "predictions.jsonl"
    with pred_path.open("w", encoding="utf-8") as f:
        for row in predictions:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    acc["predictions"] = str(pred_path)
    if args.rollout_dir:
        acc["rollout_dir"] = str(input_path)
    else:
        acc["input_predictions"] = str(input_path)

    acc_path = Path(args.output) if args.output else output_dir / "accuracy.json"
    acc_path.parent.mkdir(parents=True, exist_ok=True)
    acc_path.write_text(json.dumps(acc, indent=2) + "\n")
    print(f"Wrote predictions: {pred_path}")
    print(f"Wrote accuracy:    {acc_path}")
    print(f"Soft accuracy:  {acc['soft_accuracy']:.4f} ({acc['num_soft_correct']}/{acc['num_samples']})")
    print(f"Exact accuracy: {acc['exact_accuracy']:.4f} ({acc['num_exact_correct']}/{acc['num_samples']})")


if __name__ == "__main__":
    main()
