"""Convert VeRL rotate-flip rollouts into prediction and accuracy files.

VeRL rollout logs contain rows like:
    {"input": ..., "output": "D", "gts": "B", "score": 0.0, "step": 1}

This script writes the same prediction/accuracy artifacts used by the baseline
jobs so all three systems can be plotted together.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any


def extract_answer(text: Any) -> str:
    if text is None:
        return ""
    clean = re.sub(r"<\|[^|]+?\|>", " ", str(text)).strip()
    clean = " ".join(clean.split())
    match = re.search(r"(?:answer|word|text)\s*(?:is|:|=)?\s*([A-Za-z]+)", clean, flags=re.IGNORECASE)
    if match:
        return match.group(1).strip()
    match = re.search(r"\b([A-Za-z][A-Za-z]+)\b", clean)
    return match.group(1).strip() if match else clean


def normalize_text(text: Any) -> str:
    clean = re.sub(r"<\|[^|]+?\|>", " ", str(text or "")).strip()
    return " ".join(clean.strip(" .,!?:;\"'").split())


def load_rollout_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        with path.open(encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                row["_rollout_file"] = str(path)
                row["_line_no"] = line_no
                rows.append(row)
    return rows


def load_metadata_answers(path: Path | None) -> dict[Any, str]:
    if path is None:
        return {}
    answers: dict[Any, str] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            answers[row.get("question_id")] = str(row.get("canonical_answer", row.get("answer", "")))
    return answers


def normalize_gt(gts: Any) -> str:
    if isinstance(gts, list):
        gts = gts[0] if gts else ""
    return normalize_text(gts)


def get_gold(row: dict[str, Any], metadata_answers: dict[Any, str]) -> str:
    qid = row.get("question_id")
    if qid in metadata_answers:
        return normalize_gt(metadata_answers[qid])
    if row.get("answers"):
        return normalize_gt(row["answers"])
    return normalize_gt(row.get("gts", row.get("ground_truth", row.get("answer"))))


def build_prediction_rows(rows: list[dict[str, Any]], metadata_answers: dict[Any, str] | None = None) -> list[dict[str, Any]]:
    predictions: list[dict[str, Any]] = []
    metadata_answers = metadata_answers or {}
    for idx, row in enumerate(rows):
        gold = get_gold(row, metadata_answers)
        raw_prediction = row.get("output", row.get("raw_prediction", row.get("prediction", "")))
        pred = extract_answer(raw_prediction)
        correct = normalize_text(pred) == gold and bool(gold)
        predictions.append(
            {
                "question_id": row.get("question_id", idx),
                "question": row.get("question", ""),
                "prediction": pred,
                "raw_prediction": raw_prediction,
                "answers": [gold],
                "accuracy": 1.0 if correct else 0.0,
                "step": row.get("step"),
                "score": row.get("score"),
                "rollout_file": row.get("_rollout_file"),
                "rollout_line": row.get("_line_no"),
            }
        )
    return predictions


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_accuracy(path: Path, predictions: list[dict[str, Any]], rollout_paths: list[Path]) -> None:
    num_samples = len(predictions)
    num_correct = sum(1 for row in predictions if row["accuracy"] == 1.0)
    num_case_insensitive_correct = sum(
        1
        for row in predictions
        if row["answers"]
        and normalize_text(row["prediction"]).lower() == normalize_text(row["answers"][0]).lower()
        and normalize_text(row["answers"][0])
    )
    overall = num_correct / num_samples if num_samples else 0.0
    case_insensitive_overall = num_case_insensitive_correct / num_samples if num_samples else 0.0
    by_step: dict[str, dict[str, int]] = {}
    for row in predictions:
        step = str(row.get("step", "unknown"))
        by_step.setdefault(step, {"num_samples": 0, "num_correct": 0})
        by_step[step]["num_samples"] += 1
        by_step[step]["num_correct"] += int(row["accuracy"] == 1.0)

    result = {
        "mode": "mcq",
        "overall_accuracy": round(overall, 4),
        "case_insensitive_accuracy": round(case_insensitive_overall, 4),
        "num_samples": num_samples,
        "num_correct": num_correct,
        "num_case_insensitive_correct": num_case_insensitive_correct,
        "num_perfect": num_correct,
        "timestamp": datetime.now().isoformat(),
        "rollout_files": [str(path) for path in rollout_paths],
        "by_step": {
            step: {
                **counts,
                "accuracy": round(counts["num_correct"] / counts["num_samples"], 4) if counts["num_samples"] else 0.0,
            }
            for step, counts in sorted(by_step.items())
        },
        "per_sample": [
            {
                "question_id": row["question_id"],
                "prediction": row["prediction"],
                "top_answer": row["answers"][0] if row["answers"] else "",
                "accuracy": row["accuracy"],
                "case_insensitive_accuracy": (
                    1.0
                    if row["answers"]
                    and normalize_text(row["prediction"]).lower() == normalize_text(row["answers"][0]).lower()
                    and normalize_text(row["answers"][0])
                    else 0.0
                ),
                "step": row.get("step"),
            }
            for row in predictions
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate rotate-flip VeRL rollout JSONL files")
    parser.add_argument("--rollout-dir", default="results/rotate_flip_rl/trained_tool/rollouts")
    parser.add_argument("--rollout-file", action="append", default=None, help="Explicit rollout JSONL path; can be repeated")
    parser.add_argument("--output-dir", default="results/rotate_flip_rl/trained_tool")
    parser.add_argument("--metadata", default=None, help="Optional metadata JSONL used to backfill gold answers by question_id.")
    args = parser.parse_args()

    if args.rollout_file:
        rollout_paths = [Path(path) for path in args.rollout_file]
    else:
        rollout_paths = sorted(Path(args.rollout_dir).glob("*.jsonl"))
    if not rollout_paths:
        raise FileNotFoundError(f"no rollout JSONL files found in {args.rollout_dir}")

    rows = load_rollout_rows(rollout_paths)
    if not rows:
        raise ValueError(f"no rollout rows found in {rollout_paths}")

    output_dir = Path(args.output_dir)
    metadata_answers = load_metadata_answers(Path(args.metadata)) if args.metadata else {}
    predictions = build_prediction_rows(rows, metadata_answers=metadata_answers)
    predictions_path = output_dir / "predictions.jsonl"
    accuracy_path = output_dir / "accuracy.json"
    write_jsonl(predictions_path, predictions)
    write_accuracy(accuracy_path, predictions, rollout_paths)

    num_correct = sum(1 for row in predictions if row["accuracy"] == 1.0)
    accuracy = num_correct / len(predictions) if predictions else 0.0
    print(f"Loaded {len(rows)} rollout rows from {len(rollout_paths)} file(s)")
    print(f"Wrote predictions: {predictions_path}")
    print(f"Wrote accuracy:    {accuracy_path}")
    print(f"Accuracy: {accuracy:.4f} ({num_correct}/{len(predictions)})")


if __name__ == "__main__":
    main()
