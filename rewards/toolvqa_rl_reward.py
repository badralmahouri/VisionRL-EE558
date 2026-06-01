"""Soft reward for ToolVQA RL trajectories.

ToolVQA answers are often short open-ended phrases or numbers, so exact string
matching is too brittle for RL. This reward mirrors the matching used by
evaluation/compute_toolvqa_rl_accuracy.py:

- exact normalized match
- numeric match when the ground truth contains numbers
- short ground-truth phrase contained in the prediction
"""

from __future__ import annotations

import json
import re
from typing import Any

TOOLS_BLOCK = re.compile(r"<\|tools_prefix\|>(\[.*?\])<\|tools_suffix\|>", re.DOTALL)


def _normalize(text: Any) -> str:
    text = re.sub(r"<\|[^|]+?\|>", " ", str(text))
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9.%-]+", " ", text)
    return " ".join(text.split()).strip(" .,!?:;\t\n")


def _extract_numbers(text: str) -> list[str]:
    return re.findall(r"[-+]?\d+(?:\.\d+)?%?", text)


def _soft_match(pred: str, gt: str) -> bool:
    pred_norm = _normalize(pred)
    gt_norm = _normalize(gt)
    if not pred_norm or not gt_norm:
        return False
    if pred_norm == gt_norm:
        return True

    gt_nums = _extract_numbers(gt_norm)
    if gt_nums:
        pred_nums = _extract_numbers(pred_norm)
        if any(num in pred_nums for num in gt_nums):
            return True

    gt_words = gt_norm.split()
    if 1 <= len(gt_words) <= 5:
        return f" {gt_norm} " in f" {pred_norm} "
    return False


def _extract_display_answers(text: str) -> list[str] | None:
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


def _strip_special(text: str) -> str:
    text = re.sub(r"<\|tools_prefix\|>.*?<\|tools_suffix\|>", " ", text, flags=re.DOTALL)
    text = re.sub(r"<\|[^|]+?\|>", " ", text)
    return " ".join(text.split()).strip()


def _extract_prediction(output: str) -> str:
    answers = _extract_display_answers(output)
    if answers:
        return answers[-1].strip()

    clean = _strip_special(output)
    patterns = [
        r"(?:final\s+answer|answer)\s*(?:is|:|=)?\s*([^\.\n]+)",
        r"^([^\.\n]+)$",
    ]
    for pattern in patterns:
        match = re.search(pattern, clean, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip().strip('"\'')
    return clean


def compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    **kwargs,
) -> float:
    pred = _extract_prediction(str(solution_str or ""))
    if isinstance(ground_truth, (list, tuple)):
        return 1.0 if any(_soft_match(pred, str(gt)) for gt in ground_truth) else 0.0
    return 1.0 if _soft_match(pred, str(ground_truth)) else 0.0


def _run_self_tests() -> None:
    cases = [
        ("exact phrase", "Hot liquids.", "Hot liquids.", 1.0),
        ("short phrase contained", "The answer is hot liquids for someone feeling down.", "Hot liquids.", 1.0),
        ("numeric contained", "There are 4 more actions.", "4", 1.0),
        ("display_answers", '<|tools_prefix|>[{"display_answers": {"answers": ["Microphone"]}}]<|tools_suffix|>', "microphone", 1.0),
        ("wrong", "The answer is a controller.", "Microphone", 0.0),
    ]
    failures = 0
    for label, sol, gt, expected in cases:
        got = compute_score("toolvqa_rl", sol, gt)
        ok = got == expected
        failures += int(not ok)
        print(f"[{'OK' if ok else 'FAIL'}] {label}: got={got} expected={expected}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    _run_self_tests()
