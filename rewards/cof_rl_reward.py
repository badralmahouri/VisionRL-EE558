"""1/0 exact final-answer reward for synthetic RL trajectories.

The model can answer in either Apertus's native display_answers format:
    <|tools_prefix|>[{"display_answers": {"answers": ["<X>", ...]}}]<|tools_suffix|>

or, for simple MCQ tasks, as plain text:
    B
    Answer: B
    The answer is B.

We first pull the *last* display_answers call out of the rollout's solution
string. If no display_answers call exists, we fall back to extracting a plain
multiple-choice answer.

Wired into verl via:
    reward.custom_reward_function:
      path: <abs>/rewards/cof_rl_reward.py
      name: compute_score

The filename is legacy; rotate/flip configs use this module as their shared
exact-match reward.
"""

import json
import re

TOOLS_BLOCK = re.compile(r"<\|tools_prefix\|>(\[.*?\])<\|tools_suffix\|>", re.DOTALL)
CHOICE_RE = re.compile(r"\b([ABCD])\b", re.IGNORECASE)
FINAL_CHOICE_PATTERNS = [
    re.compile(r"(?:final\s+answer|answer|target|choice)\s*(?:is|:|=)?\s*([ABCD])\b", re.IGNORECASE),
    re.compile(r"\b([ABCD])\s*(?:is\s+)?(?:the\s+)?(?:answer|target|choice)\b", re.IGNORECASE),
]


def _extract_display_answers(solution_str: str) -> list[str] | None:
    """Return the `answers` list of the last display_answers call, or None."""
    if not solution_str:
        return None
    blocks = TOOLS_BLOCK.findall(solution_str)
    if not blocks:
        return None
    try:
        calls = json.loads(blocks[-1])
    except json.JSONDecodeError:
        return None
    if not isinstance(calls, list) or not calls:
        return None
    call = calls[-1]
    if not isinstance(call, dict):
        return None
    args = call.get("display_answers")
    if not isinstance(args, dict) or not isinstance(args.get("answers"), list):
        return None
    return [str(a) for a in args["answers"]]


def _extract_plain_choice(solution_str: str) -> list[str] | None:
    """Return a plain-text A/B/C/D answer fallback, or None."""
    if not solution_str:
        return None
    clean = re.sub(r"<\|[^|]+?\|>", " ", str(solution_str)).strip()
    if not clean:
        return None

    if re.fullmatch(r"[ABCD]", clean, flags=re.IGNORECASE):
        return [clean.upper()]

    for pattern in FINAL_CHOICE_PATTERNS:
        matches = pattern.findall(clean)
        if matches:
            return [str(matches[-1]).upper()]

    standalone = CHOICE_RE.findall(clean)
    if standalone:
        return [standalone[-1].upper()]
    return None


def _normalize(s: str) -> str:
    return s.strip().lower().rstrip(".,!?;: ")


def _extract_plain_text_answer(solution_str: str) -> list[str] | None:
    """Return a concise plain-text answer for non-MCQ exact-match tasks."""
    if not solution_str:
        return None
    clean = re.sub(r"<\|[^|]+?\|>", " ", str(solution_str)).strip()
    clean = " ".join(clean.split())
    if not clean:
        return None
    match = re.search(r"(?:final\s+answer|answer|word|text)\s*(?:is|:|=)?\s*([A-Za-z0-9 _'-]+)", clean, re.IGNORECASE)
    if match:
        return [match.group(1).strip().strip(" .,!?:;\"'")]
    return [clean.strip(" .,!?:;\"'")]


def compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    **kwargs,
) -> float:
    preds = _extract_display_answers(solution_str)
    if not preds:
        preds = _extract_plain_choice(solution_str)
    if not preds:
        preds = _extract_plain_text_answer(solution_str)
    if not preds:
        return 0.0
    if isinstance(ground_truth, (list, tuple)):
        gts = {_normalize(str(g)) for g in ground_truth}
        norm_preds = {_normalize(p) for p in preds}
        return 1.0 if gts == norm_preds else 0.0
    target = _normalize(str(ground_truth))
    return 1.0 if any(_normalize(p) == target for p in preds) else 0.0


# ---------------------------------------------------------------------------
# Self-tests: run with `python rewards/cof_rl_reward.py`
# ---------------------------------------------------------------------------


def _run_self_tests():
    cases = [
        # (label, solution_str, ground_truth, expected)

        (
            "plain single-letter answer",
            "B",
            "B",
            1.0,
        ),
        (
            "plain final answer phrase",
            "After drawing the line, the answer is C.",
            "C",
            1.0,
        ),
        (
            "plain wrong single-letter answer",
            "D",
            "A",
            0.0,
        ),
        (
            "happy path",
            'sure thing <|tools_prefix|>[{"display_answers": {"answers": ["B"]}}]<|tools_suffix|>',
            "B",
            1.0,
        ),
        (
            "case + trailing punct normalization",
            '<|tools_prefix|>[{"display_answers": {"answers": ["yes."]}}]<|tools_suffix|>',
            "Yes",
            1.0,
        ),
        (
            "wrong answer",
            '<|tools_prefix|>[{"display_answers": {"answers": ["A"]}}]<|tools_suffix|>',
            "B",
            0.0,
        ),
        (
            "multiple tool blocks - take the last display_answers",
            (
                '<|tools_prefix|>[{"rotate_flip_tool": {"operation": "rotate_90"}}]<|tools_suffix|>'
                "...some more thinking..."
                '<|tools_prefix|>[{"display_answers": {"answers": ["C"]}}]<|tools_suffix|>'
            ),
            "C",
            1.0,
        ),
        (
            "no display_answers at all -> 0.0",
            '<|tools_prefix|>[{"rotate_flip_tool": {"operation": "rotate_90"}}]<|tools_suffix|>',
            "anything",
            0.0,
        ),
        (
            "malformed JSON inside tool block -> 0.0",
            '<|tools_prefix|>[{"display_answers": {"answers": ["X"]]<|tools_suffix|>',
            "X",
            0.0,
        ),
        (
            "empty solution -> 0.0",
            "",
            "X",
            0.0,
        ),
        (
            "whitespace + mixed case match",
            '<|tools_prefix|>[{"display_answers": {"answers": ["  Hello World  "]}}]<|tools_suffix|>',
            "hello world",
            1.0,
        ),
        (
            "string gt matches one of multiple answers",
            '<|tools_prefix|>[{"display_answers": {"answers": ["A", "B", "C"]}}]<|tools_suffix|>',
            "B",
            1.0,
        ),
        (
            "list gt - exact set match",
            '<|tools_prefix|>[{"display_answers": {"answers": ["a", "B"]}}]<|tools_suffix|>',
            ["A", "b"],
            1.0,
        ),
        (
            "list gt - missing element -> 0.0",
            '<|tools_prefix|>[{"display_answers": {"answers": ["A"]}}]<|tools_suffix|>',
            ["A", "B"],
            0.0,
        ),
        (
            "empty answers list -> 0.0",
            '<|tools_prefix|>[{"display_answers": {"answers": []}}]<|tools_suffix|>',
            "X",
            0.0,
        ),
    ]
    failures = 0
    for label, sol, gt, expected in cases:
        got = compute_score("cof_rl", sol, gt)
        ok = got == expected
        if not ok:
            failures += 1
        print(f"[{'OK' if ok else 'FAIL'}] {label}: got={got} expected={expected}")
    print(f"\n{len(cases) - failures}/{len(cases)} passed")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    _run_self_tests()
