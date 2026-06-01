"""Reward function for Lens-search RL.

Apertus emits tool calls in its native format:
    <|tools_prefix|>[{"display_answers": {"answers": ["<X>", ...]}}]<|tools_suffix|>

We pull the LAST `display_answers` call from the rollout's solution string and
string-match its `answers` list against `ground_truth`. The reward shape:

  - Tier 1 (default, `enable_tool_bonus=False`):
        R = 1.0 if normalized prediction matches ground_truth else 0.0
  - Tier 2 (`enable_tool_bonus=True`, DeepEyes form):
        R = R_acc + 0.05 * 𝕀[R_acc > 0 and lens_search_called]
    The bonus is only awarded if the final answer is also correct, which is the
    anti-reward-hacking lever published in DeepEyes (Liu et al., 2025-05).

Pattern follows rewards/cof_rl_reward.py (which we don't extend in-place
because we need different tool-call detection logic).

Wired into verl via:
    reward.custom_reward_function:
      path: <abs>/rewards/lens_search_rl_reward.py
      name: compute_score
"""
from __future__ import annotations

import json
import os


def _find_json_arrays(text: str) -> list[str]:
    r"""Return every balanced top-level [...] substring (handles nesting).

    solution_str arrives with <|tools_prefix|>/<|tools_suffix|> already stripped
    (the reward manager decodes with skip_special_tokens=True, and tokens 71/72
    are flagged special). The JSON arrays the model emits —
    [{"lens_search": {...}}] and [{"display_answers": {...}}] — survive intact,
    so we scan for balanced brackets instead of the (now-absent) special-token
    wrapper. A non-greedy `\[.*?\]` regex cannot be used: it stops at the first
    ']' inside a nested answers list. Tool-result blocks like
    [Search results: ...] get captured too but fail json.loads and are skipped.
    """
    if not text:
        return []
    arrays: list[str] = []
    depth = 0
    start: int | None = None
    for i, ch in enumerate(text):
        if ch == "[":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "]" and depth > 0:
            depth -= 1
            if depth == 0 and start is not None:
                arrays.append(text[start : i + 1])
                start = None
    return arrays


def _extract_display_answers(solution_str: str) -> list[str] | None:
    """Return the `answers` list of the LAST display_answers call, or None."""
    for arr in reversed(_find_json_arrays(solution_str)):
        try:
            calls = json.loads(arr)
        except json.JSONDecodeError:
            continue
        if not isinstance(calls, list) or not calls:
            continue
        call = calls[-1]
        if not isinstance(call, dict):
            continue
        args = call.get("display_answers")
        if not isinstance(args, dict) or not isinstance(args.get("answers"), list):
            continue
        return [str(a) for a in args["answers"]]
    return None


def _lens_search_was_called(solution_str: str) -> bool:
    """Return True if any JSON array contains a lens_search call."""
    for arr in _find_json_arrays(solution_str):
        try:
            calls = json.loads(arr)
        except json.JSONDecodeError:
            continue
        if not isinstance(calls, list):
            continue
        for call in calls:
            if isinstance(call, dict) and "lens_search" in call:
                return True
    return False


def _normalize(s: str) -> str:
    return s.strip().lower().rstrip(".,!?;: ")


def compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    **kwargs,
) -> float:
    preds = _extract_display_answers(solution_str)
    if not preds:
        return 0.0

    # Correctness term
    if isinstance(ground_truth, (list, tuple)):
        gts = {_normalize(str(g)) for g in ground_truth}
        norm_preds = {_normalize(p) for p in preds}
        r_acc = 1.0 if gts == norm_preds else 0.0
    else:
        target = _normalize(str(ground_truth))
        r_acc = 1.0 if any(_normalize(p) == target for p in preds) else 0.0

    # Optional tier-2 tool-bonus (DeepEyes form): only if correct AND tool was called
    if os.getenv("LENS_REWARD_TIER", "1").strip() == "2":
        if r_acc > 0 and _lens_search_was_called(solution_str):
            return r_acc + 0.05

    return r_acc


# ---------------------------------------------------------------------------
# Self-tests: run with `python rewards/lens_search_rl_reward.py`
# ---------------------------------------------------------------------------


def _run_self_tests():
    cases = [
        # (label, solution_str, ground_truth, expected_tier1)
        (
            "happy path — correct answer, no tool call",
            '<|tools_prefix|>[{"display_answers": {"answers": ["namus"]}}]<|tools_suffix|>',
            "namus",
            1.0,
        ),
        (
            "case + trailing punct normalization",
            '<|tools_prefix|>[{"display_answers": {"answers": ["Namus."]}}]<|tools_suffix|>',
            "namus",
            1.0,
        ),
        (
            "wrong answer",
            '<|tools_prefix|>[{"display_answers": {"answers": ["foobar"]}}]<|tools_suffix|>',
            "namus",
            0.0,
        ),
        (
            "happy path — correct + lens_search called",
            (
                '<|tools_prefix|>[{"lens_search": {"query": "What is this?"}}]<|tools_suffix|>'
                "...tool result..."
                '<|tools_prefix|>[{"display_answers": {"answers": ["namus"]}}]<|tools_suffix|>'
            ),
            "namus",
            1.0,
        ),
        (
            "no display_answers -> 0.0",
            '<|tools_prefix|>[{"lens_search": {"query": "?"}}]<|tools_suffix|>',
            "namus",
            0.0,
        ),
        (
            "malformed JSON -> 0.0",
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
            "list gt - set match",
            '<|tools_prefix|>[{"display_answers": {"answers": ["a", "B"]}}]<|tools_suffix|>',
            ["A", "b"],
            1.0,
        ),
        # --- BARE format: real rollouts arrive with special tokens stripped ---
        (
            "bare (stripped) two-call rollout -> 1.0",
            (
                "The user is asking about the logo. I'll look it up."
                '[{"lens_search": {"query": "what company is this logo?"}}]'
                "[Search results:\n1. Jardine Matheson Holdings\n2. JM Group]"
                "Based on the search results, the answer is Jardine Matheson."
                '[{"display_answers": {"answers": ["Jardine Matheson"]}}]'
            ),
            "jardine matheson",
            1.0,
        ),
        (
            "bare nested answers list (non-greedy regex would fail here) -> 1.0",
            '[{"display_answers": {"answers": ["aquarius"]}}]',
            "Aquarius",
            1.0,
        ),
        (
            "bare wrong answer -> 0.0",
            (
                '[{"lens_search": {"query": "?"}}]'
                "[Search results:\n1. Proxyium]"
                '[{"display_answers": {"answers": ["Proxyium"]}}]'
            ),
            "scorpios",
            0.0,
        ),
        (
            "bare tool-result block alone (no display_answers) -> 0.0",
            '[{"lens_search": {"query": "?"}}][Search results:\n1. X\n2. Y]',
            "X",
            0.0,
        ),
    ]
    failures = 0
    for label, sol, gt, expected in cases:
        got = compute_score("lens_search_rl", sol, gt)
        ok = got == expected
        if not ok:
            failures += 1
        print(f"[{'OK' if ok else 'FAIL'}] {label}: got={got} expected={expected}")
    print(f"\n[Tier 1] {len(cases) - failures}/{len(cases)} passed")

    # Tier-2 spot checks
    os.environ["LENS_REWARD_TIER"] = "2"
    tier2_cases = [
        (
            "tier-2: correct + tool called -> 1.05",
            (
                '<|tools_prefix|>[{"lens_search": {"query": "?"}}]<|tools_suffix|>'
                '<|tools_prefix|>[{"display_answers": {"answers": ["X"]}}]<|tools_suffix|>'
            ),
            "X",
            1.05,
        ),
        (
            "tier-2: correct + no tool call -> 1.0 (no bonus)",
            '<|tools_prefix|>[{"display_answers": {"answers": ["X"]}}]<|tools_suffix|>',
            "X",
            1.0,
        ),
        (
            "tier-2: wrong + tool called -> 0.0 (no bonus)",
            (
                '<|tools_prefix|>[{"lens_search": {"query": "?"}}]<|tools_suffix|>'
                '<|tools_prefix|>[{"display_answers": {"answers": ["wrong"]}}]<|tools_suffix|>'
            ),
            "X",
            0.0,
        ),
    ]
    t2_fail = 0
    for label, sol, gt, expected in tier2_cases:
        got = compute_score("lens_search_rl", sol, gt)
        ok = abs(got - expected) < 1e-9
        if not ok:
            t2_fail += 1
        print(f"[{'OK' if ok else 'FAIL'}] {label}: got={got} expected={expected}")
    print(f"\n[Tier 2] {len(tier2_cases) - t2_fail}/{len(tier2_cases)} passed")

    del os.environ["LENS_REWARD_TIER"]
    if failures or t2_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    _run_self_tests()
