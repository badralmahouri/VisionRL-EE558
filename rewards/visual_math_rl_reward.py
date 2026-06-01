"""Visual-math RL reward — v1 (regex binary), v2 (continuous+penalty), v3 (judge-only).

This file holds all three reward variants so we can replay-compare them
offline on the same v1 rollout logs. The active reward is chosen by
configs/visual_math_rl_grpo.yaml::custom_reward_function.name.

Failure-mode history:
  - v1 (regex binary {0, 1}, job 2355716): MathVision OOD +6pp at step 50 but
    DynaMath in-dist regressed -4pp. Diagnostic: format compliance climbed
    42% -> 57% while pass rate WITHIN format-compliant rollouts stayed flat at
    ~20%. Textbook format-hacking via the regex anchor.
  - v2 (continuous [-0.1, 1.0] with fake-format penalty, job 2378427): output
    length collapsed to 7.78 tokens mean by step 50. The -0.1 trap made
    "say nothing -> reward 0" reward-dominant over "try -> risk -0.1".

v3 = pure binary Qwen3-32B LLM-judge reward — REPLACES regex entirely.
  - YES from judge -> 1.0; NO / parse-error / empty output / judge unavailable -> 0.0
  - No regex in the reward path -> removes v1 format-hacking by construction.
  - No -0.1 penalty -> removes v2 "say nothing is safer" trap.
  - Same Qwen3-32B judge already used for eval scoring in
    inference/llm_judge_math.py -> aligns training reward with eval metric.

Wiring in configs/visual_math_rl_grpo.yaml:
    custom_reward_function:
      path: <abs>/rewards/visual_math_rl_reward.py
      name: compute_score_v3   # or compute_score_v2 / compute_score_v1 to replay

LLM judge endpoint:
    REQUIRED for v3: set VISUAL_MATH_JUDGE_URL=http://host:port/v1
    OPTIONAL for v2: same env var; v2 uses judge only as fallback when regex
        extraction is ambiguous.
"""

from __future__ import annotations

import json
import os
import re
import urllib.request

# ---------------------------------------------------------------------------
# Extraction regexes (priority-ordered)
# ---------------------------------------------------------------------------

# 1. Anchored phrase + answer token (the "intended" format the model is asked to use).
#    Captures number OR single letter A-E. Case-insensitive on the anchor.
#    We use the LAST anchor match so "answer is 70. Wait, actually 35." -> 35.
_ANCHOR_RE = re.compile(
    r"(?:final\s+answer|the\s+answer\s+is|answer\s*[:=])\s*"
    r"\$?\\?boxed\{?(-?\d+(?:\.\d+)?|[A-Ea-e])\}?\$?"
    r"|(?:final\s+answer|the\s+answer\s+is|answer\s*[:=])\s*"
    r"\$?(-?\d+(?:\.\d+)?|[A-Ea-e])\$?",
    re.IGNORECASE,
)

# 2. Apertus display_answers tool call.
_DISPLAY_ANSWERS_RE = re.compile(
    r'"display_answers"\s*:\s*\{\s*"answers"\s*:\s*\[\s*"?([^"\]\s][^"\]]*?)"?\s*[,\]]'
)

# 3. <answer>X</answer> tag.
_ANSWER_TAG_RE = re.compile(r"<answer>\s*([^<\n]+?)\s*</answer>", re.IGNORECASE)

# 4. Last number/letter fallback.
_LAST_TOKEN_RE = re.compile(
    r"\$?\\?boxed\{?(-?\d+(?:\.\d+)?|[A-Ea-e])\}?\$?|\b([A-E])\b|(-?\d+(?:\.\d+)?)"
)


# ---------------------------------------------------------------------------
# Extraction (now returns both the extracted token AND which path matched)
# ---------------------------------------------------------------------------


def _strip_math_decor(s: str) -> str:
    """Strip TeX/math wrappers around a captured token."""
    s = s.strip().strip("$")
    s = re.sub(r"^\\boxed\{?", "", s)
    if s.endswith("}"):
        s = s[:-1]
    return s.strip()


def extract_answer(output: str) -> tuple[str | None, str]:
    """Return (extracted_token, which_path) where which_path is one of:
    'anchor', 'display_answers', 'answer_tag', 'last_token', 'none'.
    """
    if not output:
        return None, "none"

    # 1. Anchored phrase — strongest signal of "model intends this as final answer"
    anchor_matches = list(_ANCHOR_RE.finditer(output))
    if anchor_matches:
        m = anchor_matches[-1]
        token = m.group(1) or m.group(2)
        if token:
            return _strip_math_decor(token), "anchor"

    # 2. display_answers tool call
    m = _DISPLAY_ANSWERS_RE.search(output)
    if m:
        return _strip_math_decor(m.group(1)), "display_answers"

    # 3. <answer> tag
    tag_matches = _ANSWER_TAG_RE.findall(output)
    if tag_matches:
        return _strip_math_decor(tag_matches[-1]), "answer_tag"

    # 4. Last number/letter fallback (weakest)
    tok_matches = list(_LAST_TOKEN_RE.finditer(output))
    if tok_matches:
        m = tok_matches[-1]
        token = m.group(1) or m.group(2) or m.group(3)
        if token:
            return _strip_math_decor(token), "last_token"

    return None, "none"


def _normalize(s) -> str:
    """Normalize for comparison: lower-case, strip whitespace + trailing punct."""
    if s is None:
        return ""
    s = str(s).strip().lower()
    s = re.sub(r"[\s.,!?;:]+$", "", s)
    s = s.strip("$")
    s = re.sub(r"^\\boxed\{?", "", s)
    if s.endswith("}"):
        s = s[:-1]
    return s.strip()


def _try_float(s) -> float | None:
    try:
        return float(_normalize(s))
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Numeric partial credit
# ---------------------------------------------------------------------------


def numeric_partial(pred: float, gold: float) -> float:
    """Continuous numeric score with tolerance bands.

    Returns 1.0, 0.9, 0.7, 0.5, or 0.0 based on relative error.
    """
    # Relative error, normalized by max(1, |gold|) to handle gold=0 cleanly.
    denom = max(1.0, abs(gold))
    diff = abs(pred - gold) / denom
    if diff <= 1e-4:
        return 1.0
    if diff <= 0.01:
        return 0.9
    if diff <= 0.05:
        return 0.7
    if diff <= 0.10:
        return 0.5
    return 0.0


# ---------------------------------------------------------------------------
# LLM judge fallback (optional, env-var gated)
# ---------------------------------------------------------------------------

# Verbatim from inference/llm_judge_math.py:31-41 — the proven prompt already
# used for eval scoring across v1/v2 runs. Tolerates numeric rounding, accepts
# equivalent forms (1/2 = 0.5, sqrt(4) = 2, B = (B)).
_JUDGE_PROMPT_TEMPLATE = (
    "You are evaluating whether a model's answer to a math problem is "
    "semantically equivalent to the ground-truth answer. Answer with a single "
    "word: YES or NO.\n\n"
    "Question: {question}\n"
    "Gold answer: {gold}\n"
    "Model answer: {pred}\n\n"
    "Are they semantically equivalent? Allow small numeric rounding (within "
    "0.1%), accept different but equivalent forms (1/2 = 0.5, sqrt(4) = 2, "
    "B = (B)). Answer YES or NO:"
)

# Strip <think>...</think> blocks the way llm_judge_math.py does (Qwen3 thinking mode).
_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def _judge_call(
    question: str,
    pred: str,
    gold: str,
    url: str,
    timeout: float = 60.0,
) -> str | None:
    """Call Qwen3-32B sglang endpoint at `url`. Return 'YES' / 'NO' / None on error.

    Endpoint expected to be OpenAI-compatible (/v1/chat/completions style).
    Mirrors the prompt + parsing logic from inference/llm_judge_math.py so the
    training reward and the eval scorer use the same judge behaviour.
    """
    prompt = _JUDGE_PROMPT_TEMPLATE.format(
        question=(question or "").strip()[:2000],
        gold=str(gold),
        pred=str(pred)[-2000:],
    )
    body = {
        "model": "Qwen/Qwen3-32B",
        "messages": [{"role": "user", "content": prompt}],
        # llm_judge_math.py uses max_tokens=256 to leave room for thinking-mode tokens
        # before YES/NO; same here.
        "max_tokens": 256,
        "temperature": 0.0,
        # sglang accepts OpenAI-compatible extra_body for Qwen3 chat_template kwargs
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        url.rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode())
        raw = (data["choices"][0]["message"]["content"] or "").strip()
        cleaned = _THINK_RE.sub("", raw).strip().upper()
        head = cleaned[:32]
        if "YES" in head:
            return "YES"
        if "NO" in head:
            return "NO"
    except Exception:
        return None
    return None


def _maybe_judge(question: str, pred: str, gold: str) -> str | None:
    """Call judge if VISUAL_MATH_JUDGE_URL is set, else return None."""
    url = os.environ.get("VISUAL_MATH_JUDGE_URL")
    if not url:
        return None
    return _judge_call(question, pred, gold, url)


# ---------------------------------------------------------------------------
# compute_score — verl-compatible entry point
# ---------------------------------------------------------------------------


def _extract_question(extra_info) -> str:
    """Pull the question text from extra_info; tolerate missing/None."""
    if not extra_info:
        return ""
    if isinstance(extra_info, dict):
        q = extra_info.get("question")
        if q:
            return str(q)
    return ""


def compute_score_v2(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    **kwargs,
):
    """v2 reward (continuous, regex + judge fallback). Kept for offline replay
    and as the previous active reward (job 2378427, collapsed to 7-token outputs).

    Reward scale:
      1.0    exact match
      0.9    numeric within 1%
      0.7    numeric within 5% OR judge YES
      0.5    numeric within 10%
      0.0    no extractable answer (neutral)
     -0.1    format present + wrong (penalty for fake-format compliance)
    """
    solution_str = solution_str or ""

    # Handle list ground truth (multi-answer set) by checking each
    if isinstance(ground_truth, (list, tuple)):
        gts = list(ground_truth)
    else:
        gts = [ground_truth]

    extracted, path = extract_answer(solution_str)

    # Path 1: extracted nothing → neutral 0 (don't penalize a non-commit)
    if extracted is None:
        return 0.0

    n_pred = _normalize(extracted)

    # Path 2: exact string match (any gold)
    for gt in gts:
        if n_pred == _normalize(gt):
            return 1.0

    # Path 3: numeric partial credit (any gold parseable as float)
    pred_f = _try_float(extracted)
    if pred_f is not None:
        best = 0.0
        for gt in gts:
            gold_f = _try_float(gt)
            if gold_f is None:
                continue
            score = numeric_partial(pred_f, gold_f)
            if score > best:
                best = score
        if best > 0.0:
            return best

    # Path 4: LLM judge fallback (only when we have format compliance but no
    # match — this is the "ambiguous, give the judge a chance" case)
    if path in ("anchor", "display_answers", "answer_tag"):
        gt_str = ", ".join(str(g) for g in gts)
        verdict = _maybe_judge(_extract_question(extra_info), solution_str, gt_str)
        if verdict == "YES":
            return 0.7
        # If judge says NO OR judge unavailable (returned None), fall through

    # Path 5: format present (anchor / display_answers / answer_tag) but wrong
    # → penalize "fake format compliance"
    if path in ("anchor", "display_answers", "answer_tag"):
        return -0.1

    # Path 6: extraction from last_token (weakest) and wrong → neutral 0
    return 0.0


def compute_score_v3(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    **kwargs,
):
    """v3 reward — PURE binary Qwen3-32B judge. No regex in the reward path.

      1.0  judge says YES
      0.0  anything else (NO, judge parse error, empty output, judge down)

    Returns a BARE FLOAT (1.0 / 0.0), matching compute_score_v2's contract.

    Why not a {"score":..., "extra_info":{...}} dict: verl's "naive" reward
    manager folds every non-"score" key of a dict return into
    reward_extra_infos_dict, then np.means each variable in
    _validate -> process_validation_metrics. A nested "extra_info" value is a
    dict (not a scalar/str), so it crashed val aggregation with
    "unsupported operand type(s) for /: 'dict' and 'int'" (job 2435505).
    A bare float keeps only the numeric score in that dict. The v1/v2-vs-v3
    regex comparison is recomputed OFFLINE from the rollout logs
    (rollout_data_dir + _offline_replay), so no in-band side-logging is needed.

    REQUIRES VISUAL_MATH_JUDGE_URL env var. Without it, raises explicitly
    rather than silently mis-rewarding.
    """
    url = os.environ.get("VISUAL_MATH_JUDGE_URL")
    if not url:
        raise RuntimeError(
            "compute_score_v3 requires VISUAL_MATH_JUDGE_URL pointing at a "
            "Qwen3-32B sglang /v1 endpoint."
        )

    text = (solution_str or "").strip()
    if not text:
        return 0.0

    question = _extract_question(extra_info)
    # Normalize ground_truth to a single string for the judge prompt.
    if isinstance(ground_truth, (list, tuple)):
        gold = ", ".join(str(g) for g in ground_truth)
    else:
        gold = str(ground_truth)

    verdict = _judge_call(question, text, gold, url)
    return 1.0 if verdict == "YES" else 0.0




# ---------------------------------------------------------------------------
# Self-tests + offline replay
# ---------------------------------------------------------------------------

# Disable judge for unit tests (deterministic).
def _no_judge(*a, **k):
    return None


_CASES = [
    # ---- back-compat: previous 24 cases (binary outcomes still pass) -----
    ("anchored numeric exact", "Step 1: ... Final answer: 70", "70", 1.0),
    ("anchored MC letter exact", "I think it's option B. The answer is B", "B", 1.0),
    ("anchored lowercase letter exact", "answer: b", "B", 1.0),
    ("anchored with period exact", "The answer is 70.", "70", 1.0),
    ("anchored TeX $...$ exact", "Final answer: $70$", "70", 1.0),
    ("anchored \\boxed{} exact", "Final answer: \\boxed{70}", "70", 1.0),
    ("model corrects itself — last anchor wins",
     "The answer is 70. Wait, actually the answer is 35.", "35", 1.0),
    ("model corrects itself — wrong if we picked the first",
     "The answer is 70. Wait, actually the answer is 35.", "70", -0.1),
    ("display_answers tool call exact",
     '<|tools_prefix|>[{"display_answers": {"answers": ["42"]}}]<|tools_suffix|>',
     "42", 1.0),
    ("<answer> tag fallback exact",
     "Looking at the diagram... <answer>56</answer>", "56", 1.0),
    ("last-number fallback no anchor exact",
     "The maximum is reached at x=3, giving y=8.4", "8.4", 1.0),
    ("last-letter fallback no anchor exact",
     "We compare options A, B, and C. C looks right.", "C", 1.0),
    ("empty output → 0.0", "", "X", 0.0),
    ("malformed output (no numbers/letters) → 0.0", "I don't know.", "X", 0.0),
    ("gold normalization — trailing period", "Final answer: 70", "70.", 1.0),
    ("gold normalization — case for letter", "Final answer: b", "B", 1.0),
    ("list gold — predicted matches one (single-element list)",
     "Final answer: A", ["A"], 1.0),
    ("list gold — predicted matches some", "Final answer: B", ["A", "B"], 1.0),
    ("anchored answer with $ around exact", "Answer: $42$", "42", 1.0),
    # ---- v2 NEW: numeric partial credit -------------------------------------
    ("numeric exact (5 vs 5)", "Final answer: 5", "5", 1.0),
    ("numeric within 1% (5.04 vs 5)", "Final answer: 5.04", "5", 0.9),
    ("numeric within 5% (5.2 vs 5)", "Final answer: 5.2", "5", 0.7),
    ("numeric within 10% (5.4 vs 5)", "Final answer: 5.4", "5", 0.5),
    ("numeric far off (7 vs 5, format present) → −0.1",
     "Final answer: 7", "5", -0.1),
    ("numeric far off (70 vs 5, format present) → −0.1",
     "Final answer: 70", "5", -0.1),
    ("numeric within 10% with large gold (90 vs 100)",
     "Final answer: 90", "100", 0.5),
    ("numeric exact from last-token fallback",
     "I think it's 5.", "5", 1.0),
    ("numeric far from last-token fallback → 0.0 (no fake format penalty here)",
     "Hmm, perhaps 7?", "5", 0.0),
    # ---- v2 NEW: format-no-content penalty ----------------------------------
    ("wrong MC letter with format → −0.1",
     "Step 1: think. Final answer: B", "A", -0.1),
    ("wrong number with format → −0.1",
     "Final answer: 100", "50", -0.1),
    ("right answer but no format → 1.0 via last_token",
     "Detailed reasoning... the value is 50.", "50", 1.0),
    # ---- v2 NEW: extraction edge cases ---------------------------------------
    ("anchor missing but \\boxed{} present (last_token catches it)",
     "After computation, \\boxed{8}.", "8", 1.0),
    ("multiple numbers in output, take last (= correct)",
     "First we had 5, then 10, but the answer is 15.", "15", 1.0),
    ("ratio gold (not numeric) wrong extraction → -0.1 (format wrong)",
     "Final answer: 3", "3:2:4", -0.1),
    ("ratio gold + last-token fallback no penalty", "ratio 3", "3:2:4", 0.0),
    ("negative number exact", "Final answer: -4", "-4", 1.0),
    ("decimal numeric exact (0.3333 vs 0.3333)",
     "Final answer: 0.3333", "0.3333", 1.0),
    ("decimal numeric close (0.333 vs 0.333333)",
     "Final answer: 0.333", "0.333333", 0.9),
]


def _run_self_tests() -> int:
    """Print pass/fail for each case. Returns count of failures (0 = all pass)."""
    # Force judge OFF for deterministic test
    saved = os.environ.pop("VISUAL_MATH_JUDGE_URL", None)
    try:
        failures = 0
        for label, sol, gt, expected in _CASES:
            got = compute_score_v2("visual_math_rl", sol, gt)
            ok = abs(got - expected) < 1e-6
            if not ok:
                failures += 1
            print(f"[{'OK  ' if ok else 'FAIL'}] {label}: got={got} expected={expected}")
        print(f"\n{len(_CASES) - failures}/{len(_CASES)} passed")
        return failures
    finally:
        if saved is not None:
            os.environ["VISUAL_MATH_JUDGE_URL"] = saved


# ---------------------------------------------------------------------------
# Offline replay — compare v1 (binary) vs v2 (continuous) on rollout_logs
# ---------------------------------------------------------------------------


def _compute_score_v1(sol: str, gt) -> float:
    """The v1 binary reward (what we trained on in job 2355716). For comparison."""
    extracted, _ = extract_answer(sol)
    if extracted is None:
        return 0.0
    n_pred = _normalize(extracted)
    if isinstance(gt, (list, tuple)):
        gts = list(gt)
    else:
        gts = [gt]
    if any(n_pred == _normalize(g) for g in gts):
        return 1.0
    pred_f = _try_float(extracted)
    if pred_f is not None:
        for g in gts:
            gold_f = _try_float(g)
            if gold_f is not None and abs(pred_f - gold_f) <= 1e-4 * max(1.0, abs(gold_f)):
                return 1.0
    return 0.0


def _offline_replay(rollout_dir: str, run_v3: bool = False, v3_sample: int = 50) -> None:
    """Re-score rollouts with v1, v2 (and optionally v3 if judge URL set).

    v3 hits the judge for every row — expensive. By default we sample
    `v3_sample` rows per step to keep this O(minutes) not O(hours). Set
    v3_sample=0 to score all rows.
    """
    from pathlib import Path
    import statistics
    import random

    rd = Path(rollout_dir)
    if not rd.exists():
        print(f"  rollout_dir not found: {rd}")
        return

    steps = sorted(int(p.stem) for p in rd.glob("*.jsonl"))
    if not steps:
        print("  no rollout files found")
        return
    sample_steps = [s for s in steps if s in (1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100)]

    if run_v3 and not os.environ.get("VISUAL_MATH_JUDGE_URL"):
        print("  WARN: --v3 requested but VISUAL_MATH_JUDGE_URL not set; skipping v3 column")
        run_v3 = False

    header = f"\n{'step':>5} {'n':>4} {'v1_mean':>8} {'v1_pass':>7} {'v2_mean':>8} {'v2_pos':>7} {'v2_pen':>7}"
    if run_v3:
        header += f" {'v3_n':>5} {'v3_mean':>8} {'v3_pass':>8}"
    print(header)
    print("-" * (len(header) - 1))

    rng = random.Random(42)
    for step in sample_steps:
        rows = []
        with open(rd / f"{step}.jsonl") as f:
            for line in f:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        if not rows:
            continue
        v1s, v2s = [], []
        for r in rows:
            out = r.get("output", "")
            gt_field = r.get("gts") or r.get("ground_truth") or r.get("gt")
            gt = gt_field[0] if isinstance(gt_field, list) and gt_field else gt_field
            if gt is None:
                continue
            v1s.append(_compute_score_v1(out, gt))
            v2s.append(compute_score_v2("dynamath", out, gt))
        if not v1s:
            continue

        v1_mean = statistics.mean(v1s)
        v1_pass = sum(1 for x in v1s if x >= 0.5) / len(v1s)
        v2_mean = statistics.mean(v2s)
        v2_pos = sum(1 for x in v2s if x > 0) / len(v2s)
        v2_pen = sum(1 for x in v2s if x < 0) / len(v2s)
        line = (f"{step:>5} {len(v1s):>4} {v1_mean:>8.3f} {v1_pass:>6.1%} {v2_mean:>8.3f} "
                f"{v2_pos:>6.1%} {v2_pen:>6.1%}")

        if run_v3:
            # Sample for v3 (judge calls are slow)
            target = list(range(len(rows)))
            if v3_sample > 0 and len(target) > v3_sample:
                target = rng.sample(target, v3_sample)
            v3s = []
            for i in target:
                r = rows[i]
                out = r.get("output", "")
                gt_field = r.get("gts") or r.get("ground_truth") or r.get("gt")
                gt = gt_field[0] if isinstance(gt_field, list) and gt_field else gt_field
                question = r.get("question", "") or r.get("prompt", "")
                if gt is None:
                    continue
                result = compute_score_v3(
                    "dynamath", out, gt,
                    extra_info={"question": question},
                )
                v3s.append(result["score"] if isinstance(result, dict) else result)
            if v3s:
                v3_mean = statistics.mean(v3s)
                v3_pass = sum(1 for x in v3s if x >= 0.5) / len(v3s)
                line += f" {len(v3s):>5} {v3_mean:>8.3f} {v3_pass:>7.1%}"
            else:
                line += f" {'-':>5} {'-':>8} {'-':>8}"
        print(line, flush=True)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "replay":
        rollout_dir = sys.argv[2] if len(sys.argv) > 2 else "slurm/rollout_logs/visual_math_rl"
        run_v3 = "--v3" in sys.argv
        # default v3 sample size = 50 rows per step; --v3-all means score every row
        v3_sample = 0 if "--v3-all" in sys.argv else 50
        print(f"=== Offline replay on {rollout_dir} (v3={run_v3}, v3_sample={v3_sample}) ===")
        _offline_replay(rollout_dir, run_v3=run_v3, v3_sample=v3_sample)
    else:
        sys.exit(_run_self_tests())
