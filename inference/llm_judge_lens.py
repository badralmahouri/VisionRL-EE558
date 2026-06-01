"""Qwen3-32B judge for Lens-style knowledge-VQA outputs.

Cloned from `llm_judge_math.py` with Lens-specific changes:
  - Multi-answer gold: each row has `gold_answers` (list); model answer is correct
    if it matches ANY of them.
  - Three-way verdict: YES / PARTIAL / NO (vs math's two-way YES/NO).
  - Per-Lens-capability breakdown in the summary.

Input row schema (JSONL):
  name, source, question, problem, answer, gold_answers, lens_category,
  lens_capability, png, model_output, n_ibq_tokens, image_size, ...

Outputs:
  <input>.judged.jsonl    same rows + verdict + judge_raw
  <input>.summary.json    {accuracy_yes_partial, accuracy_yes_only, per_lens_capability, ...}

Usage:
    python inference/llm_judge_lens.py \\
        --input  evaluation/lens_probe_outputs/lens_probe_${SLURM_JOB_ID}.jsonl \\
        --judge-url http://localhost:30100/v1 \\
        --max-workers 8
"""

import argparse
import concurrent.futures as cf
import json
import re
import sys
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

JUDGE_MODEL = "Qwen/Qwen3-32B"

JUDGE_PROMPT_TEMPLATE = (
    "You are evaluating whether a model's answer to an image-based question is "
    "correct. The image is described by a question, several gold answers from "
    "human annotators, and the model's answer.\n\n"
    "Question: {question}\n"
    "Gold answers (any one of these counts as correct, case-insensitive): {gold_list}\n"
    "Model answer: {pred}\n\n"
    "Decide one of three verdicts:\n"
    "  YES   — model answer matches one of the gold answers in meaning (paraphrases OK; e.g., \"frosting\" matches \"icing\")\n"
    "  PARTIAL — model answer is in the right category but less specific than gold (e.g., \"tree\" for gold \"oak tree\"), OR more specific than gold but still consistent (e.g., \"golden retriever\" for gold \"dog\"), OR refers to the right entity by a different attribute\n"
    "  NO    — model answer is wrong, off-topic, or just says \"unknown\"\n\n"
    "Respond with a single word: YES, PARTIAL, or NO."
)

THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def make_client(base_url: str):
    """OpenAI HTTP client with verify=False (verl_env SSL_CERT_FILE bug)."""
    import httpx
    from openai import OpenAI
    http_client = httpx.Client(verify=False, timeout=httpx.Timeout(120.0))
    return OpenAI(base_url=base_url, api_key="EMPTY", http_client=http_client)


def wait_for_server(base_url: str, timeout_s: int = 600):
    health_url = base_url.rstrip("/").rsplit("/v1", 1)[0] + "/health"
    start = time.time()
    last_err = None
    while time.time() - start < timeout_s:
        try:
            with urllib.request.urlopen(health_url, timeout=5) as resp:
                if resp.status == 200:
                    print(f"  judge server READY at {base_url}", flush=True)
                    return
        except Exception as e:
            last_err = e
        time.sleep(5)
    raise RuntimeError(f"judge at {base_url} not ready in {timeout_s}s: {last_err}")


def judge_one(client, question: str, gold_list: list, pred: str) -> tuple[str, str]:
    """Returns (verdict, raw_response). Verdict in {YES, PARTIAL, NO, PARSE_ERROR}."""
    # Deduplicate gold list for cleaner prompt
    seen = set()
    unique_gold = []
    for g in gold_list:
        g = str(g).strip()
        if g and g.lower() not in seen:
            seen.add(g.lower())
            unique_gold.append(g)

    prompt = JUDGE_PROMPT_TEMPLATE.format(
        question=question,
        gold_list=", ".join(f'"{g}"' for g in unique_gold) if unique_gold else '""',
        pred=pred,
    )
    try:
        response = client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=128,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
    except Exception as e:
        return "PARSE_ERROR", f"judge_call_error: {type(e).__name__}: {e}"
    raw = response.choices[0].message.content or ""
    cleaned = THINK_RE.sub("", raw).strip().upper()
    head = cleaned[:64]

    # Check in order of specificity: PARTIAL before YES (PARTIAL contains the substring "AL"
    # but YES doesn't; the issue is whether the word PARTIAL appears)
    if re.search(r"\bPARTIAL\b", head):
        return "PARTIAL", raw[:200]
    if re.search(r"\bYES\b", head):
        return "YES", raw[:200]
    if re.search(r"\bNO\b", head):
        return "NO", raw[:200]
    return "PARSE_ERROR", raw[:200]


def judge_all(rows: list, judge_url: str, max_workers: int) -> list:
    client = make_client(judge_url)

    def _do(idx_row):
        idx, row = idx_row
        verdict, raw = judge_one(
            client,
            row.get("question", ""),
            row.get("gold_answers", []) or [str(row.get("answer", ""))],
            str(row.get("model_output", "")),
        )
        return idx, verdict, raw

    judged = [None] * len(rows)
    completed = 0
    t0 = time.time()
    verdict_counts = defaultdict(int)

    with cf.ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_do, (i, row)) for i, row in enumerate(rows)]
        for fut in cf.as_completed(futures):
            i, verdict, raw = fut.result()
            row = dict(rows[i])
            row["verdict"] = verdict
            row["judge_raw"] = raw
            judged[i] = row
            verdict_counts[verdict] += 1
            completed += 1
            if completed % 20 == 0 or completed == len(rows):
                rate = completed / (time.time() - t0)
                print(
                    f"  judge [{completed}/{len(rows)}] {rate:.2f} rows/s "
                    f"verdicts={dict(verdict_counts)}",
                    flush=True,
                )

    return judged


def summarize(judged: list) -> dict:
    n = len(judged)
    n_yes = sum(1 for r in judged if r["verdict"] == "YES")
    n_partial = sum(1 for r in judged if r["verdict"] == "PARTIAL")
    n_no = sum(1 for r in judged if r["verdict"] == "NO")
    n_parse = sum(1 for r in judged if r["verdict"] == "PARSE_ERROR")
    denom = n_yes + n_partial + n_no
    accuracy_yes = n_yes / denom if denom else 0.0
    accuracy_yes_partial = (n_yes + n_partial) / denom if denom else 0.0

    # Per Lens-capability
    by_cap = defaultdict(lambda: {"yes": 0, "partial": 0, "no": 0, "parse_error": 0})
    for r in judged:
        cap = r.get("lens_capability") or "unknown"
        v = r["verdict"]
        if v == "YES":
            by_cap[cap]["yes"] += 1
        elif v == "PARTIAL":
            by_cap[cap]["partial"] += 1
        elif v == "NO":
            by_cap[cap]["no"] += 1
        else:
            by_cap[cap]["parse_error"] += 1
    per_cap = {}
    for cap, c in by_cap.items():
        denom = c["yes"] + c["partial"] + c["no"]
        per_cap[cap] = {
            **c,
            "n": denom + c["parse_error"],
            "accuracy_yes": (c["yes"] / denom) if denom else 0.0,
            "accuracy_yes_partial": ((c["yes"] + c["partial"]) / denom) if denom else 0.0,
        }

    # Per OK-VQA category
    by_cat = defaultdict(lambda: {"yes": 0, "partial": 0, "no": 0, "parse_error": 0})
    for r in judged:
        cat = r.get("lens_category") or "unknown"
        v = r["verdict"]
        if v == "YES":
            by_cat[cat]["yes"] += 1
        elif v == "PARTIAL":
            by_cat[cat]["partial"] += 1
        elif v == "NO":
            by_cat[cat]["no"] += 1
        else:
            by_cat[cat]["parse_error"] += 1
    per_cat = {}
    for cat, c in by_cat.items():
        denom = c["yes"] + c["partial"] + c["no"]
        per_cat[cat] = {
            **c,
            "n": denom + c["parse_error"],
            "accuracy_yes": (c["yes"] / denom) if denom else 0.0,
            "accuracy_yes_partial": ((c["yes"] + c["partial"]) / denom) if denom else 0.0,
        }

    return {
        "accuracy_yes_only": accuracy_yes,
        "accuracy_yes_partial": accuracy_yes_partial,
        "n_total": n,
        "n_yes": n_yes,
        "n_partial": n_partial,
        "n_no": n_no,
        "n_parse_error": n_parse,
        "per_lens_capability": per_cap,
        "per_lens_category": per_cat,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--judge-url", default="http://localhost:30100/v1")
    ap.add_argument("--max-workers", type=int, default=8)
    ap.add_argument("--server-wait-s", type=int, default=600)
    args = ap.parse_args()

    in_path = Path(args.input)
    judged_path = in_path.with_suffix(in_path.suffix + ".judged.jsonl")
    summary_path = in_path.with_suffix(in_path.suffix + ".summary.json")

    rows = []
    with open(in_path) as f:
        for line in f:
            rows.append(json.loads(line))
    print(f"Loaded {len(rows)} rows from {in_path}", flush=True)

    print(f"Waiting for judge at {args.judge_url} ...", flush=True)
    wait_for_server(args.judge_url, args.server_wait_s)

    judged = judge_all(rows, args.judge_url, args.max_workers)

    with open(judged_path, "w", encoding="utf-8") as f:
        for r in judged:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(judged)} judged rows to {judged_path}", flush=True)

    summary = summarize(judged)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Final accuracy ===", flush=True)
    print(
        f"  YES-only          = {summary['accuracy_yes_only']*100:.2f}%  ({summary['n_yes']}/{summary['n_total']})",
        flush=True,
    )
    print(
        f"  YES+PARTIAL       = {summary['accuracy_yes_partial']*100:.2f}%  ({summary['n_yes'] + summary['n_partial']}/{summary['n_total']})",
        flush=True,
    )
    print(f"  parse errors      = {summary['n_parse_error']}", flush=True)
    print(f"  summary -> {summary_path}", flush=True)

    print("\n=== Per Lens-capability ===", flush=True)
    for cap, c in sorted(summary["per_lens_capability"].items(), key=lambda kv: -kv[1]["accuracy_yes_partial"]):
        print(f"  {cap:30s}  N={c['n']:3d}  YES={c['accuracy_yes']*100:.0f}%  YES+PARTIAL={c['accuracy_yes_partial']*100:.0f}%", flush=True)


if __name__ == "__main__":
    main()
