"""Qwen3-32B LLM-judge over a baseline JSONL of math predictions.

Lifted from data_prep/prepare_calc_sft_distill.py (judge phase). Strips the
distillation-trajectory glue; consumes a flat JSONL with at minimum
{index, ground_truth, model_output, question} per row, calls the judge
concurrently, and writes:
  <input>.judged.jsonl        same rows + verdict + judge_raw
  <input>.summary.json        {accuracy, n_yes, n_no, n_parse_error,
                                per_subject_accuracy, per_level_accuracy}

Usage (judge server already running on localhost:30100):
    python inference/llm_judge_math.py \\
        --input  evaluation/baseline_outputs/calc_mathvision200_baseline.jsonl \\
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

# Verbatim from data_prep/prepare_calc_sft_distill.py:299-309
JUDGE_PROMPT_TEMPLATE = (
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


def judge_one(client, question: str, gold: str, pred: str) -> tuple[str, str]:
    """Returns (verdict, raw_response). Verdict in {YES, NO, PARSE_ERROR}."""
    prompt = JUDGE_PROMPT_TEMPLATE.format(question=question, gold=gold, pred=pred)
    try:
        response = client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=256,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
    except Exception as e:
        return "PARSE_ERROR", f"judge_call_error: {type(e).__name__}: {e}"
    raw = response.choices[0].message.content or ""
    cleaned = THINK_RE.sub("", raw).strip().upper()
    head = cleaned[:32]
    if "YES" in head:
        return "YES", raw[:200]
    if "NO" in head:
        return "NO", raw[:200]
    return "PARSE_ERROR", raw[:200]


def judge_all(
    rows: list[dict],
    judge_url: str,
    max_workers: int,
) -> list[dict]:
    client = make_client(judge_url)

    def _do(idx_row):
        idx, row = idx_row
        verdict, raw = judge_one(
            client,
            row.get("question", ""),
            str(row.get("ground_truth", "")),
            str(row.get("model_output", "")),
        )
        return idx, verdict, raw

    judged: list[dict] = [None] * len(rows)  # type: ignore
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


def summarize(judged: list[dict]) -> dict:
    n_yes = sum(1 for r in judged if r["verdict"] == "YES")
    n_no = sum(1 for r in judged if r["verdict"] == "NO")
    n_parse = sum(1 for r in judged if r["verdict"] == "PARSE_ERROR")
    denom = n_yes + n_no
    accuracy = n_yes / denom if denom > 0 else 0.0

    by_subject = defaultdict(lambda: {"yes": 0, "no": 0, "parse_error": 0})
    for r in judged:
        s = r.get("subject") or "unknown"
        if r["verdict"] == "YES":
            by_subject[s]["yes"] += 1
        elif r["verdict"] == "NO":
            by_subject[s]["no"] += 1
        else:
            by_subject[s]["parse_error"] += 1
    per_subject = {}
    for s, c in by_subject.items():
        d = c["yes"] + c["no"]
        per_subject[s] = {
            **c,
            "n": c["yes"] + c["no"] + c["parse_error"],
            "accuracy": (c["yes"] / d) if d > 0 else 0.0,
        }

    by_level = defaultdict(lambda: {"yes": 0, "no": 0, "parse_error": 0})
    for r in judged:
        lv = r.get("level") or "unknown"
        if r["verdict"] == "YES":
            by_level[lv]["yes"] += 1
        elif r["verdict"] == "NO":
            by_level[lv]["no"] += 1
        else:
            by_level[lv]["parse_error"] += 1
    per_level = {}
    for lv, c in by_level.items():
        d = c["yes"] + c["no"]
        per_level[lv] = {
            **c,
            "n": c["yes"] + c["no"] + c["parse_error"],
            "accuracy": (c["yes"] / d) if d > 0 else 0.0,
        }

    return {
        "accuracy": accuracy,
        "n_total": len(judged),
        "n_yes": n_yes,
        "n_no": n_no,
        "n_parse_error": n_parse,
        "per_subject": per_subject,
        "per_level": per_level,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Baseline JSONL from run_calc_baseline.py")
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
        f"  accuracy = {summary['accuracy']*100:.2f}%  "
        f"(YES={summary['n_yes']}  NO={summary['n_no']}  PARSE_ERROR={summary['n_parse_error']})",
        flush=True,
    )
    print(f"  summary -> {summary_path}", flush=True)


if __name__ == "__main__":
    main()
