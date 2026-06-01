"""Build the ToolVQA-based Lens-search SFT parquets (Phase D.7.2).

This replaces the previous FVQA-based SFT (267 rows, taught JS-style tool calls
instead of Apertus delimiters — see plan file D.4 post-mortem). ToolVQA ships
explicit tool-call trajectories with real outputs, so we re-encode 2,654 rows
into Apertus delimiters without synthesizing tool results blindly.

D.7.1 locked the filter to 2,654 surviving rows:
  - 562 single-call rows where context[0].name == "ImageDescription"
  - 2,092 two-call chains where (context[0].name, context[1].name) ==
    ("ImageDescription", "GoogleSearch")

Source: `DietCoke4671/ToolVQA` on HuggingFace, Apache-2.0, 21,105 train rows.

Three operating modes:

  # 1) Dry-run: print filter cascade counts only. No download, no GPU, no I/O.
  python data_prep/prepare_lens_search_sft_toolvqa.py --dry-run

  # 2) CPU-only dry-render: skip image download + IBQ encode (uses a
  #    placeholder image-token string), but actually render via
  #    apply_chat_template + tokenize → verify tokens 71/72 appear.
  python data_prep/prepare_lens_search_sft_toolvqa.py --dry-render --limit 5

  # 3) Full encode: needs 1 GPU + HF network access; ~60-90 min for 2,654 rows.
  python data_prep/prepare_lens_search_sft_toolvqa.py
"""
from __future__ import annotations

import argparse
import collections
import io
import json
import re
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# Verbatim copy from prepare_lens_search_sft.py to avoid coupling the two files.
APERTUS_INSTRUCTION = (
    "If you need information about the entity, landmark, object, or text in the "
    "image, call the lens_search tool with a short natural-language query. Use "
    "the search results to inform your answer. Then call the display_answers "
    "tool exactly once at the end of your response, passing your final answer "
    "as the single element of the `answers` argument."
)

# Inlined from prepare_cof_rl_parse.py to keep this script torch-free in
# --dry-render mode (the cof file imports torch at module load).
APERTUS_SYSTEM = "You are a helpful assistant with access to tools."

DISPLAY_ANSWERS_TOOL = {
    "name": "display_answers",
    "description": "Display the answers to the user.",
    "parameters": {
        "type": "object",
        "properties": {
            "answers": {
                "type": "array",
                "items": {"type": "string"},
                "description": "The final answer.",
            },
        },
        "required": ["answers"],
    },
}


def _load_config(path: str) -> dict:
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def _filter_toolvqa(ds, max_tool_output_chars: int, max_answer_chars: int):
    """Return (single_call_rows, two_call_rows) plus diagnostic counts.

    Per D.7.1 evidence:
      - single_call: len(context)==1 AND context[0].name == "ImageDescription"
      - two_call:    len(context)==2 AND
                     (context[0].name, context[1].name) ==
                     ("ImageDescription", "GoogleSearch")
    Both paths also require a clean (non-empty, ≤ max_answer_chars) answer and
    non-empty outputs for each call (≤ max_tool_output_chars).
    """
    def has_clean_answer(r) -> bool:
        ans = r.get("answer") or ""
        if not isinstance(ans, str) or not ans.strip():
            return False
        if len(ans) > max_answer_chars:
            return False
        return True

    def has_clean_output(call) -> bool:
        out = call.get("output") or ""
        if not isinstance(out, str) or not out.strip():
            return False
        if len(out) > max_tool_output_chars:
            return False
        return True

    n_total = len(ds)
    ctx_lens = collections.Counter(len(r["context"]) for r in ds)

    single_call_raw = [r for r in ds if len(r["context"]) == 1]
    tool_names_single = collections.Counter(r["context"][0]["name"] for r in single_call_raw)

    single_call = [
        r for r in single_call_raw
        if r["context"][0]["name"] == "ImageDescription"
        and has_clean_answer(r)
        and has_clean_output(r["context"][0])
    ]

    two_call_raw = [r for r in ds if len(r["context"]) == 2]
    tool_pairs = collections.Counter(
        (r["context"][0]["name"], r["context"][1]["name"]) for r in two_call_raw
    )
    two_call = [
        r for r in two_call_raw
        if r["context"][0]["name"] == "ImageDescription"
        and r["context"][1]["name"] == "GoogleSearch"
        and has_clean_answer(r)
        and has_clean_output(r["context"][0])
        and has_clean_output(r["context"][1])
    ]

    diag = {
        "n_total": n_total,
        "ctx_lens": dict(sorted(ctx_lens.items())),
        "tool_names_single": dict(tool_names_single.most_common()),
        "tool_pairs_top5": dict(tool_pairs.most_common(5)),
        "single_call_kept": len(single_call),
        "two_call_kept": len(two_call),
    }
    return single_call, two_call, diag


def _make_tool_result(row) -> str:
    """Build the tool_result string per D.7.1.2 spec."""
    ctx = row["context"]
    if len(ctx) == 1:
        out = (ctx[0].get("output") or "").strip()
        return "Visual: " + out[:300]
    elif len(ctx) == 2:
        vis = (ctx[0].get("output") or "").strip()
        web = (ctx[1].get("output") or "").strip()
        return "Visual: " + vis[:200] + "\nWeb: " + web[:300]
    raise ValueError(f"Unsupported context length: {len(ctx)}")


def _build_messages(
    image_token_str: str,
    question: str,
    tool_result: str,
    answer: str,
    apertus_system: str,
    lens_search_tool: dict,
    display_answers_tool: dict,
):
    """Mirror of build_messages in prepare_lens_search_sft.py:65-120."""
    from data_prep.lens_search_common import (
        derive_query_from_question,
        synthesize_thinking_blocks,
    )

    pre_thought, post_thought = synthesize_thinking_blocks(question, answer)
    query = derive_query_from_question(question)

    return [
        {"role": "system", "content": apertus_system},
        {
            "role": "user",
            "content": f"{image_token_str}\n\n{question}\n\n{APERTUS_INSTRUCTION}",
        },
        {
            "role": "assistant",
            "content": {
                "blocks": [
                    {"type": "thoughts", "text": pre_thought},
                    {
                        "type": "tool_calls",
                        "calls": [
                            {
                                "name": "lens_search",
                                "arguments": json.dumps(
                                    {"query": query}, ensure_ascii=False
                                ),
                            }
                        ],
                    },
                ]
            },
        },
        {"role": "tool", "content": tool_result},
        {
            "role": "assistant",
            "content": {
                "blocks": [
                    {"type": "thoughts", "text": post_thought},
                    {
                        "type": "tool_calls",
                        "calls": [
                            {
                                "name": "display_answers",
                                "arguments": json.dumps(
                                    {"answers": [answer]}, ensure_ascii=False
                                ),
                            }
                        ],
                    },
                ]
            },
        },
    ]


def _sanitize_filename(s: str) -> str:
    """images/part2/000000576676.jpg → part2_000000576676.jpg"""
    s = re.sub(r"^images/", "", s)
    s = s.replace("/", "_")
    s = re.sub(r"[^A-Za-z0-9._-]", "_", s)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", default=False,
                    help="Print filter cascade counts only; no encoder runs.")
    ap.add_argument("--dry-render", action="store_true", default=False,
                    help="Render via chat template + tokenize (CPU only); "
                         "skip image download + IBQ encoding. Use a "
                         "placeholder image-token string. Validates the "
                         "smoking-gun token-71/72 check without GPU.")
    ap.add_argument("--limit", type=int, default=None,
                    help="Encode only the first N viable rows (smoke knob).")
    ap.add_argument("--out-dir", default="data_prep/lens_search")
    ap.add_argument("--val-ratio", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--config", default="configs/apertus.yaml")
    ap.add_argument("--max-tool-output-chars", type=int, default=2000)
    ap.add_argument("--max-answer-chars", type=int, default=200)
    ap.add_argument("--max-download-workers", type=int, default=16,
                    help="Parallel threads for the HF image pre-fetch. "
                         "Set to 1 to fall back to the original serial path.")
    args = ap.parse_args()

    # --- Load ToolVQA ---
    print("Loading ToolVQA (DietCoke4671/ToolVQA, train split) ...")
    from huggingface_hub import hf_hub_download
    jsonl_path = hf_hub_download(
        "DietCoke4671/ToolVQA", "train.jsonl", repo_type="dataset"
    )
    ds = [json.loads(line) for line in open(jsonl_path)]
    print(f"  rows: {len(ds)}")
    if ds:
        print(f"  columns: {sorted(ds[0].keys())}")

    # --- Filter cascade with diagnostic counts (always printed) ---
    print("\n=== Filter cascade ===")
    single_call, two_call, diag = _filter_toolvqa(
        ds, args.max_tool_output_chars, args.max_answer_chars
    )
    print(f"[stage 0] total rows: {diag['n_total']}")
    print(f"[stage 1] len(context) distribution: {diag['ctx_lens']}")
    print(f"[stage 2] context[0].name (single-call rows): {diag['tool_names_single']}")
    print(f"[stage 3] top-5 (name0, name1) pairs (two-call rows): {diag['tool_pairs_top5']}")
    print(f"[stage 4] single-call ImageDescription survivors: {diag['single_call_kept']}")
    print(f"[stage 5] two-call (ImageDescription, GoogleSearch) survivors: {diag['two_call_kept']}")
    total_viable = diag["single_call_kept"] + diag["two_call_kept"]
    print(f"  TOTAL viable: {total_viable}")

    # --- Show 3 sample surviving rows (first single-call, first two two-call) ---
    samples = (single_call[:1] + two_call[:2]) if total_viable >= 3 else single_call + two_call
    print("\n=== Sample surviving rows (verbatim, 3 rows) ===")
    for i, r in enumerate(samples):
        kind = "single" if len(r["context"]) == 1 else "chain"
        print(f"\n--- Sample {i+1} ({kind}, type={r.get('type', '?')}) ---")
        print(f"  question: {r.get('question', '')!r}")
        print(f"  answer:   {r.get('answer', '')!r}")
        print(f"  image:    {r.get('image_path', '')}")
        for j, c in enumerate(r["context"]):
            out = (c.get("output") or "")
            print(f"  context[{j}].name:   {c.get('name')}")
            print(f"  context[{j}].output (len={len(out)}): {out[:200]!r}{'...' if len(out) > 200 else ''}")
        print(f"  → merged tool_result: {_make_tool_result(r)[:300]!r}")

    if args.dry_run:
        print("\n(dry-run — no encoder, no parquets written)")
        return

    if total_viable < 500:
        raise SystemExit(
            f"Only {total_viable} viable rows — refusing to build SFT parquets."
        )

    # --- Subset assembly: preserve insertion order per plan ---
    viable = list(single_call) + list(two_call)
    if args.limit:
        viable = viable[: args.limit]
        print(f"\n--limit applied: encoding first {len(viable)} rows")

    # --- Load Apertus tokenizer (always needed for rendering) + LENS_SEARCH_TOOL ---
    from data_prep.lens_search_common import LENS_SEARCH_TOOL

    config = _load_config(args.config)
    apertus_ckpt = config["model"]["checkpoint"]
    print(f"\nLoading Apertus tokenizer from {apertus_ckpt} ...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(apertus_ckpt, trust_remote_code=True)

    # --- IBQ vision model (skip in dry-render mode) ---
    vq_model = None
    if not args.dry_render:
        print(f"Loading IBQ vision tokenizer from {config['model']['vq_model']} ...")
        from inference.vision import encode_image, load_vq_model, smart_resize  # noqa: F401
        vq_model = load_vq_model(config["model"]["vq_model"], device="cuda:0")
        print("IBQ model loaded")
    else:
        print("--dry-render: skipping IBQ vision model load (CPU-only)")

    # --- Output directories ---
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    images_dir = out_dir / "images_sft_toolvqa"
    if not args.dry_render:
        images_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = out_dir / "sft_toolvqa_metadata.jsonl"

    # --- Parallel HF image pre-fetch (skipped in dry-render) ---
    # hf_hub_download caches at HF's standard location and is idempotent on
    # the second call. Pre-warming the cache in parallel decouples network
    # latency from the GPU-bound encode loop downstream. The encode loop's
    # own hf_hub_download call becomes a cache hit (microseconds).
    n_dl_fail = 0
    if not args.dry_render and len(viable) > 0:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        image_relpaths = [r["image_path"] for r in viable if r.get("image_path")]
        print(f"\n=== Pre-fetching {len(image_relpaths)} ToolVQA images "
              f"(max_workers={args.max_download_workers}) ===")
        t_dl = time.time()

        def _fetch(p):
            try:
                return p, hf_hub_download(
                    "DietCoke4671/ToolVQA", p, repo_type="dataset"
                )
            except Exception as e:
                return p, e

        with ThreadPoolExecutor(max_workers=args.max_download_workers) as ex:
            futures = [ex.submit(_fetch, p) for p in image_relpaths]
            n_done = 0
            for fut in as_completed(futures):
                p, result = fut.result()
                n_done += 1
                if isinstance(result, Exception):
                    n_dl_fail += 1
                    if n_dl_fail <= 5:
                        print(f"  ! DL fail {p}: {result}")
                if n_done % 200 == 0 or n_done == len(futures):
                    rate = n_done / max(1e-3, time.time() - t_dl)
                    print(f"  [{n_done}/{len(futures)}] downloads complete "
                          f"({rate:.1f} dl/s, fails so far: {n_dl_fail})")
        dl_elapsed = time.time() - t_dl
        print(f"Pre-fetch complete: "
              f"{len(image_relpaths) - n_dl_fail}/{len(image_relpaths)} in {dl_elapsed:.0f}s "
              f"({len(image_relpaths) / max(1e-3, dl_elapsed):.1f} dl/s). "
              f"Failures: {n_dl_fail}")
        if n_dl_fail > 0:
            print("  (encode loop will retry then skip rows whose download failed)")

    # --- Encode loop ---
    rendered_rows: list[dict] = []
    text_lens: list[int] = []
    n_skipped = 0
    n_image_fail = 0
    smoking_gun_results: list[bool] = []  # (token71 AND token72) per row
    PLACEHOLDER_IMG_TOKENS = "<|vision_start|><|placeholder|><|vision_end|>"

    t_render = time.time()
    with open(metadata_path, "w", encoding="utf-8") as out_f:
        for i, row in enumerate(viable):
            question = row.get("question", "").strip()
            answer = row.get("answer", "").strip()
            image_relpath = row.get("image_path", "")
            if not question or not answer or not image_relpath:
                n_skipped += 1
                continue

            data_id = _sanitize_filename(image_relpath).rsplit(".", 1)[0]
            kind = "single" if len(row["context"]) == 1 else "chain"

            # --- Image: download + IBQ encode, OR placeholder string ---
            if args.dry_render:
                image_tokens = PLACEHOLDER_IMG_TOKENS
                img_local_path = "<dry-render>"
            else:
                try:
                    local_img = hf_hub_download(
                        "DietCoke4671/ToolVQA",
                        image_relpath,
                        repo_type="dataset",
                    )
                    from PIL import Image
                    from inference.vision import encode_image, smart_resize
                    img = Image.open(local_img).convert("RGB")
                    resized = smart_resize(img)
                    img_local_path = images_dir / f"{data_id}.jpg"
                    resized.save(img_local_path, "JPEG", quality=92)
                    image_tokens = encode_image(resized, vq_model)
                except Exception as e:
                    print(f"  SKIP row {data_id}: image fail: {e}")
                    n_image_fail += 1
                    continue

            # --- Tool result string ---
            tool_result = _make_tool_result(row)

            # --- Build structured messages ---
            messages = _build_messages(
                image_tokens, question, tool_result, answer,
                APERTUS_SYSTEM, LENS_SEARCH_TOOL, DISPLAY_ANSWERS_TOOL,
            )

            # --- Render via chat template ---
            text = tokenizer.apply_chat_template(
                messages,
                tools=[LENS_SEARCH_TOOL, DISPLAY_ANSWERS_TOOL],
                enable_thinking=True,
                add_generation_prompt=False,
                tokenize=False,
            )

            # --- Smoking-gun: tokenize and check for token 71 + 72 ---
            ids = tokenizer.encode(text, add_special_tokens=False)
            has71 = 71 in ids
            has72 = 72 in ids
            smoking_gun_results.append(has71 and has72)
            if not (has71 and has72):
                # Don't hard-fail mid-loop yet; we want full counts at end. But
                # do print a loud warning so the run log captures it.
                print(f"  ! smoking-gun MISS row {data_id}: has71={has71} has72={has72}")

            record = {
                "text": text,
                "image_paths": [str(img_local_path)] if not args.dry_render else [],
                "data_id": data_id,
                "kind": kind,
                "image_relpath": image_relpath,
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            rendered_rows.append(record)
            text_lens.append(len(text))

            if (i + 1) % 50 == 0 or i == len(viable) - 1:
                rate = (i + 1) / max(1e-3, time.time() - t_render)
                print(f"  [{i+1}/{len(viable)}] rendered={len(rendered_rows)} "
                      f"skipped={n_skipped} img_fail={n_image_fail} | {rate:.2f} rows/s")

    print(f"\nWrote {len(rendered_rows)} records to {metadata_path}")
    print(f"  skipped (no q/a/image_path): {n_skipped}")
    print(f"  image_fail: {n_image_fail}")

    if text_lens:
        text_lens.sort()
        n = len(text_lens)
        print(f"text char-length: min={text_lens[0]} "
              f"p50={text_lens[n // 2]} p95={text_lens[int(n * 0.95)]} max={text_lens[-1]}")

    # --- Smoking-gun final verdict ---
    n_passing = sum(1 for ok in smoking_gun_results if ok)
    n_total_sg = len(smoking_gun_results)
    print(f"\n=== SMOKING-GUN CHECK ===")
    print(f"  Rows with both token 71 (<|tools_prefix|>) AND token 72 "
          f"(<|tools_suffix|>): {n_passing} / {n_total_sg}")
    if n_total_sg > 0 and n_passing < n_total_sg:
        raise SystemExit(
            f"SMOKING-GUN FAIL: only {n_passing}/{n_total_sg} rows contain both "
            f"token 71 and token 72. Chat template is dropping the tool-call "
            f"delimiters. Aborting before any parquet write."
        )
    print(f"  → all rows contain Apertus tool delimiters. OK.")

    # --- Print 3 verbatim rendered samples ---
    print("\n=== 3 rendered text samples (verbatim, first 4000 chars) ===")
    for i, rec in enumerate(rendered_rows[:3]):
        print(f"\n--- Sample {i+1} (kind={rec['kind']}, data_id={rec['data_id']}) ---")
        print(rec["text"][:4000])
        if len(rec["text"]) > 4000:
            print(f"... [truncated, total len={len(rec['text'])}]")

    if not rendered_rows:
        raise SystemExit("No rendered rows — refusing to write empty parquets.")

    # --- Train/val split ---
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(rendered_rows))
    n_val = max(1, int(round(len(rendered_rows) * args.val_ratio))) if len(rendered_rows) > 1 else 0
    val_set = set(perm[:n_val].tolist())
    train_recs = [r for i, r in enumerate(rendered_rows) if i not in val_set]
    val_recs = [r for i, r in enumerate(rendered_rows) if i in val_set]

    train_path = out_dir / "sft_toolvqa_train.parquet"
    val_path = out_dir / "sft_toolvqa_val.parquet"
    pq.write_table(pa.Table.from_pylist(train_recs), train_path)
    pq.write_table(pa.Table.from_pylist(val_recs), val_path)
    print(f"\nWrote {len(train_recs)} → {train_path}")
    print(f"Wrote {len(val_recs)} → {val_path}")

    # --- Kind breakdown for both splits ---
    from collections import Counter
    print(f"\ntrain kind distribution: {dict(Counter(r['kind'] for r in train_recs))}")
    print(f"val   kind distribution: {dict(Counter(r['kind'] for r in val_recs))}")

    print(f"\nLicense: Apache-2.0 (DietCoke4671/ToolVQA, verified 2026-05-25).")


if __name__ == "__main__":
    main()
