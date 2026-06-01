"""Build the FVQA Lens-search SFT parquets.

Writes:
  data_prep/lens_search/sft_train.parquet  (~950 rows)
  data_prep/lens_search/sft_val.parquet    (~50 rows; 5% val split)
  data_prep/lens_search/sft_metadata.jsonl (human-inspectable)

Each row's `text` field is the **fully rendered Apertus conversation**, ready
for the `CoFSFTDataset` loader (which masks tool-output spans automatically).

Source: `lmms-lab/FVQA` (Apache-2.0, 4,856 train rows; we take the first 1,000
after a seed-42 deterministic shuffle for the SFT subset). Tool-result content
comes from `fvqa_train_image_search_results_cache.pkl` (4,849 entries; rows
with missing cache entries are skipped).

Pattern: mirrors `prepare_cof_sft_parse.py`. The Apertus chat template
(`apply_chat_template`) does the actual rendering from structured `messages`
dicts to the `<|tools_prefix|>...<|tools_suffix|>` token stream.

Usage:
    python data_prep/prepare_lens_search_sft.py --n 1000 --val_ratio 0.05
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data_prep.lens_search_common import (
    LENS_SEARCH_TOOL,
    load_fvqa_cache,
    format_cached_result,
    synthesize_thinking_blocks,
    derive_query_from_question,
    extract_question,
    extract_ground_truth,
)
from data_prep.prepare_cof_rl_parse import (
    APERTUS_SYSTEM,
    DISPLAY_ANSWERS_TOOL,
    load_config,
)


APERTUS_INSTRUCTION = (
    "If you need information about the entity, landmark, object, or text in the "
    "image, call the lens_search tool with a short natural-language query. Use "
    "the search results to inform your answer. Then call the display_answers "
    "tool exactly once at the end of your response, passing your final answer "
    "as the single element of the `answers` argument."
)


def build_messages(
    image_token_str: str,
    question: str,
    cached_result: str,
    gold_answer: str,
) -> list:
    """Construct the structured Apertus messages list for one SFT row."""
    pre_thought, post_thought = synthesize_thinking_blocks(question, gold_answer)
    query = derive_query_from_question(question)

    return [
        {"role": "system", "content": APERTUS_SYSTEM},
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
        {"role": "tool", "content": cached_result},
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
                                    {"answers": [gold_answer]}, ensure_ascii=False
                                ),
                            }
                        ],
                    },
                ]
            },
        },
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="data_prep/lens_search")
    ap.add_argument("--n", type=int, default=1000,
                    help="Target SFT subset size (before val split)")
    ap.add_argument("--val-ratio", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--config", default="configs/apertus.yaml")
    ap.add_argument("--fvqa-cache", default=None,
                    help="Path to fvqa_train.parquet (auto-discovered via HF cache if omitted)")
    ap.add_argument("--fvqa-pkl", default=None,
                    help="Path to fvqa_train_image_search_results_cache.pkl")
    ap.add_argument("--max-text-chars", type=int, default=32000,
                    help="Skip rows whose rendered text exceeds this; safety cap")
    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    images_dir = out_dir / "images_sft"
    images_dir.mkdir(parents=True, exist_ok=True)

    # 1) Locate FVQA data
    from huggingface_hub import hf_hub_download
    if args.fvqa_cache is None:
        args.fvqa_cache = hf_hub_download(
            "lmms-lab/FVQA", "fvqa_train.parquet", repo_type="dataset"
        )
    if args.fvqa_pkl is None:
        args.fvqa_pkl = hf_hub_download(
            "lmms-lab/FVQA",
            "fvqa_train_image_search_results_cache.pkl",
            repo_type="dataset",
        )
    print(f"FVQA parquet: {args.fvqa_cache}")
    print(f"FVQA cache  : {args.fvqa_pkl}")

    # 2) Load FVQA + cache
    t0 = time.time()
    table = pq.read_table(args.fvqa_cache)
    print(f"Loaded FVQA: {table.num_rows} rows, cols={table.column_names}")
    cache = load_fvqa_cache(args.fvqa_pkl)
    print(f"Loaded cache: {len(cache)} entries")

    # 3) Subset selection (seed-shuffled, then take first --n; record selected
    # data_ids so the RL prep can exclude them).
    n_total = table.num_rows
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n_total)
    sft_indices = list(perm[: args.n])

    selected_ids: list[str] = []
    rows_to_process = []
    for idx in sft_indices:
        idx = int(idx)
        row = {c: table.column(c)[idx].as_py() for c in table.column_names}
        data_id = row.get("data_id")
        if data_id not in cache:
            continue
        rows_to_process.append((idx, row))
        selected_ids.append(data_id)
    print(f"Selected {len(rows_to_process)} rows (with cache entries; {args.n - len(rows_to_process)} skipped for missing cache)")

    # 4) Load Apertus tokenizer + IBQ model
    config = load_config(args.config)
    print(f"Loading Apertus tokenizer from {config['model']['checkpoint']} ...")
    from transformers import AutoTokenizer
    from inference.vision import encode_image, load_vq_model, smart_resize

    tokenizer = AutoTokenizer.from_pretrained(
        config["model"]["checkpoint"], trust_remote_code=True
    )
    print(f"Loading IBQ vision tokenizer from {config['model']['vq_model']} ...")
    vq_model = load_vq_model(config["model"]["vq_model"], device="cuda:0")
    print("Models loaded")

    # 5) Render each row
    metadata_path = out_dir / "sft_metadata.jsonl"
    text_lens: list[int] = []
    skipped = 0
    written = 0
    rendered_rows: list[dict] = []
    t_render = time.time()

    with open(metadata_path, "w", encoding="utf-8") as out_f:
        for i, (idx, row) in enumerate(rows_to_process):
            data_id = row["data_id"]
            question = extract_question(row.get("prompt"))
            gold = extract_ground_truth(row.get("reward_model"))
            if not question or not gold:
                skipped += 1
                continue

            # Image from `images` field (list of {bytes, path} dicts)
            imgs = row.get("images") or []
            if not imgs or not isinstance(imgs, list):
                skipped += 1
                continue
            img_blob = imgs[0]
            img_bytes = img_blob.get("bytes") if isinstance(img_blob, dict) else None
            if not img_bytes:
                skipped += 1
                continue
            try:
                img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                resized = smart_resize(img)
                img_path = images_dir / f"{data_id}.jpg"
                resized.save(img_path, "JPEG", quality=92)
                img_tokens = encode_image(resized, vq_model)
            except Exception as e:
                print(f"  SKIP row {data_id}: IBQ encode failed: {e}")
                skipped += 1
                continue

            cache_entry = cache[data_id]
            tool_result_text = format_cached_result(cache_entry)

            messages = build_messages(img_tokens, question, tool_result_text, gold)

            text = tokenizer.apply_chat_template(
                messages,
                tools=[LENS_SEARCH_TOOL, DISPLAY_ANSWERS_TOOL],
                enable_thinking=True,
                add_generation_prompt=False,
                tokenize=False,
            )
            if len(text) > args.max_text_chars:
                skipped += 1
                continue

            record = {
                "text": text,
                "image_paths": [str(img_path)],
                "data_id": data_id,
                "category": row.get("category", "unknown"),
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            rendered_rows.append(record)
            text_lens.append(len(text))
            written += 1

            if (i + 1) % 100 == 0 or i == len(rows_to_process) - 1:
                rate = (i + 1) / (time.time() - t_render)
                print(f"  [{i+1}/{len(rows_to_process)}] written={written} skipped={skipped} | {rate:.2f} rows/s")

    print(f"\nWrote {written} records to {metadata_path}  (skipped {skipped})")
    if text_lens:
        text_lens.sort()
        n = len(text_lens)
        print(f"text char-length: min={text_lens[0]} p50={text_lens[n//2]} p95={text_lens[int(n*0.95)]} max={text_lens[-1]}")

    # 6) Train/val split → parquets
    if not rendered_rows:
        raise SystemExit("No SFT rows rendered — aborting")
    rng2 = np.random.default_rng(args.seed)
    perm2 = rng2.permutation(len(rendered_rows))
    n_val = max(1, int(round(len(rendered_rows) * args.val_ratio))) if len(rendered_rows) > 1 else 0
    val_set = set(perm2[:n_val].tolist())
    train_recs = [r for i, r in enumerate(rendered_rows) if i not in val_set]
    val_recs = [r for i, r in enumerate(rendered_rows) if i in val_set]

    train_path = out_dir / "sft_train.parquet"
    val_path = out_dir / "sft_val.parquet"
    pq.write_table(pa.Table.from_pylist(train_recs), train_path)
    pq.write_table(pa.Table.from_pylist(val_recs), val_path)
    print(f"\nWrote {len(train_recs)} → {train_path}")
    print(f"Wrote {len(val_recs)} → {val_path}")

    # 7) Save the data_ids used so the RL prep can exclude them
    selected_path = out_dir / "sft_selected_data_ids.json"
    with open(selected_path, "w") as f:
        json.dump(sorted(set(selected_ids)), f, indent=2)
    print(f"Wrote {len(set(selected_ids))} selected data_ids → {selected_path}")

    # 8) Print 1 rendered sample for inspection
    if rendered_rows:
        print(f"\n=== Sample rendered SFT text (data_id={rendered_rows[0]['data_id']}, category={rendered_rows[0]['category']}) ===")
        print(rendered_rows[0]["text"][:4000])
        print("...")
        print(f"(total text length: {len(rendered_rows[0]['text'])} chars)")

    print(f"\nLicense: Apache-2.0 (lmms-lab/FVQA, verified 2026-05-20).")
    print(f"Total time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
