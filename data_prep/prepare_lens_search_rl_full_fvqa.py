"""Build the FULL-FVQA Lens-search RL parquets (Phase D.7.2).

Replaces the prior `prepare_lens_search_rl.py` which excluded the 1,000 FVQA
rows held out for SFT cold-start. After the D.7.1 pivot to ToolVQA-based SFT,
no FVQA rows are spoken for: all 4,856 are available for RL. We hold out
200 rows (seed=42) as an in-distribution validation set; the rest become
the RL train set.

Writes:
  data_prep/lens_search/rl_fvqa_train.parquet      (~4,656 rows)
  data_prep/lens_search/rl_fvqa_indist_val.parquet (~200 rows)
  data_prep/lens_search/rl_fvqa_metadata.jsonl     (human-inspectable)

Schema matches `prepare_lens_search_rl.py` exactly (verl GRPO).

Usage:
    python data_prep/prepare_lens_search_rl_full_fvqa.py
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="data_prep/lens_search")
    ap.add_argument("--config", default="configs/apertus.yaml")
    ap.add_argument("--fvqa-parquet", default=None)
    ap.add_argument("--n-val", type=int, default=200,
                    help="Hold-out in-dist val set size (seed-fixed).")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    images_dir = out_dir / "images_rl_fvqa"
    images_dir.mkdir(parents=True, exist_ok=True)

    # 1) Locate FVQA train parquet
    from huggingface_hub import hf_hub_download
    if args.fvqa_parquet is None:
        args.fvqa_parquet = hf_hub_download(
            "lmms-lab/FVQA", "fvqa_train.parquet", repo_type="dataset"
        )
    print(f"FVQA parquet: {args.fvqa_parquet}")

    # 2) Load FVQA + IBQ models
    t0 = time.time()
    table = pq.read_table(args.fvqa_parquet)
    print(f"Loaded FVQA: {table.num_rows} rows, cols={table.column_names}")

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

    # 3) Build all RL records (no SFT exclusion)
    metadata_path = out_dir / "rl_fvqa_metadata.jsonl"
    records: list[dict] = []
    n_skipped = 0
    text_lens: list[int] = []
    t_render = time.time()

    USER_BLOCK_OPEN = "<|user_start|>"
    USER_BLOCK_CLOSE = "<|user_end|>"

    with open(metadata_path, "w", encoding="utf-8") as out_f:
        for i in range(table.num_rows):
            row = {c: table.column(c)[i].as_py() for c in table.column_names}
            data_id = row.get("data_id")

            question = extract_question(row.get("prompt"))
            gold = extract_ground_truth(row.get("reward_model"))
            if not question or not gold:
                n_skipped += 1
                continue

            imgs = row.get("images") or []
            if not imgs or not isinstance(imgs, list):
                n_skipped += 1
                continue
            img_blob = imgs[0]
            img_bytes = img_blob.get("bytes") if isinstance(img_blob, dict) else None
            if not img_bytes:
                n_skipped += 1
                continue
            try:
                img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                resized = smart_resize(img)
                img_path = images_dir / f"{data_id}.jpg"
                resized.save(img_path, "JPEG", quality=92)
                img_tokens = encode_image(resized, vq_model)
            except Exception as e:
                print(f"  SKIP row {data_id}: IBQ encode failed: {e}")
                n_skipped += 1
                continue

            sys_msg = {"role": "system", "content": APERTUS_SYSTEM}
            user_msg = {
                "role": "user",
                "content": f"{img_tokens}\n\n{question}\n\n{APERTUS_INSTRUCTION}",
            }
            rendered = tokenizer.apply_chat_template(
                [sys_msg, user_msg],
                tools=[LENS_SEARCH_TOOL, DISPLAY_ANSWERS_TOOL],
                enable_thinking=True,
                add_generation_prompt=True,
                tokenize=False,
            )
            text_lens.append(len(rendered))

            user_start = rendered.find(USER_BLOCK_OPEN)
            user_end = rendered.find(USER_BLOCK_CLOSE, user_start)
            if user_start < 0 or user_end < 0:
                print(f"  SKIP row {data_id}: user block not found in rendered prompt")
                n_skipped += 1
                continue
            user_content = rendered[user_start + len(USER_BLOCK_OPEN): user_end]

            record = {
                "data_source": "lens_search_rl",
                "agent_name": "lens_search_agent",
                "prompt": [
                    {"role": "system", "content": APERTUS_SYSTEM},
                    {"role": "user", "content": user_content},
                ],
                "ability": "lens_search",
                "reward_model": {"style": "rule", "ground_truth": gold},
                "extra_info": {
                    "index": data_id,
                    "category": row.get("category", "unknown"),
                    "answer": gold,
                    "need_tools_kwargs": True,
                    "tools_kwargs": {
                        "lens_search_tool": {
                            "create_kwargs": {
                                "image_path": str(img_path),
                                "data_id": data_id,
                            },
                        },
                    },
                },
            }
            records.append(record)
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

            if (i + 1) % 200 == 0 or i == table.num_rows - 1:
                rate = (i + 1) / max(1e-3, time.time() - t_render)
                print(f"  [{i+1}/{table.num_rows}] kept={len(records)} skip={n_skipped} | {rate:.2f} rows/s")

    print(f"\nKept {len(records)} FVQA RL rows (skipped {n_skipped} bad).")
    if text_lens:
        text_lens.sort()
        n = len(text_lens)
        print(f"prompt char-length: min={text_lens[0]} p50={text_lens[n // 2]} "
              f"p95={text_lens[int(n * 0.95)]} max={text_lens[-1]}")

    # 4) Train / in-dist val split (seed-fixed permutation, first N as val)
    if len(records) < args.n_val + 100:
        raise SystemExit(
            f"Too few RL records ({len(records)}) to hold out {args.n_val} val."
        )
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(records))
    val_idx = set(perm[: args.n_val].tolist())
    val_recs = [r for i, r in enumerate(records) if i in val_idx]
    train_recs = [r for i, r in enumerate(records) if i not in val_idx]

    # Sanity: no overlap (set membership ensures it, but assert anyway)
    train_ids = {r["extra_info"]["index"] for r in train_recs}
    val_ids = {r["extra_info"]["index"] for r in val_recs}
    overlap = train_ids & val_ids
    if overlap:
        raise SystemExit(f"Overlap between train and val: {len(overlap)} ids")

    train_path = out_dir / "rl_fvqa_train.parquet"
    val_path = out_dir / "rl_fvqa_indist_val.parquet"
    pq.write_table(pa.Table.from_pylist(train_recs), train_path)
    pq.write_table(pa.Table.from_pylist(val_recs), val_path)
    print(f"\nWrote {len(train_recs)} → {train_path}")
    print(f"Wrote {len(val_recs)} → {val_path}")

    # 5) Category breakdown per split
    from collections import Counter
    print(f"\ntrain category distribution: "
          f"{dict(Counter(r['extra_info']['category'] for r in train_recs))}")
    print(f"val   category distribution: "
          f"{dict(Counter(r['extra_info']['category'] for r in val_recs))}")

    # 6) Print one sample
    if train_recs:
        s = train_recs[0]
        print(f"\n=== Sample train row (index={s['extra_info']['index']}, "
              f"category={s['extra_info']['category']}) ===")
        print(json.dumps({
            "data_source": s["data_source"],
            "agent_name": s["agent_name"],
            "ability": s["ability"],
            "reward_model": s["reward_model"],
            "extra_info": s["extra_info"],
            "prompt[0].role": s["prompt"][0]["role"],
            "prompt[1].role": s["prompt"][1]["role"],
            "prompt[1].content_first_500": s["prompt"][1]["content"][:500],
        }, indent=2, ensure_ascii=False))

    print(f"\nLicense: Apache-2.0 (lmms-lab/FVQA, verified 2026-05-20).")
    print(f"Total time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
