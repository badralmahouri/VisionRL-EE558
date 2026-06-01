"""Build the FVQA Lens-search RL parquet.

Writes:
  data_prep/lens_search/rl_train.parquet  (~3,856 rows — FVQA minus the 1k SFT subset)
  data_prep/lens_search/rl_metadata.jsonl (human-inspectable)

Verl GRPO parquet schema (mirrors `prepare_cof_rl_parse.py`):
    {
      "data_source": "lens_search_rl",
      "agent_name":  "lens_search_agent",   # registered in apertus_agent_loop.yaml
      "prompt":      [{"role":"system","content":...}, {"role":"user","content":...}],
      "ability":     "lens_search",
      "reward_model":{"style":"rule","ground_truth": <answer>},
      "extra_info":  {
          "index": <data_id>,
          "category": <search_required|search_free>,
          "answer": <answer>,
          "need_tools_kwargs": True,
          "tools_kwargs": {
              "lens_search_tool": {
                  "create_kwargs": {"image_path": <abs path>, "data_id": <data_id>},
              },
          },
      },
    }

The `image_path` bound at `create()` time tells the LensSearchTool wrapper
which image to send to the search backend (cached or live). The `data_id` is
passed alongside so the cached mode can look up the FVQA pkl directly.

Excludes rows whose `data_id` is in `data_prep/lens_search/sft_selected_data_ids.json`
(written by `prepare_lens_search_sft.py`) so there's zero overlap between
SFT and RL splits.

Usage:
    python data_prep/prepare_lens_search_rl.py
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
    ap.add_argument("--sft-selected-ids",
                    default="data_prep/lens_search/sft_selected_data_ids.json",
                    help="JSON list of data_ids to exclude (held out for SFT)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    images_dir = out_dir / "images_rl"
    images_dir.mkdir(parents=True, exist_ok=True)

    # 1) Locate FVQA train parquet
    from huggingface_hub import hf_hub_download
    if args.fvqa_parquet is None:
        args.fvqa_parquet = hf_hub_download(
            "lmms-lab/FVQA", "fvqa_train.parquet", repo_type="dataset"
        )
    print(f"FVQA parquet: {args.fvqa_parquet}")

    # 2) Load SFT-selected exclusion list
    sft_ids_path = Path(args.sft_selected_ids)
    if not sft_ids_path.exists():
        raise SystemExit(
            f"SFT selected-ids file not found at {sft_ids_path}.\n"
            f"Run `python data_prep/prepare_lens_search_sft.py` first."
        )
    excluded = set(json.loads(sft_ids_path.read_text()))
    print(f"Excluding {len(excluded)} data_ids that were used for SFT")

    # 3) Load FVQA + IBQ models
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

    # 4) Build RL parquet rows
    metadata_path = out_dir / "rl_metadata.jsonl"
    records: list[dict] = []
    n_excluded = 0
    n_skipped = 0
    text_lens: list[int] = []
    t_render = time.time()

    USER_BLOCK_OPEN = "<|user_start|>"
    USER_BLOCK_CLOSE = "<|user_end|>"

    with open(metadata_path, "w", encoding="utf-8") as out_f:
        for i in range(table.num_rows):
            row = {c: table.column(c)[i].as_py() for c in table.column_names}
            data_id = row.get("data_id")
            if data_id in excluded:
                n_excluded += 1
                continue

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

            # Render the prompt via the chat template so the user block has
            # the right `<|user_start|>...<|user_end|>` markup; then extract
            # the user content for the verl-RL `prompt` field.
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

            # Extract the user content (mirroring `prepare_cof_rl_parse._extract_user_content`)
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
                rate = (i + 1) / (time.time() - t_render)
                print(f"  [{i+1}/{table.num_rows}] kept={len(records)} excl={n_excluded} skip={n_skipped} | {rate:.2f} rows/s")

    print(f"\nKept {len(records)} RL rows (excluded {n_excluded} SFT-selected, skipped {n_skipped} bad).")
    if text_lens:
        text_lens.sort()
        n = len(text_lens)
        print(f"prompt char-length: min={text_lens[0]} p50={text_lens[n//2]} p95={text_lens[int(n*0.95)]} max={text_lens[-1]}")

    # Category breakdown — should approximate FVQA's 2.15:1 search-required:search-free
    from collections import Counter
    cat = Counter(r["extra_info"]["category"] for r in records)
    print(f"category distribution: {dict(cat)}")

    rl_path = out_dir / "rl_train.parquet"
    pq.write_table(pa.Table.from_pylist(records), rl_path)
    print(f"\nWrote {len(records)} → {rl_path}")

    # Print a sample for inspection
    if records:
        print(f"\n=== Sample RL row (index={records[0]['extra_info']['index']}, category={records[0]['extra_info']['category']}) ===")
        print(json.dumps({
            "data_source": records[0]["data_source"],
            "agent_name": records[0]["agent_name"],
            "ability": records[0]["ability"],
            "reward_model": records[0]["reward_model"],
            "extra_info": records[0]["extra_info"],
            "prompt[0].role": records[0]["prompt"][0]["role"],
            "prompt[1].role": records[0]["prompt"][1]["role"],
            "prompt[1].content_first_500": records[0]["prompt"][1]["content"][:500],
        }, indent=2, ensure_ascii=False))

    print(f"\nLicense: Apache-2.0 (lmms-lab/FVQA, verified 2026-05-20).")
    print(f"Total time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
