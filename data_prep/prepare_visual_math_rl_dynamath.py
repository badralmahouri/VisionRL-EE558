"""Prepare DynaMath (DynaMath/DynaMath_Sample, Apache-2.0) for visual-math RL.

DynaMath ships as 10 splits (variant1..variant10), each with 501 rows = the
same 501 seed problems rendered with different parameter values.

Variant-based train/test split (no question leakage):
  - Variant 10        -> in-dist held-out eval (501 rows)
  - Variants 1..9     -> training pool (~4509 rows)
    - first 50 of training (deterministic shuffle)  -> pre-flight (removed from train)
    - first 200 of remaining training               -> plumbing smoke (subset of train)
    - rest                                          -> real RL training (~4259 rows)

Per-row schema (DynaMath/DynaMath_Sample):
  id, question, image (PIL), decoded_image, ground_truth, answer_type, subject,
  knowledge_level

Usage (interactive on a 4xGH200 node):
    python data_prep/prepare_visual_math_rl_dynamath.py
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data_prep.visual_math_rl_common import (  # noqa: E402
    build_record,
    encode_image_to_tokens,
    load_apertus_models,
    report_split_stats,
    write_parquet,
)

HF_DATASET = "DynaMath/DynaMath_Sample"
NUM_VARIANTS = 10


def _to_pil(image_field) -> Image.Image | None:
    if image_field is None:
        return None
    if isinstance(image_field, Image.Image):
        return image_field
    if isinstance(image_field, dict):
        if "bytes" in image_field and image_field["bytes"]:
            return Image.open(io.BytesIO(image_field["bytes"]))
        if "path" in image_field and image_field["path"]:
            return Image.open(image_field["path"])
    if isinstance(image_field, (bytes, bytearray)):
        return Image.open(io.BytesIO(image_field))
    if isinstance(image_field, str):
        return Image.open(image_field)
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=str(PROJECT_ROOT / "data_prep" / "visual_math_rl"))
    p.add_argument("--images-dir", default=None,
                   help="Override images dir. Default: <out-dir>/dynamath_images")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-preflight", type=int, default=50)
    p.add_argument("--n-indist-eval", type=int, default=200,
                   help="Cap variant-10 held-out at this size (variant 10 has 501 rows)")
    p.add_argument("--n-smoke", type=int, default=200)
    p.add_argument("--config", default="configs/apertus.yaml")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    images_dir = Path(args.images_dir) if args.images_dir else out_dir / "dynamath_images"
    out_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    # Load HF (all 10 variants)
    from datasets import load_dataset

    variant_rows: dict[int, list[dict]] = {}  # variant_id -> list of normalized rows
    print(f"Downloading {HF_DATASET} (10 variants)...", flush=True)
    for v in range(1, NUM_VARIANTS + 1):
        split = f"sample_variant{v}"
        ds = load_dataset(HF_DATASET, split=split, streaming=False)
        variant_rows[v] = list(ds)  # Materialize so we can iterate later
        print(f"  variant{v}: {len(variant_rows[v])} rows", flush=True)

    # Load models
    tokenizer, vq_model = load_apertus_models(args.config)

    # Process all rows: IBQ-encode each image. Store by (variant_id, row_dict).
    def process_split(rows, variant_id, label):
        out: list[dict] = []
        t0 = time.time()
        n_drop = 0
        for k, row in enumerate(rows):
            pil = _to_pil(row.get("decoded_image"))
            if pil is None:
                pil = _to_pil(row.get("image"))
            if pil is None:
                n_drop += 1
                continue
            image_tokens = encode_image_to_tokens(pil, vq_model)
            if image_tokens is None:
                n_drop += 1
                continue

            rid = f"v{variant_id}_{row.get('id', k)}"
            img_path = images_dir / f"{rid}.png"
            if not img_path.exists():
                try:
                    pil.save(img_path, "PNG")
                except Exception as e:  # noqa: BLE001
                    print(f"  warn: save failed for {img_path}: {e}", flush=True)

            out.append({
                "id": rid,
                "qname": row.get("id"),
                "variant_idx": variant_id,
                "question": row.get("question", "").strip(),
                "gold": str(row.get("ground_truth", "")).strip(),
                "answer_type": row.get("answer_type", ""),
                "subject": row.get("subject", ""),
                "knowledge_level": row.get("knowledge_level", ""),
                "image_path": str(img_path),
                "image_tokens": image_tokens,
            })

            if (k + 1) % 50 == 0:
                elapsed = time.time() - t0
                rate = (k + 1) / max(elapsed, 1e-6)
                print(f"  [{label}] [{k+1}/{len(rows)}] rate={rate:.2f}/s drop={n_drop}",
                      flush=True)
        print(f"  [{label}] done: {len(out)} usable, {n_drop} dropped", flush=True)
        return out

    print("\n=== Processing held-out (variant 10) ===", flush=True)
    indist_rows = process_split(variant_rows[NUM_VARIANTS], NUM_VARIANTS, "indist_eval")
    if len(indist_rows) > args.n_indist_eval:
        # Deterministic cap: take first n_indist_eval rows (variant 10 is already
        # in seed-order from HF, so this is stable across runs).
        print(f"  capping indist held-out from {len(indist_rows)} -> {args.n_indist_eval}",
              flush=True)
        indist_rows = indist_rows[: args.n_indist_eval]

    print("\n=== Processing training pool (variants 1..9) ===", flush=True)
    train_pool: list[dict] = []
    for v in range(1, NUM_VARIANTS):
        train_pool.extend(process_split(variant_rows[v], v, f"train_v{v}"))

    # Deterministic shuffle of training pool
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(train_pool))
    train_pool = [train_pool[int(i)] for i in perm]

    # preflight = first N
    preflight_rows = train_pool[: args.n_preflight]
    rest_train = train_pool[args.n_preflight:]
    # smoke = first N of rest
    smoke_rows = rest_train[: args.n_smoke]
    # real train = full rest (smoke is a subset, shipped separately for plumbing)
    real_train = rest_train

    print(f"\nSplits: preflight={len(preflight_rows)} indist={len(indist_rows)} "
          f"train={len(real_train)} smoke={len(smoke_rows)}", flush=True)

    def to_records(rows, data_source):
        out = []
        for r in rows:
            extra = {
                "index": r["id"],
                "qname": r["qname"],
                "variant_idx": r["variant_idx"],
                "answer": r["gold"],
                "answer_type": r["answer_type"],
                "subject": r["subject"],
                "level": r["knowledge_level"],
                "image_path": r["image_path"],
            }
            out.append(build_record(
                data_source=data_source,
                image_token_str=r["image_tokens"],
                question_text=r["question"],
                ground_truth=r["gold"],
                extra_info=extra,
            ))
        return out

    preflight_recs = to_records(preflight_rows, "dynamath")
    indist_recs = to_records(indist_rows, "dynamath")
    train_recs = to_records(real_train, "dynamath")
    smoke_recs = to_records(smoke_rows, "dynamath")

    write_parquet(preflight_recs, out_dir / "dynamath_preflight.parquet")
    write_parquet(indist_recs, out_dir / "dynamath_indist_eval.parquet")
    write_parquet(train_recs, out_dir / "dynamath_train.parquet")
    write_parquet(smoke_recs, out_dir / "dynamath_smoke.parquet")

    # Raw meta
    with open(out_dir / "dynamath_raw_meta.jsonl", "w") as f:
        for r in preflight_rows + indist_rows + real_train:
            f.write(json.dumps({k: r[k] for k in r if k != "image_tokens"},
                               ensure_ascii=False) + "\n")

    print("\n=== Splits summary ===", flush=True)
    report_split_stats("preflight", preflight_recs)
    report_split_stats("indist_eval", indist_recs)
    report_split_stats("train", train_recs)
    report_split_stats("smoke", smoke_recs)

    print(f"\nWrote parquets to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
