"""Prepare Math-VR-train (gogoduan/Math-VR-train, Apache-2.0) for visual-math RL.

Downloads the dataset from HF, IBQ-encodes images, splits into
  - mathvr_preflight.parquet  : 50  rows  (Step 2 pre-flight baseline check)
  - mathvr_indist_eval.parquet: 500 rows  (in-dist held-out eval)
  - mathvr_smoke.parquet      : 200 rows  (Step 3 plumbing smoke; subset of train)
  - mathvr_train.parquet      : 5000 rows (Step 4 real RL)

Splits are deterministic via `--seed 42`. The 200 smoke rows are the FIRST 200
of the 5000 train rows (so smoke ⊂ train; reusing the same parquet is fine but
we ship a separate smoke parquet to keep iteration time honest).

Final-answer extraction: Math-VR-train's `analysis` field is prose ending with
"Therefore, the answer is: X." We regex out X via
data_prep.visual_math_rl_common.extract_mathvr_answer. Rows where extraction
fails (no anchored answer found) are dropped and reported.

Usage (interactive on a 4xGH200 node):
    python data_prep/prepare_visual_math_rl_mathvr.py
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
    extract_mathvr_answer,
    load_apertus_models,
    report_split_stats,
    write_parquet,
)

HF_DATASET = "gogoduan/Math-VR-train"


def _to_pil(image_field) -> Image.Image | None:
    """HF datasets returns Image either as PIL or as dict/bytes; normalize."""
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
    if isinstance(image_field, str):  # path
        return Image.open(image_field)
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=str(PROJECT_ROOT / "data_prep" / "visual_math_rl"))
    p.add_argument("--images-dir", default=None,
                   help="Override images dir. Default: <out-dir>/mathvr_images")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-preflight", type=int, default=50)
    p.add_argument("--n-indist-eval", type=int, default=200)
    p.add_argument("--n-train", type=int, default=5000)
    p.add_argument("--n-smoke", type=int, default=200,
                   help="Smoke subset (subset of train, first N rows)")
    p.add_argument("--config", default="configs/apertus.yaml")
    p.add_argument("--limit", type=int, default=None,
                   help="Smoke mode: load only first N rows from HF for testing")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    images_dir = Path(args.images_dir) if args.images_dir else out_dir / "mathvr_images"
    out_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    needed = args.n_preflight + args.n_indist_eval + args.n_train
    print(f"Target rows: preflight={args.n_preflight} indist={args.n_indist_eval} "
          f"train={args.n_train} (smoke=first {args.n_smoke} of train) "
          f"= {needed} total", flush=True)

    # Load HF dataset
    from datasets import load_dataset
    print(f"Downloading {HF_DATASET} from HF...", flush=True)
    ds = load_dataset(HF_DATASET, split="train", streaming=False)
    print(f"  Source: {len(ds)} rows", flush=True)

    # Skim to find rows with extractable answers + decodable image (drop bad rows)
    print("Filtering rows for usable answer + image...", flush=True)
    rng = np.random.default_rng(args.seed)
    # First shuffle the indices, then iterate until we have enough usable
    perm = rng.permutation(len(ds))
    if args.limit:
        perm = perm[: max(args.limit, needed * 2)]

    # Load Apertus + IBQ models
    tokenizer, vq_model = load_apertus_models(args.config)

    usable: list[dict] = []
    n_drop_answer = 0
    n_drop_image = 0
    n_drop_encode = 0
    t0 = time.time()
    for k, idx in enumerate(perm):
        if len(usable) >= needed:
            break
        row = ds[int(idx)]

        # 1. extract final answer
        gold = extract_mathvr_answer(row.get("analysis", ""))
        if gold is None:
            n_drop_answer += 1
            continue

        # 2. load image (Math-VR-train has image1, image2, ..., image8 — most rows
        #    have only image1 populated)
        pil = None
        for ic in (1, 2, 3, 4, 5, 6, 7, 8):
            ifield = row.get(f"image{ic}")
            if ifield is None:
                continue
            pil = _to_pil(ifield)
            if pil is not None:
                break
        if pil is None:
            n_drop_image += 1
            continue

        # 3. IBQ encode
        image_tokens = encode_image_to_tokens(pil, vq_model)
        if image_tokens is None:
            n_drop_encode += 1
            continue

        # Save image bytes alongside parquet for debugging / re-encoding
        rid = str(row.get("id", f"mathvr_{int(idx)}"))
        img_path = images_dir / f"{rid}.png"
        if not img_path.exists():
            try:
                pil.save(img_path, "PNG")
            except Exception as e:  # noqa: BLE001
                print(f"  warn: failed to save {img_path}: {e}", flush=True)

        usable.append({
            "id": rid,
            "question": row.get("question", "").strip(),
            "analysis": row.get("analysis", "").strip(),
            "gold": gold,
            "category": row.get("category", ""),
            "image_path": str(img_path),
            "image_tokens": image_tokens,
        })

        if (k + 1) % 50 == 0 or len(usable) >= needed:
            elapsed = time.time() - t0
            rate = (k + 1) / max(elapsed, 1e-6)
            print(f"  [{k+1}/{len(perm)} scanned] usable={len(usable)}  "
                  f"drop_ans={n_drop_answer} drop_img={n_drop_image} "
                  f"drop_enc={n_drop_encode}  rate={rate:.2f} rows/s",
                  flush=True)

    print(f"\nCollected {len(usable)} usable rows after scanning {k+1}.", flush=True)
    print(f"  dropped: answer={n_drop_answer}, image={n_drop_image}, encode={n_drop_encode}",
          flush=True)
    if len(usable) < needed:
        print(f"  WARN: only {len(usable)} usable, needed {needed}. "
              f"Will write what we have.", flush=True)

    # Split: preflight | indist | train  (in this order, deterministic)
    i0 = 0
    i1 = i0 + min(args.n_preflight, len(usable))
    i2 = i1 + min(args.n_indist_eval, len(usable) - i1)
    i3 = i2 + min(args.n_train, len(usable) - i2)
    preflight_rows = usable[i0:i1]
    indist_rows = usable[i1:i2]
    train_rows = usable[i2:i3]
    smoke_rows = train_rows[: min(args.n_smoke, len(train_rows))]

    print(f"\nSplits: preflight={len(preflight_rows)} indist={len(indist_rows)} "
          f"train={len(train_rows)} smoke={len(smoke_rows)} (subset of train)",
          flush=True)

    def to_records(rows, data_source):
        out = []
        for r in rows:
            extra = {
                "index": r["id"],
                "answer": r["gold"],
                "answer_type": "free",
                "subject": r["category"],
                "level": "",
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

    preflight_recs = to_records(preflight_rows, "math_vr_train")
    indist_recs = to_records(indist_rows, "math_vr_train")
    train_recs = to_records(train_rows, "math_vr_train")
    smoke_recs = to_records(smoke_rows, "math_vr_train")

    write_parquet(preflight_recs, out_dir / "mathvr_preflight.parquet")
    write_parquet(indist_recs, out_dir / "mathvr_indist_eval.parquet")
    write_parquet(train_recs, out_dir / "mathvr_train.parquet")
    write_parquet(smoke_recs, out_dir / "mathvr_smoke.parquet")

    # Save raw row metadata for debugging
    with open(out_dir / "mathvr_raw_meta.jsonl", "w") as f:
        for r in usable:
            f.write(json.dumps({k: r[k] for k in r if k != "image_tokens"},
                               ensure_ascii=False) + "\n")

    # Stats
    print("\n=== Splits summary ===", flush=True)
    report_split_stats("preflight", preflight_recs)
    report_split_stats("indist_eval", indist_recs)
    report_split_stats("train", train_recs)
    report_split_stats("smoke", smoke_recs)

    print(f"\nWrote parquets to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
