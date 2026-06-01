"""Prepare OOD eval suite: MathVision-500 + MathVerse Vision-Only-500.

Both are MIT-licensed:
  - MathLLMs/MathVision     -> sample 500 rows from `testmini` split, EXCLUDE
                                the 200 IDs already on disk at
                                data_prep/calc_eval/mathvision_200/raw.jsonl
                                (so the new eval is genuinely held-out).
  - AI4Math/MathVerse       -> filter to `Vision Only` subset of `testmini`,
                                sample 500 rows.

Outputs:
  - mathvision_eval.parquet
  - mathverse_visionOnly_eval.parquet

Used for OOD eval during training (every 50 iters) and the pre/post
image-perception gate (compare pre-RL vs post-RL MathVision-500 score).

Usage (interactive on a 4xGH200 node):
    python data_prep/prepare_visual_math_rl_eval.py
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


def _load_existing_mathvision_ids() -> set[str]:
    """Read the 200 IDs from data_prep/calc_eval/mathvision_200/raw.jsonl."""
    path = PROJECT_ROOT / "data_prep" / "calc_eval" / "mathvision_200" / "raw.jsonl"
    if not path.exists():
        return set()
    ids = set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            ids.add(str(row.get("id", "")))
    return ids


def _format_question_with_options(question: str, options) -> str:
    """If MathVision has options, append them inline to the question."""
    if not options:
        return question
    opts = list(options)
    if not opts:
        return question
    opt_lines = []
    for i, opt in enumerate(opts):
        letter = "ABCDE"[i] if i < 5 else f"({i+1})"
        opt_lines.append(f"({letter}) {opt}")
    return f"{question.strip()}\n\nOptions:\n" + "\n".join(opt_lines)


def prepare_mathvision(out_dir: Path, images_dir: Path, vq_model, n: int, seed: int):
    """Sample n MathVision rows EXCLUDING the on-disk 200, IBQ-encode, write parquet."""
    from datasets import load_dataset

    excluded = _load_existing_mathvision_ids()
    print(f"\n=== MathVision-500 prep ===", flush=True)
    print(f"  Excluding {len(excluded)} existing on-disk IDs", flush=True)

    ds = load_dataset("MathLLMs/MathVision", split="test")
    print(f"  Source: {len(ds)} rows", flush=True)

    candidates = [i for i, row in enumerate(ds) if str(row.get("id", "")) not in excluded]
    print(f"  Candidates after exclusion: {len(candidates)}", flush=True)

    rng = np.random.default_rng(seed)
    sampled = rng.choice(candidates, size=min(n, len(candidates)), replace=False).tolist()

    records: list[dict] = []
    t0 = time.time()
    n_drop = 0
    for k, idx in enumerate(sampled):
        row = ds[int(idx)]
        # MathVision: image field is named `decoded_image` in some versions, or
        # multiple image{1..} fields. Try common names.
        pil = _to_pil(row.get("decoded_image"))
        if pil is None:
            for c in ("image", "image1", "image_1"):
                pil = _to_pil(row.get(c))
                if pil is not None:
                    break
        if pil is None:
            n_drop += 1
            continue

        image_tokens = encode_image_to_tokens(pil, vq_model)
        if image_tokens is None:
            n_drop += 1
            continue

        rid = str(row.get("id", f"mv_{int(idx)}"))
        img_path = images_dir / f"{rid}.png"
        if not img_path.exists():
            try:
                pil.save(img_path, "PNG")
            except Exception as e:  # noqa: BLE001
                print(f"  warn save: {e}", flush=True)

        question_text = _format_question_with_options(
            row.get("question", ""), row.get("options")
        )

        extra = {
            "index": rid,
            "answer": str(row.get("answer", "")),
            "answer_type": "mc" if row.get("options") else "free",
            "subject": row.get("subject", ""),
            "level": str(row.get("level", "")),
            "image_path": str(img_path),
        }
        records.append(build_record(
            data_source="mathvision",
            image_token_str=image_tokens,
            question_text=question_text,
            ground_truth=str(row.get("answer", "")),
            extra_info=extra,
        ))

        if (k + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f"  [mathvision] [{k+1}/{len(sampled)}] {(k+1)/max(elapsed,1e-6):.2f}/s drop={n_drop}",
                  flush=True)

    write_parquet(records, out_dir / "mathvision_eval.parquet")
    print(f"  Wrote {len(records)} rows ({n_drop} dropped)", flush=True)
    report_split_stats("mathvision", records)
    return records


def prepare_mathverse(out_dir: Path, images_dir: Path, vq_model, n: int, seed: int):
    """Sample n MathVerse Vision-Only rows, IBQ-encode, write parquet."""
    from datasets import load_dataset

    print(f"\n=== MathVerse Vision-Only-500 prep ===", flush=True)
    # MathVerse: split is "testmini" with a `problem_version` field selecting
    # one of {"Text Only", "Text Lite", "Text Dominant", "Vision Intensive",
    # "Vision Dominant", "Vision Only"}. We want "Vision Only".
    ds = load_dataset("AI4Math/MathVerse", "testmini", split="testmini")
    print(f"  Source: {len(ds)} rows in testmini", flush=True)

    vo_idx = [i for i, row in enumerate(ds)
              if str(row.get("problem_version", "")).strip() == "Vision Only"]
    print(f"  Vision-Only subset: {len(vo_idx)} rows", flush=True)

    if not vo_idx:
        # Fall back: include all variants and warn (some HF mirrors may use
        # different field names).
        print("  WARN: no `Vision Only` rows found; using all testmini rows",
              flush=True)
        vo_idx = list(range(len(ds)))

    rng = np.random.default_rng(seed)
    sampled = rng.choice(vo_idx, size=min(n, len(vo_idx)), replace=False).tolist()

    records: list[dict] = []
    t0 = time.time()
    n_drop = 0
    for k, idx in enumerate(sampled):
        row = ds[int(idx)]
        pil = _to_pil(row.get("decoded_image")) or _to_pil(row.get("image"))
        if pil is None:
            n_drop += 1
            continue

        image_tokens = encode_image_to_tokens(pil, vq_model)
        if image_tokens is None:
            n_drop += 1
            continue

        rid = str(row.get("sample_index", row.get("id", f"mverse_{int(idx)}")))
        img_path = images_dir / f"{rid}.png"
        if not img_path.exists():
            try:
                pil.save(img_path, "PNG")
            except Exception as e:  # noqa: BLE001
                print(f"  warn save: {e}", flush=True)

        question_text = row.get("query_cot") or row.get("query_wo") or row.get("question", "")
        extra = {
            "index": rid,
            "answer": str(row.get("answer", "")),
            "answer_type": str(row.get("answer_type", "")),
            "subject": str(row.get("subject", row.get("subfield", ""))),
            "level": str(row.get("metadata", {}).get("subfield", "")) if isinstance(row.get("metadata"), dict) else "",
            "problem_version": str(row.get("problem_version", "")),
            "image_path": str(img_path),
        }
        records.append(build_record(
            data_source="mathverse_visionOnly",
            image_token_str=image_tokens,
            question_text=question_text,
            ground_truth=str(row.get("answer", "")),
            extra_info=extra,
        ))

        if (k + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f"  [mathverse] [{k+1}/{len(sampled)}] {(k+1)/max(elapsed,1e-6):.2f}/s drop={n_drop}",
                  flush=True)

    write_parquet(records, out_dir / "mathverse_visionOnly_eval.parquet")
    print(f"  Wrote {len(records)} rows ({n_drop} dropped)", flush=True)
    report_split_stats("mathverse_vo", records)
    return records


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=str(PROJECT_ROOT / "data_prep" / "visual_math_rl"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-mathvision", type=int, default=200)
    p.add_argument("--n-mathverse", type=int, default=200)
    p.add_argument("--config", default="configs/apertus.yaml")
    p.add_argument("--only", choices=["mathvision", "mathverse", "both"], default="both")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer, vq_model = load_apertus_models(args.config)

    if args.only in ("mathvision", "both"):
        mv_imgs = out_dir / "mathvision_eval_images"
        mv_imgs.mkdir(parents=True, exist_ok=True)
        prepare_mathvision(out_dir, mv_imgs, vq_model, args.n_mathvision, args.seed)

    if args.only in ("mathverse", "both"):
        ms_imgs = out_dir / "mathverse_visionOnly_eval_images"
        ms_imgs.mkdir(parents=True, exist_ok=True)
        prepare_mathverse(out_dir, ms_imgs, vq_model, args.n_mathverse, args.seed)

    print(f"\nDone. Parquets in {out_dir}", flush=True)


if __name__ == "__main__":
    main()
