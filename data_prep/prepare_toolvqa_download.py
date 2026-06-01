"""Download/sample ToolVQA into local metadata.jsonl + images.

This is separate from prepare_evals_all.py because RL needs more than the old
200-example smoke subset. It writes the same local layout consumed by
prepare_toolvqa_rl_parse.py:

    data_prep/toolvqa/metadata.jsonl
    data_prep/toolvqa/images/<question_id>.jpg

Examples:
    python data_prep/prepare_toolvqa_download.py --num-samples 1000 --force
    python data_prep/prepare_toolvqa_download.py --num-samples 5000 --text-only --force
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from datasets import load_dataset
from huggingface_hub import hf_hub_download
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT_DIR = PROJECT_ROOT / "data_prep" / "toolvqa"
IMAGE_ANSWER_PREFIXES = ("image/", "images/", "output/")
IMAGE_ANSWER_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp")


def is_text_answer(answer: Any, max_answer_chars: int) -> bool:
    if answer is None:
        return False
    text = str(answer).strip()
    if not text or len(text) > max_answer_chars:
        return False
    low = text.lower()
    if low.startswith(IMAGE_ANSWER_PREFIXES) or low.endswith(IMAGE_ANSWER_SUFFIXES):
        return False
    return True


def load_existing_ids(metadata_path: Path) -> set[int]:
    ids: set[int] = set()
    if not metadata_path.exists():
        return ids
    with metadata_path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    ids.add(int(json.loads(line)["question_id"]))
                except Exception:
                    continue
    return ids


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample/download ToolVQA rows and images")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true", help="Overwrite existing metadata/images selection.")
    parser.add_argument("--text-only", action="store_true", help="Sample only rows with text answers usable by the current RL reward.")
    parser.add_argument("--max-answer-chars", type=int, default=80)
    parser.add_argument("--keep-tools", nargs="*", default=None, help="Optional filter by ToolVQA context tool names.")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    img_dir = out_dir / "images"
    metadata_path = out_dir / "metadata.jsonl"

    if metadata_path.exists() and not args.force:
        existing = load_existing_ids(metadata_path)
        if len(existing) >= args.num_samples:
            print(f"Existing {metadata_path} has {len(existing)} rows; use --force to rebuild.")
            return
        print(f"Existing {metadata_path} has only {len(existing)} rows; rebuilding to reach {args.num_samples}.")

    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir.mkdir(parents=True, exist_ok=True)

    print("Loading DietCoke4671/ToolVQA train.jsonl ...")
    ds = load_dataset(
        "json",
        data_files="hf://datasets/DietCoke4671/ToolVQA/train.jsonl",
        split="train",
    )
    print(f"Dataset rows: {len(ds)}")

    rng = random.Random(args.seed)
    indices = list(range(len(ds)))
    rng.shuffle(indices)
    keep_tools = set(args.keep_tools or [])

    records: list[dict[str, Any]] = []
    skipped_answer = 0
    skipped_tool = 0
    downloaded = 0

    for idx in indices:
        if len(records) >= args.num_samples:
            break
        row = ds[idx]
        if args.text_only and not is_text_answer(row.get("answer"), args.max_answer_chars):
            skipped_answer += 1
            continue
        tool_names = {c.get("name") for c in (row.get("context") or []) if isinstance(c, dict)}
        if keep_tools and not (tool_names & keep_tools):
            skipped_tool += 1
            continue

        image_path_in_repo = row["image_path"]
        local_path = hf_hub_download(
            repo_id="DietCoke4671/ToolVQA",
            filename=image_path_in_repo,
            repo_type="dataset",
        )
        qid = int(idx)
        out_img = img_dir / f"{qid}.jpg"
        Image.open(local_path).convert("RGB").save(out_img, "JPEG")
        downloaded += 1

        records.append({
            "question_id": qid,
            "image_path": image_path_in_repo,
            "question": row.get("question"),
            "context": row.get("context"),
            "ori_question": row.get("ori_question"),
            "thought_rethink": row.get("thought_rethink"),
            "thought_question": row.get("question"),
            "answer": row.get("answer"),
            "type": row.get("type"),
        })
        if len(records) % 100 == 0:
            print(f"  saved {len(records)}/{args.num_samples}")

    if len(records) < args.num_samples:
        print(f"WARNING: requested {args.num_samples}, saved {len(records)} after filtering.")
    with metadata_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Saved {len(records)} rows to {metadata_path}")
    print(f"Saved images to {img_dir}")
    print(f"Skipped non-text answers: {skipped_answer}")
    print(f"Skipped by tool filter: {skipped_tool}")
    print(f"Downloaded images: {downloaded}")


if __name__ == "__main__":
    main()
