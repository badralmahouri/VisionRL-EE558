"""Parse raw line_drawing_tool data into Apertus RL parquet.

This is the GPU/Apertus stage. It loads raw.jsonl from
prepare_line_drawing_rl_generate.py, encodes images with Emu3.5/IBQ, and writes
metadata.jsonl plus train/val parquet files for VeRL GRPO training. If test_raw.jsonl exists, it also writes test_metadata.jsonl and test.parquet.

Example:
    export PYTHONPATH="/users/$USER/capscratch/Emu3.5/src:/users/$USER/capscratch/verl-apertus:${PYTHONPATH:-}"
    python data_prep/prepare_line_drawing_rl_parse.py --limit 8
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from inference.vision import encode_image, load_vq_model, smart_resize

APERTUS_TOOL_SYSTEM = "You are a helpful assistant with access to tools."
APERTUS_NO_TOOL_SYSTEM = "You are a helpful assistant."
LINE_RL_INSTRUCTION = (
    "Use the line_drawing_tool to draw a straight line connecting the centers "
    "of the red and blue dots. Then answer with the single letter of the labeled "
    "target that the line passes through. Call the display_answers tool exactly "
    "once at the end of your response."
)
LINE_RL_NO_TOOL_INSTRUCTION = (
    "Answer the visual question directly. Identify which labeled target, A, B, C, or D, "
    "lies on the straight line connecting the centers of the red and blue dots. "
    "Return only the single letter."
)


def load_config(path: str) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def split_indices(n: int, val_ratio: float, seed: int) -> set[int]:
    if n <= 1:
        return set()
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_val = max(1, int(round(n * val_ratio)))
    return set(perm[:n_val].tolist())


def build_user_content(question: str, image_token_str: str, no_tool: bool = False) -> str:
    instruction = LINE_RL_NO_TOOL_INSTRUCTION if no_tool else LINE_RL_INSTRUCTION
    return f"{image_token_str}\n{question}\n\n{instruction}"


def build_parquet_record(
    meta: dict[str, Any],
    image_token_str: str,
    image_path: Path,
    split: str,
    no_tool: bool = False,
) -> dict[str, Any]:
    extra_info = {
        "index": meta["question_id"],
        "split": split,
        "answer": meta["answer"],
        "line_points": meta["points"],
        "no_tool_ablation": no_tool,
    }
    if not no_tool:
        extra_info.update({
            "need_tools_kwargs": True,
            "tools_kwargs": {
                "line_drawing_tool": {
                    "create_kwargs": {"image_path": str(image_path)},
                },
            },
        })

    return {
        "data_source": "line_drawing_rl",
        "agent_name": "cof_tool_agent",
        "prompt": [
            {"role": "system", "content": APERTUS_NO_TOOL_SYSTEM if no_tool else APERTUS_TOOL_SYSTEM},
            {"role": "user", "content": build_user_content(meta["question"], image_token_str, no_tool=no_tool)},
        ],
        "ability": "spatial_line_reasoning_no_tool" if no_tool else "spatial_line_reasoning",
        "reward_model": {"style": "rule", "ground_truth": meta["answer"]},
        "extra_info": extra_info,
    }


def load_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
                if limit is not None and len(rows) >= limit:
                    break
    return rows


def encode_rows(
    rows: list[dict[str, Any]],
    output_dir: Path,
    image_root: Path,
    vq_model: Any,
    split_for_index: Any,
    no_tool: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    metadata_rows: list[dict[str, Any]] = []
    records_by_split: dict[str, list[dict[str, Any]]] = {}

    for i, row in enumerate(rows):
        image_path = (image_root / row["image_file"]).resolve()
        if not image_path.exists():
            raise FileNotFoundError(image_path)

        image = Image.open(image_path).convert("RGB")
        image = smart_resize(image)
        image_token_str = encode_image(image, vq_model)
        split = split_for_index(i)

        meta = {
            **row,
            "image_path": str(image_path),
            "split": split,
        }
        metadata_rows.append(meta)
        records_by_split.setdefault(split, []).append(build_parquet_record(meta, image_token_str, image_path, split, no_tool=no_tool))

        if (i + 1) % 50 == 0 or i == len(rows) - 1:
            print(f"  encoded {i + 1}/{len(rows)}")

    return metadata_rows, records_by_split


def write_metadata(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse line-drawing RL data into Apertus parquet")
    parser.add_argument("--input", default=None, help="Default: data_prep/line_drawing_rl/raw.jsonl")
    parser.add_argument("--test-input", default=None, help="Default: data_prep/line_drawing_rl/test_raw.jsonl if it exists")
    parser.add_argument("--output-dir", default="data_prep/line_drawing_rl")
    parser.add_argument("--image-root", default=None, help="Directory that contains image_file paths. Defaults to output-dir.")
    parser.add_argument("--no-tool", action="store_true", help="Build a no-tool ablation dataset with direct-answer prompts.")
    parser.add_argument("--config", default="configs/apertus.yaml")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--test-limit", type=int, default=None)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--vq-device", default="cuda:0")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    image_root = Path(args.image_root) if args.image_root else output_dir
    input_path = Path(args.input) if args.input else output_dir / "raw.jsonl"
    test_input_path = Path(args.test_input) if args.test_input else output_dir / "test_raw.jsonl"
    config = load_config(args.config)

    rows = load_jsonl(input_path, args.limit)
    if not rows:
        raise ValueError(f"no rows found in {input_path}")

    print(f"Loaded {len(rows)} raw rows from {input_path}")
    print(f"Image root: {image_root}")
    print(f"No-tool ablation: {args.no_tool}")
    print(f"Loading IBQ vision tokenizer from {config['model']['vq_model']} ...")
    vq_model = load_vq_model(config["model"]["vq_model"], device=args.vq_device)

    val_idx = split_indices(len(rows), args.val_ratio, args.seed)
    metadata_rows, records_by_split = encode_rows(
        rows,
        output_dir,
        image_root,
        vq_model,
        lambda i: "val" if i in val_idx else "train",
        no_tool=args.no_tool,
    )
    train_records = records_by_split.get("train", [])
    val_records = records_by_split.get("val", [])

    output_dir.mkdir(parents=True, exist_ok=True)
    write_metadata(output_dir / "metadata.jsonl", metadata_rows)
    pq.write_table(pa.Table.from_pylist(train_records), output_dir / "train.parquet")
    pq.write_table(pa.Table.from_pylist(val_records), output_dir / "val.parquet")

    print(f"\nWrote {len(metadata_rows)} metadata rows to {output_dir / 'metadata.jsonl'}")
    print(f"Wrote {len(train_records)} train rows to {output_dir / 'train.parquet'}")
    print(f"Wrote {len(val_records)} val rows to {output_dir / 'val.parquet'}")

    if test_input_path.exists():
        test_rows = load_jsonl(test_input_path, args.test_limit)
        if not test_rows:
            raise ValueError(f"no rows found in {test_input_path}")
        print(f"\nLoaded {len(test_rows)} test rows from {test_input_path}")
        test_metadata_rows, test_records_by_split = encode_rows(
            test_rows,
            output_dir,
            image_root,
            vq_model,
            lambda _i: "test",
            no_tool=args.no_tool,
        )
        test_records = test_records_by_split.get("test", [])
        write_metadata(output_dir / "test_metadata.jsonl", test_metadata_rows)
        pq.write_table(pa.Table.from_pylist(test_records), output_dir / "test.parquet")
        print(f"Wrote {len(test_metadata_rows)} test metadata rows to {output_dir / 'test_metadata.jsonl'}")
        print(f"Wrote {len(test_records)} test rows to {output_dir / 'test.parquet'}")
    else:
        print(f"\nNo test raw file found at {test_input_path}; skipping test parquet.")


if __name__ == "__main__":
    main()
