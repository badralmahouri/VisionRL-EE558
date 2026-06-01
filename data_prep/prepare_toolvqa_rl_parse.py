"""Parse local ToolVQA samples into Apertus RL parquet.

Keeps text-answer ToolVQA examples, wires one visual tool (line_drawing_tool or
rotate_flip_tool, selected with --tool-name) plus a final display_answers
instruction, and writes train/val/test parquet for VeRL GRPO. Pass --no-tool to
build the tool-free ablation.

Example:
    export PYTHONPATH="/users/$USER/capscratch/Emu3.5/src:/users/$USER/capscratch/verl-apertus:${PYTHONPATH:-}"
    python data_prep/prepare_toolvqa_rl_parse.py --limit 8
    python data_prep/prepare_toolvqa_rl_parse.py --tool-name rotate_flip_tool --limit 8
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
TOOLVQA_RL_INSTRUCTIONS = {
    "line_drawing_tool": (
        "Answer the visual question. You may use line_drawing_tool to draw helpful "
        "lines on the image, for example to connect objects, align locations, compare "
        "positions, or mark spatial relationships. Call the display_answers tool "
        "exactly once at the end of your response, passing the final answer as a short string."
    ),
    "rotate_flip_tool": (
        "Answer the visual question. You may use rotate_flip_tool to rotate or flip "
        "the image when content appears sideways, upside down, or mirrored. Call the "
        "display_answers tool exactly once at the end of your response, passing the "
        "final answer as a short string."
    ),
}
TOOL_NAMES = tuple(TOOLVQA_RL_INSTRUCTIONS)
TOOLVQA_NO_TOOL_INSTRUCTION = (
    "Answer the visual question directly from the image. Return only the final answer "
    "as a short string, with no tool calls."
)
IMAGE_ANSWER_PREFIXES = ("image/", "images/", "output/")
IMAGE_ANSWER_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp")


def load_config(path: str) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def load_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
                if limit is not None and len(rows) >= limit:
                    break
    return rows


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


def split_indices(
    n: int,
    seed: int,
    train_size: int | None,
    val_size: int | None,
    test_size: int | None,
    val_ratio: float | None,
    test_ratio: float | None,
) -> tuple[set[int], set[int]]:
    if n <= 1:
        return set(), set()
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)

    if train_size is not None and val_size is not None and test_size is not None:
        requested = train_size + val_size + test_size
        if n < requested:
            raise ValueError(
                f"Need at least {requested} filtered rows for fixed split "
                f"train={train_size}, val={val_size}, test={test_size}; got {n}. "
                "Increase TOOLVQA_NUM_SAMPLES or relax filters."
            )
        val_start = train_size
        test_start = train_size + val_size
        val_idx = set(perm[val_start:test_start].tolist())
        test_idx = set(perm[test_start:test_start + test_size].tolist())
        return val_idx, test_idx

    vr = 0.10 if val_ratio is None else val_ratio
    tr = 0.10 if test_ratio is None else test_ratio
    n_test = max(1, int(round(n * tr))) if tr > 0 else 0
    n_val = max(1, int(round(n * vr))) if vr > 0 and n - n_test > 1 else 0
    test_idx = set(perm[:n_test].tolist())
    val_idx = set(perm[n_test:n_test + n_val].tolist())
    return val_idx, test_idx


def build_user_content(
    question: str,
    image_token_str: str,
    no_tool: bool = False,
    tool_name: str = "line_drawing_tool",
) -> str:
    instruction = TOOLVQA_NO_TOOL_INSTRUCTION if no_tool else TOOLVQA_RL_INSTRUCTIONS[tool_name]
    return f"{image_token_str}\n{question.strip()}\n\n{instruction}"


def build_parquet_record(
    meta: dict[str, Any],
    image_token_str: str,
    image_path: Path,
    split: str,
    no_tool: bool = False,
    tool_name: str = "line_drawing_tool",
) -> dict[str, Any]:
    answer = str(meta["answer"]).strip()
    qid = meta["question_id"]
    extra_info = {
        "index": qid,
        "split": split,
        "answer": answer,
        "toolvqa_type": meta.get("type"),
        "toolvqa_tools": [c.get("name") for c in (meta.get("context") or []) if isinstance(c, dict)],
        "no_tool_ablation": no_tool,
    }
    if not no_tool:
        extra_info.update(
            {
                "need_tools_kwargs": True,
                "tools_kwargs": {
                    tool_name: {
                        "create_kwargs": {"image_path": str(image_path)},
                    },
                },
            }
        )

    return {
        "data_source": "toolvqa_rl",
        "agent_name": "single_turn_agent" if no_tool else "cof_tool_agent",
        "prompt": [
            {"role": "system", "content": APERTUS_NO_TOOL_SYSTEM if no_tool else APERTUS_TOOL_SYSTEM},
            {"role": "user", "content": build_user_content(meta["question"], image_token_str, no_tool=no_tool, tool_name=tool_name)},
        ],
        "ability": "toolvqa_visual_reasoning_no_tool" if no_tool else "toolvqa_visual_tool_reasoning",
        "reward_model": {"style": "rule", "ground_truth": answer},
        "extra_info": extra_info,
    }


def write_metadata(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse ToolVQA into Apertus RL parquet")
    parser.add_argument("--input", default="data_prep/toolvqa/metadata.jsonl")
    parser.add_argument("--image-dir", default="data_prep/toolvqa/images")
    parser.add_argument("--output-dir", default="data_prep/toolvqa_rl")
    parser.add_argument("--config", default="configs/apertus.yaml")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--val-ratio", type=float, default=None, help="Optional ratio split. Ignored when fixed split sizes are set.")
    parser.add_argument("--test-ratio", type=float, default=None, help="Optional ratio split. Ignored when fixed split sizes are set.")
    parser.add_argument("--train-size", type=int, default=950)
    parser.add_argument("--val-size", type=int, default=50)
    parser.add_argument("--test-size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--vq-device", default="cuda:0")
    parser.add_argument("--max-answer-chars", type=int, default=80)
    parser.add_argument(
        "--keep-tools",
        nargs="*",
        default=None,
        help="Optional filter: keep only rows whose ToolVQA context contains one of these tool names.",
    )
    parser.add_argument("--no-tool", action="store_true", help="Build direct-answer records without tool instructions or tool kwargs.")
    parser.add_argument(
        "--tool-name",
        default="line_drawing_tool",
        choices=list(TOOL_NAMES),
        help="Visual tool wired into the with-tool records (ignored when --no-tool).",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    image_dir = Path(args.image_dir)
    output_dir = Path(args.output_dir)
    config = load_config(args.config)

    rows = load_jsonl(input_path, args.limit)
    if not rows:
        raise ValueError(f"no rows found in {input_path}")

    kept: list[dict[str, Any]] = []
    skipped_answer = 0
    skipped_tool = 0
    missing_images = 0
    keep_tools = set(args.keep_tools or [])

    for row in rows:
        if not is_text_answer(row.get("answer"), args.max_answer_chars):
            skipped_answer += 1
            continue
        tool_names = {c.get("name") for c in (row.get("context") or []) if isinstance(c, dict)}
        if keep_tools and not (tool_names & keep_tools):
            skipped_tool += 1
            continue
        image_path = image_dir / f"{row['question_id']}.jpg"
        if not image_path.exists():
            missing_images += 1
            continue
        kept.append(row)

    if not kept:
        raise ValueError("no rows left after filtering")

    print(f"Loaded {len(rows)} rows from {input_path}")
    print(f"Kept {len(kept)} text-answer rows")
    print(f"Skipped non-text/long answers: {skipped_answer}")
    print(f"Skipped by tool filter: {skipped_tool}")
    print(f"Skipped missing images: {missing_images}")
    print(f"No-tool ablation: {args.no_tool}")
    print(f"Tool: {'none' if args.no_tool else args.tool_name}")
    print(f"Loading IBQ vision tokenizer from {config['model']['vq_model']} ...")
    vq_model = load_vq_model(config["model"]["vq_model"], device=args.vq_device)

    val_idx, test_idx = split_indices(
        len(kept),
        seed=args.seed,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )
    records_by_split: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    metadata_rows: list[dict[str, Any]] = []

    for i, row in enumerate(kept):
        split = "test" if i in test_idx else "val" if i in val_idx else "train"
        image_path = (image_dir / f"{row['question_id']}.jpg").resolve()
        image = Image.open(image_path).convert("RGB")
        image = smart_resize(image)
        image_token_str = encode_image(image, vq_model)
        meta = {**row, "image_path": str(image_path), "split": split}
        metadata_rows.append(meta)
        records_by_split[split].append(
            build_parquet_record(meta, image_token_str, image_path, split, no_tool=args.no_tool, tool_name=args.tool_name)
        )
        if (i + 1) % 50 == 0 or i == len(kept) - 1:
            print(f"  encoded {i + 1}/{len(kept)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    write_metadata(output_dir / "metadata.jsonl", metadata_rows)
    for split, records in records_by_split.items():
        pq.write_table(pa.Table.from_pylist(records), output_dir / f"{split}.parquet")
        print(f"Wrote {len(records)} {split} rows to {output_dir / f'{split}.parquet'}")
    print(f"Wrote {len(metadata_rows)} metadata rows to {output_dir / 'metadata.jsonl'}")


if __name__ == "__main__":
    main()
