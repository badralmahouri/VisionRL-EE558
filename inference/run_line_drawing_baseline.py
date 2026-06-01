"""Baseline inference on the synthetic line-drawing test set.

Supports Apertus and Qwen baselines on the same held-out synthetic test set used
by the RL checkpoint. Outputs prediction JSONL compatible with
`evaluation/compute_accuracy.py --mode mcq`.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.multiprocessing as mp
import yaml
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from inference.vision import encode_image, load_vq_model

APERTUS_SYSTEM_PROMPT = (
    "You are Apertus, a helpful assistant created by the SwissAI initiative.\n"
    "Answer with a single letter: A, B, C, or D."
)
LINE_PROMPT_SUFFIX = "\n\nAnswer with only one letter: A, B, C, or D."


def load_config(path: str) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def load_metadata(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                row["answers"] = [row["answer"]]
                rows.append(row)
                if limit is not None and len(rows) >= limit:
                    break
    return rows


def extract_choice(text: str) -> str:
    clean = re.sub(r"<\|[^|]+?\|>", " ", text).strip()
    match = re.search(r"\b([ABCD])\b", clean, flags=re.IGNORECASE)
    if match:
        return match.group(1).upper()
    match = re.search(r"([ABCD])", clean, flags=re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return clean[:1].upper() if clean else ""


def build_apertus_prompt(image_token_str: str, question: str) -> str:
    return (
        "<s>"
        f"<|system_start|>{APERTUS_SYSTEM_PROMPT}<|system_end|>"
        "<|developer_start|>"
        "Deliberation: disabled\n"
        "Tool Capabilities: disabled"
        "<|developer_end|>"
        "<|user_start|>"
        f"{image_token_str}\n{question}{LINE_PROMPT_SUFFIX}"
        "<|user_end|>"
        "<|assistant_start|>"
    )


def encode_apertus_images(metadata: list[dict[str, Any]], vq_model_path: str, device: str) -> dict[int, str]:
    print(f"Loading VQ model from {vq_model_path} ...")
    vq_model = load_vq_model(vq_model_path, device=device)
    image_tokens: dict[int, str] = {}
    start = time.time()
    for i, row in enumerate(metadata):
        image = Image.open(row["image_path"]).convert("RGB")
        image_tokens[row["question_id"]] = encode_image(image, vq_model)
        if (i + 1) % 50 == 0 or i == len(metadata) - 1:
            print(f"  Encoded [{i + 1}/{len(metadata)}] images ({time.time() - start:.1f}s)")
    del vq_model
    torch.cuda.empty_cache()
    return image_tokens


def apertus_worker(gpu_id: int, samples: list[dict[str, Any]], image_tokens: dict[int, str], config: dict[str, Any], output_path: Path) -> None:
    device = f"cuda:{gpu_id}"
    checkpoint = config["model"]["checkpoint"]
    max_new_tokens = config["generation"]["max_new_tokens"]

    print(f"[GPU {gpu_id}] Loading Apertus from {checkpoint} ...")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()

    stop_ids = [tokenizer.eos_token_id]
    assistant_end_id = tokenizer.convert_tokens_to_ids("<|assistant_end|>")
    if isinstance(assistant_end_id, int) and assistant_end_id != tokenizer.unk_token_id:
        stop_ids.append(assistant_end_id)

    results = []
    start = time.time()
    for i, sample in enumerate(samples):
        qid = sample["question_id"]
        prompt = build_apertus_prompt(image_tokens[qid], sample["question"])
        input_ids = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False).to(device)
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                eos_token_id=stop_ids,
                pad_token_id=tokenizer.pad_token_id,
            )
        generated_ids = output_ids[0, input_ids.shape[1]:]
        raw_prediction = tokenizer.decode(generated_ids, skip_special_tokens=False).strip()
        results.append({
            "question_id": qid,
            "question": sample["question"],
            "prediction": extract_choice(raw_prediction),
            "raw_prediction": raw_prediction,
            "answers": sample["answers"],
            "prompt_tokens": int(input_ids.shape[1]),
            "generated_tokens": int(generated_ids.shape[0]),
        })
        if (i + 1) % 10 == 0 or i == len(samples) - 1:
            print(f"[GPU {gpu_id}] [{i + 1}/{len(samples)}] {(i + 1) / (time.time() - start):.2f} samples/s")

    with output_path.open("w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[GPU {gpu_id}] Wrote {len(results)} predictions to {output_path}")


def qwen_worker(gpu_id: int, samples: list[dict[str, Any]], config: dict[str, Any], output_path: Path) -> None:
    device = f"cuda:{gpu_id}"
    checkpoint = config["model"]["checkpoint"]
    max_new_tokens = config["generation"]["max_new_tokens"]

    print(f"[GPU {gpu_id}] Loading Qwen from {checkpoint} ...")
    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    processor = AutoProcessor.from_pretrained(checkpoint)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        checkpoint,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    model.eval()

    results = []
    start = time.time()
    for i, sample in enumerate(samples):
        image = Image.open(sample["image_path"]).convert("RGB")
        messages = [
            {"role": "system", "content": "Answer with only one letter: A, B, C, or D."},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": f"{sample['question']}{LINE_PROMPT_SUFFIX}"},
                ],
            },
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(device)
        prompt_len = inputs.input_ids.shape[1]
        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        generated_ids = output_ids[0, prompt_len:]
        raw_prediction = processor.decode(generated_ids, skip_special_tokens=True).strip()
        results.append({
            "question_id": sample["question_id"],
            "question": sample["question"],
            "prediction": extract_choice(raw_prediction),
            "raw_prediction": raw_prediction,
            "answers": sample["answers"],
            "prompt_tokens": int(prompt_len),
            "generated_tokens": int(generated_ids.shape[0]),
        })
        if (i + 1) % 10 == 0 or i == len(samples) - 1:
            print(f"[GPU {gpu_id}] [{i + 1}/{len(samples)}] {(i + 1) / (time.time() - start):.2f} samples/s")

    with output_path.open("w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[GPU {gpu_id}] Wrote {len(results)} predictions to {output_path}")


def run_parallel(model_name: str, metadata: list[dict[str, Any]], config: dict[str, Any], output_dir: Path) -> Path:
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise RuntimeError("No GPUs available. Run this via SLURM or an interactive GPU node.")
    print(f"Available GPUs: {num_gpus}")

    chunks = [[] for _ in range(num_gpus)]
    for i, sample in enumerate(metadata):
        chunks[i % num_gpus].append(sample)

    image_tokens = None
    if model_name == "apertus":
        image_tokens = encode_apertus_images(metadata, config["model"]["vq_model"], device="cuda:0")

    mp.set_start_method("spawn", force=True)
    partial_paths = []
    processes = []
    for gpu_id, chunk in enumerate(chunks):
        partial_path = output_dir / f"predictions_gpu{gpu_id}.jsonl"
        partial_paths.append(partial_path)
        if model_name == "apertus":
            proc = mp.Process(target=apertus_worker, args=(gpu_id, chunk, image_tokens, config, partial_path))
        else:
            proc = mp.Process(target=qwen_worker, args=(gpu_id, chunk, config, partial_path))
        proc.start()
        processes.append(proc)

    for proc in processes:
        proc.join()
    for gpu_id, proc in enumerate(processes):
        if proc.exitcode != 0:
            raise RuntimeError(f"Worker GPU {gpu_id} failed with exit code {proc.exitcode}")

    final_path = output_dir / "predictions.jsonl"
    merged = []
    for partial_path in partial_paths:
        with partial_path.open(encoding="utf-8") as f:
            for line in f:
                merged.append(json.loads(line))
        partial_path.unlink(missing_ok=True)
    merged.sort(key=lambda row: row["question_id"])
    with final_path.open("w", encoding="utf-8") as f:
        for row in merged:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return final_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Line-drawing baseline inference")
    parser.add_argument("--model", choices=["apertus", "qwen"], required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--metadata", default="data_prep/line_drawing_rl/test_metadata.jsonl")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    metadata_path = Path(args.metadata)
    output_dir = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / "results" / "line_drawing_rl" / f"{args.model}_baseline"
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = load_metadata(metadata_path, args.limit)
    print(f"Loaded {len(metadata)} line-drawing samples from {metadata_path}")
    final_path = run_parallel(args.model, metadata, config, output_dir)
    print("\n=== Inference Complete ===")
    print(f"Output: {final_path}")


if __name__ == "__main__":
    main()
