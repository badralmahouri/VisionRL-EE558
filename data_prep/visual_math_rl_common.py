"""Shared helpers for visual-math RL data prep scripts.

The 3 prepare_visual_math_rl_*.py scripts (Math-VR-train, DynaMath, eval) all
need the same: load tokenizer + IBQ vq_model, encode an image, build a VeRL
parquet record with the right schema.

Reused by:
  - data_prep/prepare_visual_math_rl_mathvr.py
  - data_prep/prepare_visual_math_rl_dynamath.py
  - data_prep/prepare_visual_math_rl_eval.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Iterable

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from inference.vision import encode_image  # noqa: E402

# --- constants used in user-prompt construction ----------------------------

APERTUS_SYSTEM = "You are a helpful assistant."

# Path-B format instruction (free-prose; no display_answers).
# Reward function anchors on "Final answer:" to make extraction reliable.
FORMAT_INSTRUCTION = (
    "Solve the problem step by step. "
    "Write 'Final answer: X' on the LAST line of your response, "
    "where X is your final answer (just the number or letter, no extra text)."
)


# --- IBQ encoding wrapper --------------------------------------------------


def encode_image_to_tokens(image: Image.Image, vq_model) -> str | None:
    """Return Apertus IBQ token string for an image, or None on failure."""
    try:
        return encode_image(image.convert("RGB"), vq_model)
    except Exception as exc:  # noqa: BLE001
        print(f"  encode_image failed: {exc}", flush=True)
        return None


# --- VeRL parquet record builder -------------------------------------------


def build_record(
    data_source: str,
    image_token_str: str,
    question_text: str,
    ground_truth: str,
    extra_info: dict,
    ability: str = "visual_math",
) -> dict:
    """Build a single VeRL-compatible parquet record (no-tool, free-prose).

    Schema matches what `data_prep/calc_eval/*/eval.parquet` use, minus the
    tool-agent fields. VeRL reads `prompt` as a list of {role, content} dicts
    and renders the final chat template at runtime.

    Args:
        data_source: e.g. "math_vr_train", "dynamath", "mathvision_500".
        image_token_str: pre-encoded IBQ token string for the image.
        question_text: raw question text from the source dataset.
        ground_truth: gold answer string (or list of strings).
        extra_info: per-row metadata (subject, level, image_path, ...).
        ability: tag used by VeRL for ability-bucket logging.
    """
    question_clean = question_text.strip()
    user_content = (
        f"{image_token_str}\n\n"
        f"{question_clean}\n\n"
        f"{FORMAT_INSTRUCTION}"
    )
    # Ensure v3 LLM-judge reward has access to the question text (the prompt's
    # user_content is pre-rendered with IBQ image tokens and the FORMAT
    # instruction, so we can't easily recover the plain question from it at
    # reward time). Idempotent: only set if caller didn't already.
    extra_info = dict(extra_info) if extra_info else {}
    extra_info.setdefault("question", question_clean)
    return {
        "data_source": data_source,
        # verl-builtin "single_turn_agent" = naive single-turn chat completion,
        # no tools. Defined at verl/experimental/agent_loop/single_turn_agent_loop.py.
        # An empty string here would trigger:
        #   "AssertionError: Agent loop  not registered, registered agent loops:
        #    dict_keys(['single_turn_agent', 'tool_agent', 'cof_tool_agent'])"
        "agent_name": "single_turn_agent",
        "prompt": [
            {"role": "system", "content": APERTUS_SYSTEM},
            {"role": "user", "content": user_content},
        ],
        "ability": ability,
        "reward_model": {"style": "rule", "ground_truth": str(ground_truth)},
        "extra_info": extra_info,
    }


def write_parquet(records: list[dict], path: Path) -> int:
    """Write records to a parquet file. Returns row count."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(records), path)
    return len(records)


# --- Math-VR-train answer extraction ---------------------------------------

# Math-VR-train `analysis` ends with: "Therefore, the answer is: X." style
# prose with optional LaTeX wrappers. Examples (from verification):
#   "Therefore, the answer is $4\\sqrt{3}$."
#   "Therefore, the answer is: **B**."
#   "Therefore, the answer is: $\\dfrac{30}{13}$."
#   "**Therefore, the answer is:** $3$."
_MATHVR_ANSWER_RE = re.compile(
    r"(?:therefore|thus|hence)[, ]*\**\s*(?:the\s+)?answer\s+is\s*:?\s*\**\s*"
    r"\$?([^$\.\n]+?)\$?\s*\.?\s*$",
    re.IGNORECASE | re.DOTALL,
)


def extract_mathvr_answer(analysis_text: str) -> str | None:
    """Pull the final answer string out of a Math-VR-train `analysis` field.

    Returns None if no anchored "Therefore, the answer is X" can be found.
    Caller should skip such rows.
    """
    if not analysis_text:
        return None

    # Take only the last 200 chars to anchor on the FINAL "therefore the answer
    # is" if the analysis has multiple intermediate ones.
    tail = analysis_text.strip()
    last_chunk = tail[-400:] if len(tail) > 400 else tail

    m = _MATHVR_ANSWER_RE.search(last_chunk)
    if not m:
        return None

    raw = m.group(1).strip()
    # Strip common decorators
    raw = raw.strip("*").strip()
    raw = raw.strip("$").strip()
    return raw or None


# --- HF dataset loader -----------------------------------------------------


def load_apertus_models(config_path: str | None = None):
    """Load Apertus tokenizer + Emu3.5 IBQ vision tokenizer.

    Defaults to configs/apertus.yaml paths. Override with config_path.
    """
    import yaml
    from transformers import AutoTokenizer
    from inference.vision import load_vq_model

    cfg_path = Path(config_path) if config_path else PROJECT_ROOT / "configs" / "apertus.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    print(f"Loading Apertus tokenizer from {cfg['model']['checkpoint']} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["model"]["checkpoint"], trust_remote_code=True
    )
    print(f"Loading IBQ vision tokenizer from {cfg['model']['vq_model']} ...", flush=True)
    vq_model = load_vq_model(cfg["model"]["vq_model"], device="cuda:0")
    print("Models loaded.", flush=True)
    return tokenizer, vq_model


# --- sanity helpers --------------------------------------------------------


def report_split_stats(label: str, records: Iterable[dict]) -> None:
    """Print row count + ID samples + question-length stats for a split."""
    records = list(records)
    if not records:
        print(f"  [{label}] empty split", flush=True)
        return
    q_lens = [len(r["prompt"][1]["content"]) for r in records]
    q_lens.sort()
    n = len(records)
    p50 = q_lens[n // 2]
    p95 = q_lens[int(n * 0.95)] if n > 1 else q_lens[0]
    print(
        f"  [{label}] n={n}, prompt-char p50={p50}, p95={p95}, max={q_lens[-1]}",
        flush=True,
    )
