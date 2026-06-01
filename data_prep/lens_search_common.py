"""Shared helpers for Lens-search SFT and RL data prep.

- `LENS_SEARCH_TOOL` — JSON schema for the `lens_search` tool. The model emits
  a `{"query": "<text>"}` payload; the tool wrapper takes that + the bound
  image and returns search results (cached or live GCP Vision).
- `load_fvqa_cache` — load `fvqa_train_image_search_results_cache.pkl` with a
  compatibility shim for the PIL JpegImageFile pickle format mismatch
  (cache was saved with an older PIL; we don't care about PIL state inside
  it, only the text fields).
- `format_cached_result` — turn a cache entry into the string we put in the
  `{"role": "tool"}` content slot of the SFT trajectory.
- `synthesize_thinking_blocks` — template-based generator for the two inner
  `thoughts` blocks (first one before the tool call, second one before
  `display_answers`).
- `derive_query_from_question` — for SFT we need a `query` string for the
  `lens_search` call. Simplest correct thing: pass the question itself.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any


LENS_SEARCH_TOOL = {
    "name": "lens_search",
    "description": (
        "Search the web for information about an entity, landmark, object, or "
        "text visible in the user's image. Use this when the user asks a "
        "knowledge question that requires identifying an entity in the image "
        "or looking up an attribute / fact about it."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "Short natural-language query describing what to look up "
                    "about the image. Example: 'What entity is shown in this image?' "
                    "or 'What is the capacity of this building?'"
                ),
            },
        },
        "required": ["query"],
    },
}


def _install_pil_compat_shim() -> None:
    """Patch PIL.JpegImagePlugin.JpegImageFile.__setstate__ to tolerate the
    older pickle format used in the FVQA cache (PIL <12 had 8-element state;
    PIL 12+ has 7-element). We don't need the image content from the cache —
    only the text fields — so we swallow the unpack error silently.
    """
    from PIL import Image, JpegImagePlugin

    _orig = JpegImagePlugin.JpegImageFile.__setstate__

    def _compat(self, state):
        try:
            _orig(self, state)
        except ValueError:
            try:
                Image.Image.__setstate__(self, state[:5])
            except Exception:
                pass

    JpegImagePlugin.JpegImageFile.__setstate__ = _compat


def load_fvqa_cache(pkl_path: str | Path) -> dict[str, dict[str, list]]:
    """Load the FVQA image-search cache pkl with the PIL compat shim.

    Returns: {data_id: {"tool_returned_web_title_list": [str...], "tool_returned_images_urls": [str...]}}
    """
    _install_pil_compat_shim()
    with open(pkl_path, "rb") as f:
        cache: dict[str, dict[str, list]] = pickle.load(f)
    return cache


def format_cached_result(cache_entry: dict[str, list], max_titles: int = 5) -> str:
    """Render a cache entry as the string we feed into the `{"role": "tool"}` slot.

    Format: a short header + numbered list of web titles. We drop image URLs
    (model can't read them anyway) and keep just the textual signal.
    """
    titles = cache_entry.get("tool_returned_web_title_list") or []
    titles = [str(t).strip() for t in titles[:max_titles] if t]
    if not titles:
        return "Search results: (no results)"
    lines = ["Search results:"]
    for i, t in enumerate(titles, start=1):
        # Cap each title length so the tool response fits inside the
        # config's max_response_length budget.
        if len(t) > 350:
            t = t[:347] + "..."
        lines.append(f"{i}. {t}")
    return "\n".join(lines)


def synthesize_thinking_blocks(question: str, gold_answer: str) -> tuple[str, str]:
    """Return (pre_tool_thought, post_tool_thought) for SFT trajectory synthesis.

    Template-based — we want to teach FORMAT, not deep reasoning. RL will
    refine the thinking content via reward.
    """
    pre = (
        f"The user is asking: {question.strip()} "
        "I need to look up information about what's in the image. "
        "I'll use lens_search to find relevant facts."
    )
    post = (
        "Based on the search results, I can now answer the user's question. "
        f"The answer is: {gold_answer.strip()}."
    )
    return pre, post


def derive_query_from_question(question: str) -> str:
    """For first-pass SFT, the question itself is a valid query."""
    return question.strip()


def extract_question(prompt_field: Any) -> str:
    """FVQA `prompt` field is a list of message dicts: [{"content": str, "role": "user"}].

    Pull the user content string out.
    """
    if isinstance(prompt_field, str):
        return prompt_field
    if isinstance(prompt_field, list):
        for msg in prompt_field:
            if isinstance(msg, dict) and msg.get("role") == "user":
                return str(msg.get("content", ""))
    return ""


def extract_ground_truth(reward_model_field: Any) -> str:
    """FVQA `reward_model` is dict-like with `ground_truth`."""
    if isinstance(reward_model_field, dict):
        return str(reward_model_field.get("ground_truth", "")).strip()
    return ""
