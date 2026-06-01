"""Apertus lens_search_tool for verl rollouts.

Two modes (toggled by env var LENS_TOOL_MODE, default 'cached'):

  - 'cached': lookup the per-row cached search result from
    fvqa_train_image_search_results_cache.pkl (FVQA dataset).
    The pkl is loaded once globally (thread-safe), keyed by data_id.
    Zero external-API cost — used for the smoke phase and as the
    "tool result oracle" while debugging the pipeline.

  - 'live':   call Google Cloud Vision API with WEB_DETECTION +
    LANDMARK_DETECTION + TEXT_DETECTION features on the bound
    image. Returns a formatted multi-section summary string.
    Requires GOOGLE_APPLICATION_CREDENTIALS env var pointing at a
    service-account JSON and the `google-cloud-vision` package.

Tool call from the model:
    <|tools_prefix|>[{"lens_search": {"query": "<text>"}}]<|tools_suffix|>

Image binding (at create() time from the RL parquet's tools_kwargs):
    tools_kwargs.lens_search_tool.create_kwargs = {
        "image_path": <abs path to .jpg>,
        "data_id":    <fvqa_train_NNN>,  # used by cached mode only
    }

Pattern follows tools/image_zoom_in_emu_tool.py (same BaseTool interface).
"""
from __future__ import annotations

import logging
import os
import sys
import threading
from typing import Any, Optional
from uuid import uuid4

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data_prep.lens_search_common import (
    load_fvqa_cache,
    format_cached_result,
)

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


_CACHE: Optional[dict] = None
_CACHE_LOCK = threading.Lock()
_VISION_CLIENT = None
_VISION_LOCK = threading.Lock()


def _get_cache(pkl_path: str) -> dict:
    global _CACHE
    if _CACHE is None:
        with _CACHE_LOCK:
            if _CACHE is None:
                logger.info(f"Loading FVQA search-result cache from {pkl_path}")
                _CACHE = load_fvqa_cache(pkl_path)
                logger.info(f"FVQA cache loaded: {len(_CACHE)} entries")
    return _CACHE


def _get_vision_client():
    global _VISION_CLIENT
    if _VISION_CLIENT is None:
        with _VISION_LOCK:
            if _VISION_CLIENT is None:
                from google.cloud import vision  # lazy import; live mode only
                _VISION_CLIENT = vision.ImageAnnotatorClient()
                logger.info("GCP Vision API client initialized")
    return _VISION_CLIENT


def _format_live_response(response, max_chars: int = 1800) -> str:
    """Combine WEB + LANDMARK + TEXT detection outputs into one string.

    Caps at max_chars so the tool response fits the verl
    max_tool_response_length budget (default 8192 chars, our config caps at
    2048).
    """
    parts: list[str] = []

    # WEB_DETECTION
    web = getattr(response, "web_detection", None)
    if web is not None:
        entities = []
        for e in (web.web_entities or [])[:5]:
            desc = (e.description or "").strip()
            if desc:
                entities.append(desc)
        best_guesses = [bg.label for bg in (web.best_guess_labels or [])[:1] if bg.label]
        if entities or best_guesses:
            parts.append("Web entities:")
            if best_guesses:
                parts.append(f"  Best guess: {best_guesses[0]}")
            if entities:
                parts.append("  Top entities: " + ", ".join(entities))

    # LANDMARK_DETECTION
    landmarks = getattr(response, "landmark_annotations", None) or []
    if landmarks:
        lm = landmarks[0]
        parts.append(f"Landmark: {lm.description} (score={lm.score:.2f})")

    # TEXT_DETECTION
    texts = getattr(response, "text_annotations", None) or []
    if texts:
        # textAnnotations[0] is the full text; clamp to 600 chars
        full_text = (texts[0].description or "").strip().replace("\n", " ")
        if full_text:
            if len(full_text) > 600:
                full_text = full_text[:597] + "..."
            parts.append(f"Detected text: {full_text}")

    if not parts:
        return "Search results: (no entities, landmarks, or text detected)"

    out = "Search results:\n" + "\n".join(parts)
    if len(out) > max_chars:
        out = out[: max_chars - 3] + "..."
    return out


class LensSearchTool(BaseTool):
    """Lens-style search-by-image tool. Cached (FVQA pkl) or live (GCP Vision)."""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        # Resolve mode: config beats env var; fall back to 'cached'
        self.mode: str = (config.get("mode") or os.getenv("LENS_TOOL_MODE", "cached")).strip().lower()
        if self.mode not in ("cached", "live"):
            raise ValueError(f"LensSearchTool mode must be 'cached' or 'live' (got {self.mode!r})")
        self.cache_pkl_path: str = config.get(
            "cache_pkl_path",
            "/iopsstor/scratch/cscs/badralmahouri/hf_home/hub/datasets--lmms-lab--FVQA/"
            "snapshots/bb4a4ff4c9c3fd0382d11f5d7fccd66d0b8428b5/"
            "fvqa_train_image_search_results_cache.pkl",
        )
        self.max_response_chars: int = int(config.get("max_response_chars", 1800))
        self._instance_dict: dict[str, dict[str, Any]] = {}
        logger.info(f"LensSearchTool initialized in mode={self.mode!r}")

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        if instance_id is None:
            instance_id = str(uuid4())

        create_kwargs = kwargs.get("create_kwargs", {}) or {}
        image_path = create_kwargs.get("image_path")
        data_id = create_kwargs.get("data_id")

        entry: dict[str, Any] = {
            "image_path": image_path,
            "data_id": data_id,
            "error": None,
        }
        if not image_path:
            entry["error"] = "tools_kwargs.lens_search_tool.create_kwargs.image_path is missing"
        elif self.mode == "cached" and not data_id:
            entry["error"] = "tools_kwargs.lens_search_tool.create_kwargs.data_id is required for cached mode"
        elif self.mode == "live" and not os.path.exists(image_path):
            entry["error"] = f"image_path does not exist: {image_path!r}"

        self._instance_dict[instance_id] = entry
        return instance_id, ToolResponse()

    async def execute(
        self, instance_id: str, parameters: dict[str, Any], **kwargs
    ) -> tuple[ToolResponse, float, dict]:
        entry = self._instance_dict.get(instance_id)
        if entry is None:
            return ToolResponse(text="Error: tool instance not found."), 0.0, {"success": False}

        if entry["error"]:
            return ToolResponse(text=f"Error: {entry['error']}"), 0.0, {"success": False}

        query = parameters.get("query")
        if not isinstance(query, str) or not query.strip():
            return ToolResponse(text="Error: 'query' must be a non-empty string."), 0.0, {"success": False}

        if self.mode == "cached":
            return await self._execute_cached(entry, query)
        else:
            return await self._execute_live(entry, query)

    async def _execute_cached(self, entry: dict, query: str) -> tuple[ToolResponse, float, dict]:
        try:
            cache = _get_cache(self.cache_pkl_path)
        except Exception as e:
            logger.warning(f"lens_search cached-mode load failed: {e}")
            return (
                ToolResponse(text=f"Error: failed to load FVQA cache: {e}"),
                0.0,
                {"success": False},
            )

        data_id = entry["data_id"]
        cache_entry = cache.get(data_id)
        if cache_entry is None:
            return (
                ToolResponse(text=f"Search results: (no cached results for data_id={data_id!r})"),
                0.0,
                {"success": False, "reason": "cache_miss"},
            )

        text = format_cached_result(cache_entry)
        return ToolResponse(text=text), 0.0, {"success": True, "mode": "cached"}

    async def _execute_live(self, entry: dict, query: str) -> tuple[ToolResponse, float, dict]:
        try:
            from google.cloud import vision
            client = _get_vision_client()
            with open(entry["image_path"], "rb") as f:
                content = f.read()
            image = vision.Image(content=content)
            features = [
                vision.Feature(type_=vision.Feature.Type.WEB_DETECTION),
                vision.Feature(type_=vision.Feature.Type.LANDMARK_DETECTION),
                vision.Feature(type_=vision.Feature.Type.TEXT_DETECTION),
            ]
            request = vision.AnnotateImageRequest(image=image, features=features)
            response = client.annotate_image(request)
        except Exception as e:
            logger.warning(f"lens_search live-mode call failed: {e}")
            return (
                ToolResponse(text=f"Error: GCP Vision call failed: {e}"),
                0.0,
                {"success": False},
            )

        text = _format_live_response(response, max_chars=self.max_response_chars)
        return ToolResponse(text=text), 0.0, {"success": True, "mode": "live"}

    async def release(self, instance_id: str, **kwargs) -> None:
        self._instance_dict.pop(instance_id, None)
