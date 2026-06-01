"""Apertus line_drawing_tool for verl rollouts.

Draws one or more line segments on the source image and returns the annotated
image as an IBQ token string. This mirrors the CogCoM Line(pts) manipulation:
the model supplies points, the tool creates a visual mark, and Apertus consumes
the resulting image through inline Emu3.5/IBQ image tokens.
"""

import logging
import os
import sys
import threading
from typing import Any, Optional
from uuid import uuid4

from PIL import Image, ImageDraw

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from inference.vision import encode_image, load_vq_model

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class LineDrawingEmuTool(BaseTool):
    """Draw line annotations on a source image and return IBQ token text."""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self.vq_model_path: str = config["vq_model_path"]
        self.vq_device: str = config.get("vq_device", "cuda:0")
        self.default_color: str = config.get("default_color", "red")
        self.default_width: int = int(config.get("default_width", 4))
        self.max_points: int = int(config.get("max_points", 64))

        self._instance_dict: dict[str, dict[str, Any]] = {}
        self._vq_model = None
        self._vq_lock = threading.Lock()

    def _ensure_vq_model(self):
        if self._vq_model is None:
            with self._vq_lock:
                if self._vq_model is None:
                    logger.info(f"Loading IBQ vision tokenizer from {self.vq_model_path}")
                    self._vq_model = load_vq_model(self.vq_model_path, device=self.vq_device)

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    def _parse_points(self, points: Any, image_width: int, image_height: int) -> list[tuple[int, int]]:
        if not isinstance(points, (list, tuple)):
            raise ValueError("points must be a list of [x, y] coordinates")
        if len(points) < 2:
            raise ValueError("points must contain at least two coordinates")
        if len(points) > self.max_points:
            raise ValueError(f"points must contain at most {self.max_points} coordinates")

        parsed = []
        for point in points:
            if not isinstance(point, (list, tuple)) or len(point) != 2:
                raise ValueError("each point must be [x, y]")
            x = int(round(float(point[0])))
            y = int(round(float(point[1])))
            x = max(0, min(image_width - 1, x))
            y = max(0, min(image_height - 1, y))
            parsed.append((x, y))
        return parsed

    def _parse_width(self, width: Any) -> int:
        if width is None:
            return self.default_width
        width_i = int(round(float(width)))
        return max(1, min(64, width_i))

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        if instance_id is None:
            instance_id = str(uuid4())

        create_kwargs = kwargs.get("create_kwargs", {}) or {}
        image_path = create_kwargs.get("image_path")

        entry: dict[str, Any] = {"image": None, "error": None, "image_path": image_path}
        if not image_path:
            entry["error"] = "tools_kwargs.line_drawing_tool.create_kwargs.image_path is missing"
        else:
            try:
                entry["image"] = Image.open(image_path).convert("RGB")
            except Exception as e:
                entry["error"] = f"failed to open image_path={image_path!r}: {e}"

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

        image: Image.Image = entry["image"]
        try:
            points = self._parse_points(parameters.get("points"), image.width, image.height)
            color = str(parameters.get("color") or self.default_color)
            width = self._parse_width(parameters.get("width"))
        except Exception as e:
            return ToolResponse(text=f"Error: {e}"), 0.0, {"success": False}

        try:
            annotated = image.copy()
            draw = ImageDraw.Draw(annotated)
            draw.line(points, fill=color, width=width)
            self._ensure_vq_model()
            token_str = encode_image(annotated, self._vq_model)
        except Exception as e:
            logger.warning(f"line_drawing_tool failed: {e}")
            return ToolResponse(text=f"Error: failed to draw and encode line: {e}"), 0.0, {"success": False}

        return (
            ToolResponse(text=token_str),
            0.0,
            {"success": True, "points": points, "color": color, "width": width},
        )

    async def release(self, instance_id: str, **kwargs) -> None:
        entry = self._instance_dict.pop(instance_id, None)
        if entry is not None and entry.get("image") is not None:
            try:
                entry["image"].close()
            except Exception:
                pass
