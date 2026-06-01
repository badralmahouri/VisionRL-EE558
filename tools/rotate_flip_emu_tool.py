"""Apertus rotate_flip_tool for VeRL rollouts.

Transforms the source image by rotations and/or flips and returns the resulting
image as an IBQ token string. This is useful for reading text or objects shown
sideways, tilted, upside down, or mirrored.
"""

import logging
import os
import sys
import threading
from typing import Any, Optional
from uuid import uuid4

from PIL import Image, ImageOps

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from inference.vision import encode_image, load_vq_model

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class RotateFlipEmuTool(BaseTool):
    """Rotate and/or flip a source image and return IBQ token text."""

    VALID_OPERATIONS = {
        "rotate_30",
        "rotate_45",
        "rotate_60",
        "rotate_90",
        "rotate_120",
        "rotate_135",
        "rotate_150",
        "rotate_180",
        "rotate_210",
        "rotate_225",
        "rotate_240",
        "rotate_270",
        "rotate_300",
        "rotate_315",
        "rotate_330",
        "flip_horizontal",
        "flip_vertical",
        "transpose",
        "transverse",
    }

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self.vq_model_path: str = config["vq_model_path"]
        self.vq_device: str = config.get("vq_device", "cuda:0")
        self.max_operations: int = int(config.get("max_operations", 4))
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

    def _parse_operations(self, parameters: dict[str, Any]) -> list[str]:
        operations = parameters.get("operations")
        if operations is None:
            operation = parameters.get("operation")
            operations = [operation] if operation is not None else []
        if not isinstance(operations, (list, tuple)) or not operations:
            raise ValueError("provide operation or operations")
        if len(operations) > self.max_operations:
            raise ValueError(f"operations must contain at most {self.max_operations} items")
        parsed = [str(op).strip().lower() for op in operations]
        invalid = [op for op in parsed if op not in self.VALID_OPERATIONS]
        if invalid:
            raise ValueError(f"invalid operation(s): {invalid}; valid operations are {sorted(self.VALID_OPERATIONS)}")
        return parsed

    def _apply(self, image: Image.Image, operations: list[str]) -> Image.Image:
        transformed = image.copy()
        for operation in operations:
            if operation.startswith("rotate_"):
                angle = int(operation.removeprefix("rotate_"))
                transformed = transformed.rotate(angle, expand=True, resample=Image.Resampling.BICUBIC)
            elif operation == "flip_horizontal":
                transformed = ImageOps.mirror(transformed)
            elif operation == "flip_vertical":
                transformed = ImageOps.flip(transformed)
            elif operation == "transpose":
                transformed = transformed.transpose(Image.Transpose.TRANSPOSE)
            elif operation == "transverse":
                transformed = transformed.transpose(Image.Transpose.TRANSVERSE)
        return transformed

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        if instance_id is None:
            instance_id = str(uuid4())

        create_kwargs = kwargs.get("create_kwargs", {}) or {}
        image_path = create_kwargs.get("image_path")
        entry: dict[str, Any] = {"image": None, "error": None, "image_path": image_path}
        if not image_path:
            entry["error"] = "tools_kwargs.rotate_flip_tool.create_kwargs.image_path is missing"
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

        try:
            operations = self._parse_operations(parameters)
            transformed = self._apply(entry["image"], operations)
            self._ensure_vq_model()
            token_str = encode_image(transformed, self._vq_model)
        except Exception as e:
            logger.warning(f"rotate_flip_tool failed: {e}")
            return ToolResponse(text=f"Error: failed to rotate/flip and encode image: {e}"), 0.0, {"success": False}

        return ToolResponse(text=token_str), 0.0, {"success": True, "operations": operations}

    async def release(self, instance_id: str, **kwargs) -> None:
        entry = self._instance_dict.pop(instance_id, None)
        if entry is not None and entry.get("image") is not None:
            try:
                entry["image"].close()
            except Exception:
                pass
