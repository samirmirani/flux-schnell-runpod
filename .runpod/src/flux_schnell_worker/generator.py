from __future__ import annotations

import base64
import io
import logging
import os
import secrets
import threading
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional

import torch
from diffusers import FluxPipeline
from PIL import Image

LOGGER = logging.getLogger(__name__)

DEFAULT_MODEL_ID = os.getenv("MODEL_ID", "black-forest-labs/FLUX.1-schnell")
DEFAULT_WIDTH = int(os.getenv("DEFAULT_WIDTH", 1024))
DEFAULT_HEIGHT = int(os.getenv("DEFAULT_HEIGHT", 1024))
DEFAULT_NUM_STEPS = int(os.getenv("DEFAULT_INFERENCE_STEPS", 4))
DEFAULT_GUIDANCE = float(os.getenv("DEFAULT_GUIDANCE", 1.0))
DEFAULT_IMAGE_FORMAT = os.getenv("DEFAULT_IMAGE_FORMAT", "jpeg").lower()

MIN_RESOLUTION = int(os.getenv("MIN_RESOLUTION", 256))
MAX_RESOLUTION = int(os.getenv("MAX_RESOLUTION", 1536))
STEP_MIN = int(os.getenv("MIN_INFERENCE_STEPS", 1))
STEP_MAX = int(os.getenv("MAX_INFERENCE_STEPS", 8))
GUIDANCE_MIN = float(os.getenv("MIN_GUIDANCE", 0.0))
GUIDANCE_MAX = float(os.getenv("MAX_GUIDANCE", 10.0))
PRICE_PER_MEGAPIXEL = float(os.getenv("PRICE_PER_MEGAPIXEL", 0.0024))
MAX_SEQUENCE_LENGTH = int(os.getenv("MAX_SEQUENCE_LENGTH", 512))

SUPPORTED_FORMATS = {"jpeg": "image/jpeg", "jpg": "image/jpeg", "png": "image/png"}
USE_MOCK_PIPELINE = os.getenv("USE_MOCK_PIPELINE") == "1"
ENABLE_CPU_OFFLOAD = os.getenv("ENABLE_CPU_OFFLOAD") == "1"


class FluxInputError(ValueError):
    """Raised when the incoming payload is invalid."""


@dataclass
class FluxSchnellInput:
    prompt: str
    negative_prompt: Optional[str] = None
    width: int = DEFAULT_WIDTH
    height: int = DEFAULT_HEIGHT
    num_inference_steps: int = DEFAULT_NUM_STEPS
    guidance: float = DEFAULT_GUIDANCE
    seed: int = field(default_factory=lambda: -1)
    image_format: str = DEFAULT_IMAGE_FORMAT


@dataclass
class GenerationResult:
    image_base64: str
    mime_type: str
    seed: int
    cost: float
    width: int
    height: int
    num_inference_steps: int
    guidance: float
    image_format: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "image_base64": self.image_base64,
            "mime_type": self.mime_type,
            "seed": self.seed,
            "cost": self.cost,
            "width": self.width,
            "height": self.height,
            "num_inference_steps": self.num_inference_steps,
            "guidance": self.guidance,
            "image_format": self.image_format,
        }


class FluxSchnellGenerator:
    """Wraps Flux.1-schnell inference together with validation and formatting."""

    def __init__(self) -> None:
        self._model_id = DEFAULT_MODEL_ID
        self._device = os.getenv("DEVICE", "cuda")
        dtype_name = os.getenv("TORCH_DTYPE", "bfloat16")
        self._torch_dtype = getattr(torch, dtype_name, torch.bfloat16)
        if not hasattr(torch, dtype_name):
            LOGGER.warning(
                "Unsupported TORCH_DTYPE=%s provided; falling back to bfloat16.", dtype_name
            )
        self._hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
        self._cache_dir = os.getenv("HUGGINGFACE_HUB_CACHE")
        self._pipeline: Optional[FluxPipeline] = None
        self._pipeline_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------
    def generate(self, payload: Dict[str, Any]) -> GenerationResult:
        params = self._parse_inputs(payload)
        pipeline = self._get_pipeline()
        seed = params.seed if params.seed is not None and params.seed >= 0 else self._random_seed()

        generator = torch.Generator(self._device).manual_seed(seed)

        with torch.inference_mode():
            output = pipeline(
                prompt=params.prompt,
                negative_prompt=params.negative_prompt,
                width=params.width,
                height=params.height,
                num_inference_steps=params.num_inference_steps,
                guidance_scale=params.guidance,
                generator=generator,
                max_sequence_length=MAX_SEQUENCE_LENGTH,
            )

        image = output.images[0]
        encoded, mime_type = self._encode_image(image, params.image_format)
        cost = self._calculate_cost(params.width, params.height)

        return GenerationResult(
            image_base64=encoded,
            mime_type=mime_type,
            seed=seed,
            cost=cost,
            width=params.width,
            height=params.height,
            num_inference_steps=params.num_inference_steps,
            guidance=params.guidance,
            image_format=params.image_format,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _parse_inputs(self, payload: Dict[str, Any]) -> FluxSchnellInput:
        if not isinstance(payload, dict):
            raise FluxInputError("Job input must be an object with generation parameters.")

        prompt = payload.get("prompt")
        if not prompt or not isinstance(prompt, str):
            raise FluxInputError("`prompt` (string) is required.")

        negative_prompt = payload.get("negative_prompt")
        if negative_prompt is not None and not isinstance(negative_prompt, str):
            raise FluxInputError("`negative_prompt` must be a string if provided.")

        width = self._normalize_dimension(payload.get("width", DEFAULT_WIDTH), "width")
        height = self._normalize_dimension(payload.get("height", DEFAULT_HEIGHT), "height")

        steps_raw = payload.get("num_inference_steps", DEFAULT_NUM_STEPS)
        steps = self._normalize_int(steps_raw, STEP_MIN, STEP_MAX, "num_inference_steps")

        guidance_raw = payload.get("guidance", DEFAULT_GUIDANCE)
        guidance = self._normalize_float(guidance_raw, GUIDANCE_MIN, GUIDANCE_MAX, "guidance")

        seed_raw = payload.get("seed")
        seed = self._normalize_seed(seed_raw)

        image_format = self._normalize_format(payload.get("image_format", DEFAULT_IMAGE_FORMAT))

        return FluxSchnellInput(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance=guidance,
            seed=seed,
            image_format=image_format,
        )

    def _normalize_dimension(self, value: Any, field_name: str) -> int:
        try:
            int_value = int(value)
        except (TypeError, ValueError) as err:
            raise FluxInputError(f"`{field_name}` must be an integer.") from err

        if int_value % 64 != 0:
            raise FluxInputError(f"`{field_name}` must be divisible by 64. Received {int_value}.")

        if not (MIN_RESOLUTION <= int_value <= MAX_RESOLUTION):
            raise FluxInputError(
                f"`{field_name}` must be between {MIN_RESOLUTION} and {MAX_RESOLUTION}. Received {int_value}."
            )

        return int_value

    def _normalize_int(self, value: Any, min_value: int, max_value: int, field_name: str) -> int:
        try:
            int_value = int(value)
        except (TypeError, ValueError) as err:
            raise FluxInputError(f"`{field_name}` must be an integer.") from err

        if not (min_value <= int_value <= max_value):
            raise FluxInputError(f"`{field_name}` must be between {min_value} and {max_value}.")

        return int_value

    def _normalize_float(self, value: Any, min_value: float, max_value: float, field_name: str) -> float:
        try:
            float_value = float(value)
        except (TypeError, ValueError) as err:
            raise FluxInputError(f"`{field_name}` must be a number.") from err

        if not (min_value <= float_value <= max_value):
            raise FluxInputError(f"`{field_name}` must be between {min_value} and {max_value}.")

        return float_value

    def _normalize_seed(self, value: Any) -> int:
        if value is None:
            return -1
        try:
            seed = int(value)
        except (TypeError, ValueError) as err:
            raise FluxInputError("`seed` must be an integer if provided.") from err
        return seed

    def _normalize_format(self, value: Any) -> str:
        if not isinstance(value, str):
            raise FluxInputError("`image_format` must be a string.")
        fmt = value.lower()
        if fmt == "jpg":
            fmt = "jpeg"
        if fmt not in SUPPORTED_FORMATS:
            raise FluxInputError("`image_format` must be either 'png' or 'jpeg'.")
        return fmt

    def _get_pipeline(self) -> FluxPipeline:
        if USE_MOCK_PIPELINE and self._pipeline:
            return self._pipeline

        if USE_MOCK_PIPELINE:
            LOGGER.warning("USE_MOCK_PIPELINE=1 detected. Returning dummy pipeline for local tests.")
            self._pipeline = _MockFluxPipeline()
            return self._pipeline

        if self._pipeline:
            return self._pipeline

        with self._pipeline_lock:
            if self._pipeline:
                return self._pipeline

            model_path = Path(self._model_id)
            repo_exists_locally = model_path.exists()
            if not self._hf_token and not repo_exists_locally:
                raise RuntimeError(
                    "HF_TOKEN environment variable is required to download black-forest-labs/FLUX.1-schnell."
                )

            LOGGER.info("Loading pipeline for %s", self._model_id)
            hf_token = None if repo_exists_locally else self._hf_token
            pipeline = FluxPipeline.from_pretrained(
                self._model_id,
                torch_dtype=self._torch_dtype,
                token=hf_token,
                cache_dir=self._cache_dir,
                use_safetensors=True,
            )
            if ENABLE_CPU_OFFLOAD:
                LOGGER.info("ENABLE_CPU_OFFLOAD=1 detected. Enabling model CPU offload.")
                pipeline.enable_model_cpu_offload()
            else:
                pipeline = pipeline.to(self._device)
            pipeline.set_progress_bar_config(disable=True)
            self._pipeline = pipeline
            LOGGER.info("Pipeline ready")
            return pipeline

    def _encode_image(self, image: Image.Image, image_format: str) -> tuple[str, str]:
        if image.mode != "RGB":
            image = image.convert("RGB")

        mime_type = SUPPORTED_FORMATS[image_format]
        buffer = io.BytesIO()

        save_kwargs: Dict[str, Any] = {"format": image_format.upper()}
        if image_format == "jpeg":
            save_kwargs.update({"quality": 95, "optimize": True})

        image.save(buffer, **save_kwargs)
        encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return encoded, mime_type

    def _calculate_cost(self, width: int, height: int) -> float:
        megapixels = (width * height) / 1_000_000
        cost = round(megapixels * PRICE_PER_MEGAPIXEL, 8)
        return cost

    def _random_seed(self) -> int:
        return secrets.randbelow(2**31)


class _MockFluxPipeline:
    """Lightweight pipeline used for local tests without downloading weights."""

    def __call__(self, *_, width: int, height: int, **__) -> SimpleNamespace:
        image = Image.new("RGB", (width, height), color=(98, 56, 189))
        return SimpleNamespace(images=[image])
