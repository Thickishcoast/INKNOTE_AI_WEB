from __future__ import annotations

import base64
import gc
import io
import json
import re
import threading
import uuid
from typing import Any

import ollama
import torch
from diffusers import Flux2KleinPipeline
from PIL import Image

from backend.config import (
    GENERATED_DIR,
    IMAGE_CPU_OFFLOAD,
    IMAGE_DEVICE,
    IMAGE_DTYPE,
    IMAGE_HF_TOKEN,
    IMAGE_MODEL,
    IMAGE_PROVIDER,
    IMAGE_RELEASE_AFTER_GENERATION,
    MULTIMODAL_MODEL,
    ROUTER_CONFIDENCE_THRESHOLD,
)


_DATA_URL = re.compile(r"^data:image/[^;]+;base64,(.+)$", re.DOTALL)
_PIPELINE: Flux2KleinPipeline | None = None
_PIPELINE_LOCK = threading.Lock()
_GENERATION_LOCK = threading.Lock()


def decode_image_data_url(data_url: str) -> bytes:
    match = _DATA_URL.fullmatch(data_url)
    if match is None:
        raise ValueError("Expected a base64 image data URL.")
    return base64.b64decode(match.group(1), validate=True)


def image_generation_enabled() -> bool:
    return IMAGE_PROVIDER == "flux2_local" and bool(IMAGE_MODEL.strip())


def image_dtype() -> torch.dtype:
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }.get(IMAGE_DTYPE, torch.bfloat16)


def load_flux_pipeline() -> Flux2KleinPipeline:
    global _PIPELINE

    if _PIPELINE is not None:
        return _PIPELINE

    with _PIPELINE_LOCK:
        if _PIPELINE is not None:
            return _PIPELINE

        kwargs: dict[str, object] = {
            "dtype": image_dtype(),
            "low_cpu_mem_usage": True,
            "local_files_only": True,
        }
        if IMAGE_HF_TOKEN:
            kwargs["token"] = IMAGE_HF_TOKEN

        pipeline = Flux2KleinPipeline.from_pretrained(IMAGE_MODEL, **kwargs)
        if IMAGE_CPU_OFFLOAD:
            if not torch.cuda.is_available():
                raise RuntimeError("CPU offload requires a CUDA GPU.")
            pipeline.enable_model_cpu_offload()
        else:
            pipeline.to(IMAGE_DEVICE)

        _PIPELINE = pipeline
        return pipeline


def release_flux_pipeline() -> None:
    global _PIPELINE

    with _PIPELINE_LOCK:
        _PIPELINE = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def parse_size(size: str) -> tuple[int, int]:
    width_text, height_text = size.lower().split("x", 1)
    width = int(width_text)
    height = int(height_text)
    if not 256 <= width <= 2048 or not 256 <= height <= 2048:
        raise ValueError("Image size must stay between 256 and 2048 pixels.")
    return width, height


def prepare_page(page_data_url: str) -> tuple[bytes, Image.Image]:
    image = Image.open(io.BytesIO(decode_image_data_url(page_data_url))).convert("RGB")
    image.thumbnail((1536, 1536))
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue(), image


def clean_string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def clean_flux_prompt(
    image_prompt: str,
    recognized_text: list[str],
) -> str:
    prompt = image_prompt.strip()
    for writing in recognized_text:
        prompt = re.sub(re.escape(writing), "", prompt, flags=re.IGNORECASE)
    prompt = re.sub(r"\s+", " ", prompt).strip(" \t\r\n,;:-\"'")
    if not prompt:
        raise ValueError("The visual image prompt is empty after removing canvas writing.")
    return prompt


def analyze_canvas_page(page_data_url: str, stroke_count: int = 0) -> dict[str, object]:
    page_bytes, _ = prepare_page(page_data_url)
    schema = {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["chat", "image", "both", "clarify"]},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "recognized_text": {"type": "array", "items": {"type": "string"}},
            "display_text": {"type": "string"},
            "conversation_message": {"type": "string"},
            "image_prompt": {"type": "string"},
            "clarification_question": {"type": "string"},
        },
        "required": [
            "action",
            "confidence",
            "recognized_text",
            "display_text",
            "conversation_message",
            "image_prompt",
            "clarification_question",
        ],
        "additionalProperties": False,
    }
    prompt = f"""
You are the routing agent for a single shared notebook canvas. The page may contain
handwritten text, sketches, arrows, diagrams, labels, or any mixture of them. There
are {stroke_count} recorded drawing strokes.

Choose exactly one action:
- chat: answer, explain, analyze, summarize, calculate, or discuss. A drawing by itself
  does not imply image generation.
- image: the user explicitly asks to create, render, redesign, transform, colorize, or
  produce a polished visual.
- both: the user explicitly requests both a written response and a generated visual.
- clarify: the intended action cannot be determined reliably.

Reliability rules:
- Route questions about a diagram to chat unless image creation is explicitly requested.
- Route phrases such as "make this realistic", "turn this into", "render this", or
  "create a polished version" to image.
- Use both only when both explanation and image creation are requested.
- Do not infer image generation merely because rough marks are present.

Extract the readable writing. Describe the visual content and spatial relationships.
For display_text, provide a short natural summary suitable for conversation history.
For conversation_message, combine the user's words and relevant visual context so a
text assistant can answer accurately. For image or both, image_prompt must contain only
a positive visual description of the desired scene: subject, composition, pose,
proportions, viewpoint, lighting, color, medium, and relationships. Do not put the
user's instruction, exclusions, negative wording, handwriting, typography terms, or
quoted prompt text in image_prompt. Generated notebook images are intentionally
text-free. For clarify, provide one brief clarification_question. Return JSON only.
""".strip()

    response = ollama.chat(
        model=MULTIMODAL_MODEL,
        messages=[{"role": "user", "content": prompt, "images": [page_bytes]}],
        format=schema,
        stream=False,
        think=False,
        keep_alive=0,
        options={"temperature": 0.1, "top_p": 0.2, "num_ctx": 4096},
    )

    data = json.loads(response.message.content)
    action = str(data["action"]).strip().lower()
    confidence = max(0.0, min(1.0, float(data["confidence"])))
    image_prompt = str(data["image_prompt"]).strip()
    clarification_question = str(data["clarification_question"]).strip()

    if confidence < ROUTER_CONFIDENCE_THRESHOLD:
        action = "clarify"
    if action in {"image", "both"} and not image_prompt:
        action = "clarify"
    if action == "clarify" and not clarification_question:
        clarification_question = "Should I answer your canvas as a question, generate an image, or do both?"

    return {
        "action": action,
        "confidence": confidence,
        "recognized_text": clean_string_list(data["recognized_text"]),
        "display_text": str(data["display_text"]).strip() or "Canvas submission",
        "conversation_message": str(data["conversation_message"]).strip(),
        "image_prompt": image_prompt,
        "clarification_question": clarification_question,
    }


def generate_canvas_image(
    image_prompt: str,
    size: str = "1024x1024",
    *,
    recognized_text: list[str] | None = None,
    num_inference_steps: int = 10,
) -> str:
    if not image_generation_enabled():
        raise RuntimeError("FLUX image generation is disabled.")
    if num_inference_steps not in {10, 25, 50}:
        raise ValueError("Inference steps must be one of: 10, 25, or 50.")

    width, height = parse_size(size)
    pipeline = load_flux_pipeline()
    result = None
    prompt = clean_flux_prompt(
        image_prompt,
        recognized_text or [],
    )
    pipeline_inputs: dict[str, Any] = {
        "prompt": prompt,
        "width": width,
        "height": height,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": 1.0,
    }

    try:
        with _GENERATION_LOCK:
            result = pipeline(**pipeline_inputs)
            image = result.images[0]
    finally:
        if IMAGE_RELEASE_AFTER_GENERATION:
            result = None
            pipeline = None
            release_flux_pipeline()

    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"{uuid.uuid4().hex}.png"
    image.save(GENERATED_DIR / filename, format="PNG")
    return f"/generated/{filename}"
