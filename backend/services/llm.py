import json
from collections.abc import Iterator
from typing import Any

import ollama

from backend.config import MODEL_CONTEXT_SIZE, QWEN_KEEP_ALIVE, QWEN_MODEL


async def ask_structured_model_async(
    *,
    messages: list[dict[str, Any]],
    schema: dict[str, Any],
    options: dict[str, Any] | None = None,
    think: bool = True,
) -> dict[str, Any]:
    """Async version used by the background memory worker."""
    required_fields = set(schema.get("required", []))
    client = ollama.AsyncClient()

    for attempt_think in (think, False):
        response = await client.chat(
            model=QWEN_MODEL,
            messages=messages,
            format=schema,
            stream=False,
            think=attempt_think,
            keep_alive=QWEN_KEEP_ALIVE,
            options=options or {},
        )
        content = (response.message.content or "").strip()
        if not content:
            continue

        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            continue

        fields = set(getattr(data, "keys", lambda: ())())
        if required_fields.issubset(fields):
            return data

    raise RuntimeError(
        "Qwen did not return valid structured JSON for background memory extraction."
    )


def ask_model(
    messages: list[dict[str, str]],
    *,
    think: bool = True,
) -> Iterator[str]:
    response = ollama.chat(
        model=QWEN_MODEL,
        messages=messages,
        stream=True,
        think=think,
        keep_alive=QWEN_KEEP_ALIVE,
        options={"num_ctx": MODEL_CONTEXT_SIZE},
    )

    for chunk in response:
        content = chunk.message.content
        if content:
            yield content


def unload_qwen_model() -> None:
    """Free Qwen before FLUX claims GPU memory."""
    ollama.generate(model=QWEN_MODEL, keep_alive=0)
