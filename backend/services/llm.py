from __future__ import annotations

from collections.abc import Iterator

import ollama

from backend.config import CHAT_MODEL


def ask_model(messages: list[dict[str, str]]) -> Iterator[str]:
    response = ollama.chat(
        model=CHAT_MODEL,
        messages=messages,
        stream=True,
        think=False,
        keep_alive=0,
        options={"num_ctx": 4096},
    )

    for chunk in response:
        content = chunk.message.content
        if content:
            yield content
