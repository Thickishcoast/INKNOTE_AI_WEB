from __future__ import annotations

import threading
from collections import defaultdict
from collections.abc import Iterator

from backend.config import MAX_CONTEXT_MESSAGES
from backend.services.llm import ask_model
from backend.services.semantic_memory import add_memory, extract_memories, search_memories
from backend.storage.store_history import load_conversation, save_conversation


SYSTEM_MESSAGE = {
    "role": "system",
    "content": (
        "Your name is InkNote. You are a friendly AI tutor and creative notebook "
        "companion. The user may submit handwritten text, sketches, diagrams, or a "
        "mixture of them. Treat any supplied visual description as part of the user's "
        "message. InkNote can use a FLUX image-generation tool, so never claim that "
        "InkNote cannot generate images when tool context says FLUX is handling the "
        "request. Keep answers concise and natural unless the user asks for detail."
    ),
}

_LOCKS: defaultdict[str, threading.Lock] = defaultdict(threading.Lock)


def memory_message(memories: list[str]) -> dict[str, str]:
    return {
        "role": "system",
        "content": "Relevant user memory:\n" + "\n".join(f"- {item}" for item in memories),
    }


def save_new_memories(user_message: str) -> None:
    for memory in extract_memories(user_message):
        add_memory(memory)


def process_user_message(
    conversation_id: str,
    user_input: str,
    *,
    system_context: str | None = None,
    record: bool = True,
) -> Iterator[str]:
    text = user_input.strip()
    if not text:
        raise ValueError("The message is empty.")

    with _LOCKS[conversation_id]:
        messages = load_conversation(conversation_id, SYSTEM_MESSAGE)
        memories = search_memories(text)
        messages.append({"role": "user", "content": text})

        request_messages = [messages[0], *messages[1:][-MAX_CONTEXT_MESSAGES:]]
        if memories:
            request_messages.insert(-1, memory_message(memories))
        if system_context:
            request_messages.insert(
                -1,
                {"role": "system", "content": system_context.strip()},
            )

        response_text = ""
        for chunk in ask_model(request_messages):
            response_text += chunk
            yield chunk

        if not response_text.strip():
            raise RuntimeError("The model returned an empty response.")

        if record:
            messages.append({"role": "assistant", "content": response_text})
            save_conversation(conversation_id, messages)
            threading.Thread(target=save_new_memories, args=(text,), daemon=True).start()


def record_exchange(conversation_id: str, user_input: str, assistant_output: str) -> None:
    with _LOCKS[conversation_id]:
        messages = load_conversation(conversation_id, SYSTEM_MESSAGE)
        messages.extend(
            [
                {"role": "user", "content": user_input.strip()},
                {"role": "assistant", "content": assistant_output.strip()},
            ]
        )
        save_conversation(conversation_id, messages)
