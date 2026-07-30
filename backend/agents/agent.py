import asyncio
import logging
from collections.abc import Iterator

from backend.config import MAX_CONTEXT_MESSAGES
from backend.services.llm import ask_model
from backend.services.semantic_memory import save_new_memories, search_memories
from backend.storage.store_history import load_conversation, save_conversation


LOGGER = logging.getLogger(__name__)
MEMORY_IDLE_SECONDS = 30
MEMORY_QUEUE: asyncio.Queue[str] = asyncio.Queue()
MEMORY_WORKER: asyncio.Task[None] | None = None

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


def queue_new_memories(user_message: str) -> None:
    text = user_message.strip()
    if text:
        MEMORY_QUEUE.put_nowait(text)


async def memory_worker() -> None:
    """Process queued messages one batch at a time after a short idle period."""
    while True:
        batch = [await MEMORY_QUEUE.get()]
        try:
            await asyncio.sleep(MEMORY_IDLE_SECONDS)
            while not MEMORY_QUEUE.empty():
                batch.append(MEMORY_QUEUE.get_nowait())
            await save_new_memories("\n".join(batch))
        except asyncio.CancelledError:
            raise
        except Exception:
            LOGGER.exception("Background memory extraction failed.")
        finally:
            for _ in batch:
                MEMORY_QUEUE.task_done()


def start_memory_worker() -> None:
    global MEMORY_WORKER

    if MEMORY_WORKER is None or MEMORY_WORKER.done():
        MEMORY_WORKER = asyncio.create_task(memory_worker(), name="inknote-memory-worker")


async def stop_memory_worker() -> None:
    global MEMORY_WORKER

    if MEMORY_WORKER is not None:
        MEMORY_WORKER.cancel()
        try:
            await MEMORY_WORKER
        except asyncio.CancelledError:
            pass
        MEMORY_WORKER = None

    while not MEMORY_QUEUE.empty():
        MEMORY_QUEUE.get_nowait()
        MEMORY_QUEUE.task_done()


def get_reference_context(
    conversation_id: str,
    query: str = "",
) -> tuple[list[dict[str, str]], list[str]]:
    """Return recent model history and relevant durable memories for unified routing."""
    messages = load_conversation(conversation_id, SYSTEM_MESSAGE)
    history = [
        {"role": str(message["role"]), "content": str(message["content"])}
        for message in messages[1:][-MAX_CONTEXT_MESSAGES:]
    ]
    memories = search_memories(query.strip()) if query.strip() else []
    return history, memories


def process_user_message(
    conversation_id: str,
    user_input: str,
    *,
    system_context: str | None = None,
    record: bool = True,
    think: bool = True,
) -> Iterator[str]:
    text = user_input.strip()
    if not text:
        raise ValueError("The message is empty.")

    messages = load_conversation(conversation_id, SYSTEM_MESSAGE)
    memories = search_memories(text)
    messages.append({"role": "user", "content": text})

    request_messages = [messages[0], *messages[1:][-MAX_CONTEXT_MESSAGES:]]
    if memories:
        request_messages.insert(
            -1,
            {
                "role": "system",
                "content": "Relevant user memory:\n"
                + "\n".join(f"- {item}" for item in memories),
            },
        )
    if system_context:
        request_messages.insert(
            -1,
            {"role": "system", "content": system_context.strip()},
        )

    response_text = ""
    for chunk in ask_model(request_messages, think=think):
        response_text += chunk
        yield chunk

    if not response_text.strip():
        raise RuntimeError("The model returned an empty response.")

    if record:
        messages.append({"role": "assistant", "content": response_text})
        save_conversation(conversation_id, messages)
        queue_new_memories(text)


def record_exchange(
    conversation_id: str,
    user_input: str,
    assistant_output: str,
    *,
    remember: bool = False,
) -> None:
    messages = load_conversation(conversation_id, SYSTEM_MESSAGE)
    messages.extend(
        [
            {"role": "user", "content": user_input.strip()},
            {"role": "assistant", "content": assistant_output.strip()},
        ]
    )
    save_conversation(conversation_id, messages)
    if remember:
        queue_new_memories(user_input)
