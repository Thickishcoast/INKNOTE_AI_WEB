from __future__ import annotations

import json
import math
import threading

import ollama

from backend.config import CHAT_MODEL, EMBEDDING_MODEL, MEMORY_FILE


_MEMORY_LOCK = threading.Lock()


def load_memories() -> list[dict]:
    if not MEMORY_FILE.exists():
        return []
    data = json.loads(MEMORY_FILE.read_text(encoding="utf-8"))
    return data if isinstance(data, list) else []


def save_memories(memories: list[dict]) -> None:
    MEMORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = MEMORY_FILE.with_suffix(".tmp")
    temporary_path.write_text(
        json.dumps(memories, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary_path.replace(MEMORY_FILE)


def create_embedding(text: str) -> list[float]:
    return ollama.embed(
        model=EMBEDDING_MODEL,
        input=text,
        keep_alive=0,
    )["embeddings"][0]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    magnitude_a = math.sqrt(sum(x * x for x in a))
    magnitude_b = math.sqrt(sum(y * y for y in b))
    return dot / (magnitude_a * magnitude_b) if magnitude_a and magnitude_b else 0.0


def add_memory(text: str) -> None:
    text = text.strip()
    if not text:
        return

    with _MEMORY_LOCK:
        memories = [
            item
            for item in load_memories()
            if item.get("embedding_model") == EMBEDDING_MODEL
        ]
        if any(item.get("text", "").lower() == text.lower() for item in memories):
            return
        memories.append(
            {
                "text": text,
                "embedding": create_embedding(text),
                "embedding_model": EMBEDDING_MODEL,
            }
        )
        save_memories(memories)


def search_memories(query: str, top_k: int = 3, minimum_score: float = 0.35) -> list[str]:
    with _MEMORY_LOCK:
        memories = load_memories()

    if not memories:
        return []

    query_embedding = create_embedding(query)
    scored = [
        (cosine_similarity(query_embedding, item["embedding"]), item["text"])
        for item in memories
        if item.get("embedding_model") == EMBEDDING_MODEL
    ]
    scored.sort(reverse=True)
    return [text for score, text in scored[:top_k] if score >= minimum_score]


def extract_memories(user_message: str) -> list[str]:
    schema = {
        "type": "object",
        "properties": {
            "memories": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["memories"],
    }
    prompt = (
        "Extract durable user preferences, skills, goals, projects, or background. "
        "Ignore normal questions and temporary facts. Return each memory as a complete "
        f"third-person sentence. User message: {user_message}"
    )
    response = ollama.chat(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        format=schema,
        stream=False,
        think=False,
        keep_alive=0,
        options={"temperature": 0},
    )
    return [
        item.strip()
        for item in json.loads(response.message.content)["memories"]
        if item.strip()
    ]
