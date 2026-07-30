import json
import math

import ollama

from backend.config import EMBEDDING_MODEL, MEMORY_FILE, ROUTER_CONTEXT_SIZE
from backend.services.llm import ask_structured_model_async


def load_memories() -> list[dict]:
    if not MEMORY_FILE.exists():
        return []
    return json.loads(MEMORY_FILE.read_text(encoding="utf-8"))


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


def search_memories(query: str, top_k: int = 3, minimum_score: float = 0.35) -> list[str]:
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


async def extract_memories(user_message: str) -> list[str]:
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
    data = await ask_structured_model_async(
        messages=[{"role": "user", "content": prompt}],
        schema=schema,
        options={"temperature": 0, "num_ctx": ROUTER_CONTEXT_SIZE},
        think=False,
    )
    return [
        item.strip()
        for item in data["memories"]
        if item.strip()
    ]


async def save_new_memories(user_message: str) -> None:
    """Extract, embed, and save durable memories without blocking the request."""
    candidates = await extract_memories(user_message)
    memories = [
        item
        for item in load_memories()
        if item.get("embedding_model") == EMBEDDING_MODEL
    ]
    known = {item.get("text", "").lower() for item in memories}
    new_texts = []
    for text in candidates:
        key = text.lower()
        if key not in known:
            known.add(key)
            new_texts.append(text)
    if not new_texts:
        return

    response = await ollama.AsyncClient().embed(
        model=EMBEDDING_MODEL,
        input=new_texts,
        keep_alive=0,
    )
    for text, embedding in zip(new_texts, response["embeddings"]):
        memories.append(
            {
                "text": text,
                "embedding": embedding,
                "embedding_model": EMBEDDING_MODEL,
            }
        )
    save_memories(memories)
