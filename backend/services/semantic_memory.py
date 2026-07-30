import ollama

from backend.config import EMBEDDING_MODEL, ROUTER_CONTEXT_SIZE
from backend.services.llm import ask_structured_model_async
from backend.storage.chroma_store import memory_collection, memory_id


def create_embedding(text: str) -> list[float]:
    return ollama.embed(
        model=EMBEDDING_MODEL,
        input=text,
        keep_alive=0,
    )["embeddings"][0]


def search_memories(query: str, top_k: int = 3, minimum_score: float = 0.35) -> list[str]:
    collection = memory_collection()
    memory_count = collection.count()
    if memory_count == 0:
        return []

    query_embedding = create_embedding(query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, memory_count),
        include=["documents", "distances"],
    )
    return [
        text
        for text, distance in zip(results["documents"][0], results["distances"][0])
        if distance <= 1 - minimum_score
    ]


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
    candidates_by_id = {memory_id(text): text for text in candidates}
    if not candidates_by_id:
        return

    collection = memory_collection()
    candidate_ids = list(candidates_by_id)
    known_ids = set(collection.get(ids=candidate_ids)["ids"])
    new_ids = [record_id for record_id in candidate_ids if record_id not in known_ids]
    new_texts = [candidates_by_id[record_id] for record_id in new_ids]
    if not new_texts:
        return

    response = await ollama.AsyncClient().embed(
        model=EMBEDDING_MODEL,
        input=new_texts,
        keep_alive=0,
    )
    collection.upsert(
        ids=new_ids,
        embeddings=response["embeddings"],
        documents=new_texts,
        metadatas=[{"embedding_model": EMBEDDING_MODEL} for _ in new_texts],
    )
