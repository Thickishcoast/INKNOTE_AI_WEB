import hashlib
import json
from datetime import UTC, datetime
from functools import lru_cache

import chromadb

from backend.config import CHROMA_DIR, EMBEDDING_MODEL


CONVERSATION_COLLECTION = "inknote_conversations"


@lru_cache
def chroma_client() -> chromadb.PersistentClient:
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(CHROMA_DIR))


@lru_cache
def conversation_collection():
    return chroma_client().get_or_create_collection(
        name=CONVERSATION_COLLECTION,
        embedding_function=None,
    )


def memory_collection_name() -> str:
    model_key = hashlib.sha256(EMBEDDING_MODEL.encode("utf-8")).hexdigest()[:12]
    return f"inknote_memories_{model_key}"


@lru_cache
def memory_collection():
    return chroma_client().get_or_create_collection(
        name=memory_collection_name(),
        embedding_function=None,
        metadata={"embedding_model": EMBEDDING_MODEL},
        configuration={"hnsw": {"space": "cosine"}},
    )


def memory_id(text: str) -> str:
    return hashlib.sha256(text.strip().casefold().encode("utf-8")).hexdigest()


def save_history_record(conversation_id: str, messages: list[dict[str, str]]) -> None:
    conversation_collection().upsert(
        ids=[conversation_id],
        embeddings=[[0.0]],
        documents=[json.dumps(messages, ensure_ascii=False)],
        metadatas=[
            {
                "message_count": len(messages),
                "updated_at": datetime.now(UTC).isoformat(),
            }
        ],
    )


def init_chroma_store() -> None:
    conversation_collection()
    memory_collection()
