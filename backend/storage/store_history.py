import json
import re

from backend.storage.chroma_store import conversation_collection, save_history_record


_SAFE_ID = re.compile(r"^[A-Za-z0-9_-]{1,100}$")


def validate_conversation_id(conversation_id: str) -> None:
    if not _SAFE_ID.fullmatch(conversation_id):
        raise ValueError("Invalid conversation ID.")


def load_conversation(
    conversation_id: str,
    system_message: dict[str, str],
) -> list[dict[str, str]]:
    validate_conversation_id(conversation_id)
    result = conversation_collection().get(
        ids=[conversation_id],
        include=["documents"],
    )
    if not result["documents"]:
        return [system_message.copy()]

    messages = json.loads(result["documents"][0])
    if not messages or messages[0].get("role") != "system":
        messages.insert(0, system_message.copy())
    return messages


def save_conversation(conversation_id: str, messages: list[dict[str, str]]) -> None:
    validate_conversation_id(conversation_id)
    save_history_record(conversation_id, messages)


def delete_conversation(conversation_id: str) -> None:
    validate_conversation_id(conversation_id)
    conversation_collection().delete(ids=[conversation_id])
