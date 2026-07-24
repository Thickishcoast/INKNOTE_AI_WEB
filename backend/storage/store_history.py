from __future__ import annotations

import json
import re
from pathlib import Path

from backend.config import CONVERSATION_DIR


_SAFE_ID = re.compile(r"^[A-Za-z0-9_-]{1,100}$")


def conversation_path(conversation_id: str) -> Path:
    if not _SAFE_ID.fullmatch(conversation_id):
        raise ValueError("Invalid conversation ID.")
    return CONVERSATION_DIR / f"{conversation_id}.json"


def load_conversation(
    conversation_id: str,
    system_message: dict[str, str],
) -> list[dict[str, str]]:
    path = conversation_path(conversation_id)
    if not path.exists():
        return [system_message.copy()]

    messages = json.loads(path.read_text(encoding="utf-8"))
    if not messages or messages[0].get("role") != "system":
        messages.insert(0, system_message.copy())
    return messages


def save_conversation(conversation_id: str, messages: list[dict[str, str]]) -> None:
    path = conversation_path(conversation_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(".tmp")
    temporary_path.write_text(
        json.dumps(messages, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary_path.replace(path)


def delete_conversation(conversation_id: str) -> None:
    path = conversation_path(conversation_id)
    if path.exists():
        path.unlink()
