import json
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Any

from backend.config import DATABASE_FILE


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def connect() -> sqlite3.Connection:
    DATABASE_FILE.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DATABASE_FILE)
    connection.row_factory = sqlite3.Row
    return connection


def init_database() -> None:
    with connect() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS notes (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                blocks_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )


def row_to_note(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": row["id"],
        "title": row["title"],
        "blocks": json.loads(row["blocks_json"]),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def create_note(title: str = "Untitled note") -> dict[str, Any]:
    timestamp = now()
    note = {
        "id": uuid.uuid4().hex,
        "title": title.strip() or "Untitled note",
        "blocks": [],
        "created_at": timestamp,
        "updated_at": timestamp,
    }
    with connect() as connection:
        connection.execute(
            "INSERT INTO notes VALUES (?, ?, ?, ?, ?)",
            (
                note["id"],
                note["title"],
                json.dumps(note["blocks"]),
                timestamp,
                timestamp,
            ),
        )
    return note


def list_notes() -> list[dict[str, Any]]:
    with connect() as connection:
        rows = connection.execute("SELECT * FROM notes ORDER BY updated_at DESC").fetchall()
    return [row_to_note(row) for row in rows]


def get_note(note_id: str) -> dict[str, Any] | None:
    with connect() as connection:
        row = connection.execute("SELECT * FROM notes WHERE id = ?", (note_id,)).fetchone()
    return row_to_note(row) if row else None


def save_note(
    note_id: str,
    title: str,
    blocks: list[dict[str, Any]],
) -> dict[str, Any] | None:
    with connect() as connection:
        cursor = connection.execute(
            "UPDATE notes SET title = ?, blocks_json = ?, updated_at = ? WHERE id = ?",
            (
                title.strip() or "Untitled note",
                json.dumps(blocks, ensure_ascii=False),
                now(),
                note_id,
            ),
        )
    return get_note(note_id) if cursor.rowcount else None


def delete_note(note_id: str) -> bool:
    with connect() as connection:
        cursor = connection.execute("DELETE FROM notes WHERE id = ?", (note_id,))
    return cursor.rowcount > 0
