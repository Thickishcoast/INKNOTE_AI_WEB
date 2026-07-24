"""Storage-only smoke test; it does not call Ollama."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.storage.note_store import create_note, delete_note, get_note, init_database, save_note


def main() -> None:
    init_database()
    note = create_note("Smoke test")
    note = save_note(
        note["id"],
        "Updated smoke test",
        [{"id": "one", "type": "user_text", "text": "hello"}],
    )
    assert note is not None
    loaded = get_note(note["id"])
    assert loaded and loaded["blocks"][0]["text"] == "hello"
    assert delete_note(note["id"])
    print("Storage smoke test passed.")


if __name__ == "__main__":
    main()
