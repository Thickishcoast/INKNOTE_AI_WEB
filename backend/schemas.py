from typing import Any, Literal

from pydantic import BaseModel, Field


class NoteCreate(BaseModel):
    title: str = Field(default="Untitled note", max_length=200)


class NoteUpdate(BaseModel):
    title: str = Field(max_length=200)
    blocks: list[dict[str, Any]]


class CanvasSubmitRequest(BaseModel):
    note_id: str = Field(min_length=1, max_length=100, pattern=r"^[A-Za-z0-9_-]+$")
    page_data_url: str = Field(min_length=20, max_length=15_000_000)
    strokes: list[dict[str, Any]] = Field(default_factory=list, max_length=20_000)
    typed_text: str = Field(default="", max_length=8_000)
    inference_steps: Literal[10, 25, 50] = 10
