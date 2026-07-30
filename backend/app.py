# Load PyTorch's native Windows DLLs before FastAPI initializes the server.
import torch  # noqa: F401

from contextlib import asynccontextmanager
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from backend.agents.agent import start_memory_worker, stop_memory_worker
from backend.agents.canvas_agent import process_canvas_submission
from backend.config import (
    GENERATED_DIR,
    IMAGE_MODEL,
    MODEL_CONTEXT_SIZE,
    QWEN_KEEP_ALIVE,
    QWEN_MODEL,
    ROUTER_CONTEXT_SIZE,
    STATIC_DIR,
)
from backend.services.image_generation import image_generation_enabled
from backend.storage.chroma_store import init_chroma_store
from backend.storage.note_store import (
    create_note,
    delete_note,
    get_note,
    init_database,
    list_notes,
    save_note,
)
from backend.storage.store_history import delete_conversation


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_database()
    init_chroma_store()
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    start_memory_worker()
    try:
        yield
    finally:
        await stop_memory_worker()


app = FastAPI(title="InkNote AI", version="1.5.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/generated", StaticFiles(directory=GENERATED_DIR), name="generated")


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


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/manifest.webmanifest")
def manifest() -> FileResponse:
    return FileResponse(STATIC_DIR / "manifest.webmanifest")


@app.get("/service-worker.js")
def service_worker() -> FileResponse:
    return FileResponse(STATIC_DIR / "service-worker.js", media_type="application/javascript")


@app.get("/api/health")
def health() -> dict[str, Any]:
    images_enabled = image_generation_enabled()
    return {
        "status": "online",
        "qwen_model": QWEN_MODEL,
        "model_context_size": MODEL_CONTEXT_SIZE,
        "router_context_size": ROUTER_CONTEXT_SIZE,
        "qwen_keep_alive": QWEN_KEEP_ALIVE,
        "image_generation_enabled": images_enabled,
        "image_model": IMAGE_MODEL if images_enabled else None,
    }


@app.get("/api/notes")
def notes_list() -> list[dict[str, Any]]:
    return list_notes()


@app.post("/api/notes")
def notes_create(request: NoteCreate) -> dict[str, Any]:
    return create_note(request.title)


@app.get("/api/notes/{note_id}")
def notes_get(note_id: str) -> dict[str, Any]:
    note = get_note(note_id)
    if note is None:
        raise HTTPException(status_code=404, detail="Note not found.")
    return note


@app.put("/api/notes/{note_id}")
def notes_update(note_id: str, request: NoteUpdate) -> dict[str, Any]:
    note = save_note(note_id, request.title, request.blocks)
    if note is None:
        raise HTTPException(status_code=404, detail="Note not found.")
    return note


@app.delete("/api/notes/{note_id}")
def notes_delete(note_id: str) -> dict[str, bool]:
    if not delete_note(note_id):
        raise HTTPException(status_code=404, detail="Note not found.")
    delete_conversation(note_id)
    return {"deleted": True}


@app.post("/api/canvas/submit")
async def canvas_submit(request: CanvasSubmitRequest) -> dict[str, Any]:
    return await process_canvas_submission(
        request.note_id,
        request.page_data_url,
        request.strokes,
        typed_text=request.typed_text,
        inference_steps=request.inference_steps,
    )
