from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

DATA_DIR = PROJECT_ROOT / "data"
CONVERSATION_DIR = DATA_DIR / "conversations"
MEMORY_FILE = DATA_DIR / "memory" / "knowledge.json"
DATABASE_FILE = DATA_DIR / "inknote.db"
GENERATED_DIR = DATA_DIR / "generated"
STATIC_DIR = PROJECT_ROOT / "frontend"

CHAT_MODEL = os.getenv("INKNOTE_CHAT_MODEL", "qwen2.5vl:3b")
MULTIMODAL_MODEL = os.getenv("INKNOTE_MULTIMODAL_MODEL", CHAT_MODEL)
EMBEDDING_MODEL = os.getenv("INKNOTE_EMBEDDING_MODEL", "qwen3-embedding:0.6b")
MAX_CONTEXT_MESSAGES = int(os.getenv("INKNOTE_MAX_CONTEXT_MESSAGES", "24"))
ROUTER_CONFIDENCE_THRESHOLD = float(os.getenv("INKNOTE_ROUTER_CONFIDENCE_THRESHOLD", "0.62"))

IMAGE_PROVIDER = os.getenv("INKNOTE_IMAGE_PROVIDER", "flux2_local")
IMAGE_MODEL = os.getenv("INKNOTE_IMAGE_MODEL", "black-forest-labs/FLUX.2-klein-4B")
IMAGE_DEVICE = os.getenv("INKNOTE_IMAGE_DEVICE", "cuda")
IMAGE_DTYPE = os.getenv("INKNOTE_IMAGE_DTYPE", "bfloat16").lower()
IMAGE_CPU_OFFLOAD = os.getenv("INKNOTE_IMAGE_CPU_OFFLOAD", "true").lower() == "true"
IMAGE_RELEASE_AFTER_GENERATION = (
    os.getenv("INKNOTE_IMAGE_RELEASE_AFTER_GENERATION", "true").lower() == "true"
)
IMAGE_HF_TOKEN = os.getenv("HUGGINGFACE_HUB_TOKEN", os.getenv("HF_TOKEN", ""))
