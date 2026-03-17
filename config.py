"""
config.py
---------
Central configuration for the AI Research Paper Explainer.
All tunable parameters and environment settings live here so
every other module stays free of hard-coded magic values.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# ── Load .env file (if present) ──────────────────────────────────────────────
load_dotenv()

# ── Project root ─────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent

# ── Data paths ───────────────────────────────────────────────────────────────
DATA_DIR            = BASE_DIR / "data"
UPLOADED_PAPERS_DIR = DATA_DIR / "uploaded_papers"
VECTOR_STORE_DIR    = DATA_DIR / "vector_store"

# Create directories if they don't exist
UPLOADED_PAPERS_DIR.mkdir(parents=True, exist_ok=True)
VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)

# ── LLM settings ─────────────────────────────────────────────────────────────
MISTRAL_API_KEY  = os.getenv("MISTRAL_API_KEY", "")
MISTRAL_MODEL    = os.getenv("MISTRAL_MODEL", "mistral-large-latest")   # or "open-mistral-7b"
LLM_TEMPERATURE  = float(os.getenv("LLM_TEMPERATURE", "0.3"))
LLM_MAX_TOKENS   = int(os.getenv("LLM_MAX_TOKENS", "4096"))

# ── Embedding settings ────────────────────────────────────────────────────────
EMBEDDING_MODEL  = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2"   # fast & accurate; swap for a larger model any time
)

# ── Text chunking ─────────────────────────────────────────────────────────────
CHUNK_SIZE       = int(os.getenv("CHUNK_SIZE", "800"))    # characters (≈ 150-200 tokens)
CHUNK_OVERLAP    = int(os.getenv("CHUNK_OVERLAP", "150")) # overlap keeps context across chunks

# ── Retrieval settings ────────────────────────────────────────────────────────
TOP_K_RESULTS    = int(os.getenv("TOP_K_RESULTS", "5"))   # chunks returned per query
VECTOR_DB_TYPE   = os.getenv("VECTOR_DB_TYPE", "faiss")   # "faiss" | "chroma"

# ── Streamlit UI ──────────────────────────────────────────────────────────────
APP_TITLE        = "📄 AI Research Paper Explainer"
APP_ICON         = "📄"
MAX_UPLOAD_SIZE_MB = 50

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_LEVEL        = os.getenv("LOG_LEVEL", "INFO")
