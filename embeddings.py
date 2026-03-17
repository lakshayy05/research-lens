"""
embeddings.py
-------------
Wraps the HuggingFace sentence-transformers embedding model inside a
LangChain-compatible interface.

Why sentence-transformers?
  - Runs fully locally (no API key required)
  - The all-MiniLM-L6-v2 model is fast and accurate for semantic search
  - Swappable for any other model by changing EMBEDDING_MODEL in config.py

Architecture note
  - This module is intentionally thin: it just creates and caches the
    embeddings object.  The vector_store module is responsible for using it.
"""

import logging
from functools import lru_cache
from typing import List

from langchain_huggingface import HuggingFaceEmbeddings

from config import EMBEDDING_MODEL

logger = logging.getLogger(__name__)


# ── Embedding model factory ───────────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_embedding_model(model_name: str = EMBEDDING_MODEL) -> HuggingFaceEmbeddings:
    """
    Return a cached HuggingFaceEmbeddings instance.

    The model is downloaded on first call and cached in memory
    for the duration of the process (via lru_cache).

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier, e.g.
        "sentence-transformers/all-MiniLM-L6-v2"

    Returns
    -------
    HuggingFaceEmbeddings
    """
    logger.info("Loading embedding model: %s", model_name)
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": _get_device()},
        encode_kwargs={"normalize_embeddings": True},  # cosine-sim friendly
    )
    logger.info("Embedding model loaded ✓")
    return embeddings


def embed_texts(texts: List[str], model_name: str = EMBEDDING_MODEL) -> List[List[float]]:
    """
    Convenience function: embed a list of plain strings.

    Parameters
    ----------
    texts      : List[str]
    model_name : str

    Returns
    -------
    List[List[float]]  One vector per input text.
    """
    model = get_embedding_model(model_name)
    return model.embed_documents(texts)


def embed_query(query: str, model_name: str = EMBEDDING_MODEL) -> List[float]:
    """
    Embed a single query string.

    Parameters
    ----------
    query      : str
    model_name : str

    Returns
    -------
    List[float]
    """
    model = get_embedding_model(model_name)
    return model.embed_query(query)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _get_device() -> str:
    """
    Return 'cuda' if a GPU is available, otherwise 'cpu'.
    Keeps embedding generation fast when a GPU is present.
    """
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"
