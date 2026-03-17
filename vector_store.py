"""
vector_store.py
---------------
Manages creation, persistence, and querying of the vector database.

Supports two backends controlled by config.VECTOR_DB_TYPE:
  - "faiss"  : Facebook AI Similarity Search — fast, file-based, no server needed
  - "chroma" : Chroma DB           — persistent, richer metadata filtering

The public API is intentionally identical for both backends so the rest
of the codebase never needs to know which one is in use.
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

from langchain_core.documents import Document

from config import VECTOR_DB_TYPE, VECTOR_STORE_DIR, TOP_K_RESULTS
from embeddings import get_embedding_model

logger = logging.getLogger(__name__)


# ── FAISS backend ─────────────────────────────────────────────────────────────

def _build_faiss(chunks: List[Document], embeddings) -> "FAISS":
    from langchain_community.vectorstores import FAISS
    logger.info("Building FAISS index from %d chunks …", len(chunks))
    db = FAISS.from_documents(chunks, embeddings)
    logger.info("FAISS index built ✓")
    return db


def _save_faiss(db, store_path: Path) -> None:
    store_path.mkdir(parents=True, exist_ok=True)
    db.save_local(str(store_path))
    logger.info("FAISS index saved → %s", store_path)


def _load_faiss(store_path: Path, embeddings) -> Optional["FAISS"]:
    from langchain_community.vectorstores import FAISS
    index_file = store_path / "index.faiss"
    if not index_file.exists():
        return None
    logger.info("Loading FAISS index from %s …", store_path)
    db = FAISS.load_local(
        str(store_path),
        embeddings,
        allow_dangerous_deserialization=True,   # safe: we wrote this file ourselves
    )
    return db


# ── Chroma backend ────────────────────────────────────────────────────────────

def _build_chroma(chunks: List[Document], embeddings, collection_name: str) -> "Chroma":
    from langchain_community.vectorstores import Chroma
    logger.info("Building Chroma collection '%s' from %d chunks …", collection_name, len(chunks))
    db = Chroma.from_documents(
        chunks,
        embeddings,
        collection_name=collection_name,
        persist_directory=str(VECTOR_STORE_DIR / collection_name),
    )
    logger.info("Chroma collection built ✓")
    return db


def _load_chroma(collection_name: str, embeddings) -> Optional["Chroma"]:
    from langchain_community.vectorstores import Chroma
    persist_dir = VECTOR_STORE_DIR / collection_name
    if not persist_dir.exists():
        return None
    logger.info("Loading Chroma collection '%s' …", collection_name)
    return Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=str(persist_dir),
    )


# ── Public API ────────────────────────────────────────────────────────────────

def create_vector_store(
    chunks: List[Document],
    paper_name: str,
    db_type: str = VECTOR_DB_TYPE,
):
    """
    Build a new vector store from a list of document chunks.

    Parameters
    ----------
    chunks     : List[Document]  Output of text_chunking.chunk_pages
    paper_name : str             Used as the collection/index name
    db_type    : str             "faiss" or "chroma"

    Returns
    -------
    A LangChain VectorStore instance (FAISS or Chroma).
    """
    if not chunks:
        raise ValueError("Cannot build a vector store from an empty chunk list.")

    embeddings = get_embedding_model()
    safe_name  = _sanitize_name(paper_name)

    if db_type == "faiss":
        db = _build_faiss(chunks, embeddings)
        _save_faiss(db, VECTOR_STORE_DIR / safe_name)
    elif db_type == "chroma":
        db = _build_chroma(chunks, embeddings, safe_name)
    else:
        raise ValueError(f"Unknown vector DB type: '{db_type}'. Choose 'faiss' or 'chroma'.")

    logger.info("Vector store ready for paper '%s'.", paper_name)
    return db


def load_vector_store(
    paper_name: str,
    db_type: str = VECTOR_DB_TYPE,
):
    """
    Load a previously built vector store from disk.

    Returns
    -------
    VectorStore instance, or None if no store exists for this paper.
    """
    embeddings = get_embedding_model()
    safe_name  = _sanitize_name(paper_name)

    if db_type == "faiss":
        return _load_faiss(VECTOR_STORE_DIR / safe_name, embeddings)
    elif db_type == "chroma":
        return _load_chroma(safe_name, embeddings)
    else:
        raise ValueError(f"Unknown vector DB type: '{db_type}'.")


def similarity_search(
    vector_store,
    query: str,
    k: int = TOP_K_RESULTS,
) -> List[Document]:
    """
    Retrieve the top-k most relevant chunks for a query.

    Parameters
    ----------
    vector_store : LangChain VectorStore
    query        : str   The user's question or search string
    k            : int   Number of results to return

    Returns
    -------
    List[Document]  Ordered by relevance (most relevant first).
    """
    results = vector_store.similarity_search(query, k=k)
    logger.debug("Retrieved %d chunks for query: %.60s…", len(results), query)
    return results


def similarity_search_with_scores(
    vector_store,
    query: str,
    k: int = TOP_K_RESULTS,
) -> List[Tuple[Document, float]]:
    """
    Like similarity_search but also returns the similarity score for each chunk.
    Useful for debugging retrieval quality.

    Returns
    -------
    List[Tuple[Document, float]]  (chunk, score) pairs, best first.
    """
    return vector_store.similarity_search_with_score(query, k=k)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _sanitize_name(name: str) -> str:
    """Replace characters that are unsafe as directory / collection names."""
    import re
    # Keep alphanumerics, hyphens, underscores; replace the rest with _
    safe = re.sub(r"[^\w\-]", "_", Path(name).stem)
    return safe[:64]   # cap length for filesystem safety
