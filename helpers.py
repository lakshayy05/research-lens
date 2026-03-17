"""
utils/helpers.py
----------------
Shared utility functions used across the project.

These helpers handle cross-cutting concerns that don't belong in
any single domain module:
  - Logging setup
  - Input validation
  - Text cleaning / formatting
  - Streamlit session-state helpers
"""

import logging
import re
import hashlib
from pathlib import Path
from typing import Any, Optional

from config import LOG_LEVEL


# ── Logging ───────────────────────────────────────────────────────────────────

def setup_logging(level: str = LOG_LEVEL) -> None:
    """
    Configure root logger with a human-friendly format.
    Call once at application startup.
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )


# ── Text utilities ────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Light cleaning of extracted PDF text:
      - Collapse multiple blank lines into one
      - Remove non-printable characters
      - Strip leading/trailing whitespace

    Parameters
    ----------
    text : str

    Returns
    -------
    str
    """
    # Remove non-printable chars (except common whitespace)
    text = re.sub(r"[^\x09\x0A\x0D\x20-\x7E\u00A0-\uFFFF]", " ", text)
    # Collapse 3+ blank lines → 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Collapse multiple spaces
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def truncate_text(text: str, max_chars: int, suffix: str = "…") -> str:
    """
    Truncate text to max_chars, appending suffix if truncated.

    Parameters
    ----------
    text      : str
    max_chars : int
    suffix    : str

    Returns
    -------
    str
    """
    if len(text) <= max_chars:
        return text
    return text[: max_chars - len(suffix)] + suffix


def word_count(text: str) -> int:
    """Return the number of words in a string."""
    return len(text.split())


def estimate_reading_time(text: str, wpm: int = 200) -> str:
    """
    Estimate how long it would take a human to read the text.

    Parameters
    ----------
    text : str
    wpm  : int  Words per minute (200 is an average adult reading speed).

    Returns
    -------
    str  e.g. "~5 min read"
    """
    words   = word_count(text)
    minutes = max(1, round(words / wpm))
    return f"~{minutes} min read"


# ── File utilities ────────────────────────────────────────────────────────────

def file_hash(path: Path) -> str:
    """
    Compute a short SHA-256 fingerprint for a file.
    Used to detect when the same PDF is re-uploaded without re-indexing.

    Parameters
    ----------
    path : Path

    Returns
    -------
    str  First 12 hex characters of SHA-256 hash.
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def validate_pdf_path(path: Path) -> None:
    """
    Raise an appropriate exception if the path is not a readable PDF.

    Parameters
    ----------
    path : Path

    Raises
    ------
    FileNotFoundError  If the file does not exist.
    ValueError         If the file is not a PDF.
    """
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a .pdf file, got: {path.suffix}")


# ── Streamlit session-state helpers ──────────────────────────────────────────

def st_get(key: str, default: Any = None) -> Any:
    """
    Retrieve a value from Streamlit's session state, returning default if absent.

    Parameters
    ----------
    key     : str
    default : Any

    Returns
    -------
    Any
    """
    try:
        import streamlit as st
        return st.session_state.get(key, default)
    except Exception:
        return default


def st_set(key: str, value: Any) -> None:
    """
    Set a value in Streamlit's session state.

    Parameters
    ----------
    key   : str
    value : Any
    """
    try:
        import streamlit as st
        st.session_state[key] = value
    except Exception:
        pass


def format_sources(docs, max_preview: int = 200) -> str:
    """
    Format retrieved source documents into a readable string for display.

    Parameters
    ----------
    docs        : List[Document]
    max_preview : int  Maximum characters to show per chunk.

    Returns
    -------
    str
    """
    lines = []
    for i, doc in enumerate(docs, 1):
        page = doc.metadata.get("page_number", "?")
        preview = truncate_text(doc.page_content.replace("\n", " "), max_preview)
        lines.append(f"**Chunk {i}** (Page {page}): _{preview}_")
    return "\n\n".join(lines)