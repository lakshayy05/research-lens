"""
pdf_loader.py
-------------
Handles all PDF ingestion:
  - Save the uploaded file to disk
  - Extract raw text page-by-page
  - Return structured metadata alongside the text

Uses PyMuPDF (fitz) as the primary parser because it handles complex
academic layouts (multi-column, equations, tables) better than most
alternatives.  Falls back to pypdf when fitz is unavailable.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

import streamlit as st

from config import UPLOADED_PAPERS_DIR

logger = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _try_import_fitz():
    """Try to import PyMuPDF; return None if not installed."""
    try:
        import fitz  # PyMuPDF
        return fitz
    except ImportError:
        return None


def _try_import_pypdf():
    """Try to import pypdf; return None if not installed."""
    try:
        from pypdf import PdfReader
        return PdfReader
    except ImportError:
        return None


# ── Core functions ────────────────────────────────────────────────────────────

def save_uploaded_pdf(uploaded_file) -> Path:
    """
    Persist a Streamlit UploadedFile object to the uploads directory.

    Parameters
    ----------
    uploaded_file : streamlit.runtime.uploaded_file_manager.UploadedFile

    Returns
    -------
    Path
        Absolute path of the saved file.
    """
    save_path = UPLOADED_PAPERS_DIR / uploaded_file.name
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    logger.info("Saved uploaded PDF → %s", save_path)
    return save_path


def extract_text_from_pdf(pdf_path: Path) -> List[Dict[str, Any]]:
    """
    Extract text from each page of a PDF, returning a list of page records.

    Each record has the shape:
    {
        "page_number": int,          # 1-based
        "text":        str,          # raw extracted text
        "char_count":  int,
    }

    Parameters
    ----------
    pdf_path : Path
        Path to the PDF file.

    Returns
    -------
    List[Dict[str, Any]]
        Ordered list of page records.

    Raises
    ------
    FileNotFoundError
        If pdf_path does not exist.
    RuntimeError
        If no suitable PDF parser is available.
    """
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    fitz = _try_import_fitz()
    if fitz:
        return _extract_with_fitz(fitz, pdf_path)

    PdfReader = _try_import_pypdf()
    if PdfReader:
        return _extract_with_pypdf(PdfReader, pdf_path)

    raise RuntimeError(
        "No PDF parser available. "
        "Install PyMuPDF (`pip install pymupdf`) or pypdf (`pip install pypdf`)."
    )


def _extract_with_fitz(fitz, pdf_path: Path) -> List[Dict[str, Any]]:
    """Extract text using PyMuPDF (fitz)."""
    pages = []
    with fitz.open(str(pdf_path)) as doc:
        logger.info("PyMuPDF: reading %d pages from '%s'", len(doc), pdf_path.name)
        for page_num, page in enumerate(doc, start=1):
            text = page.get_text("text")  # plain text; use "blocks" for layout-aware
            pages.append({
                "page_number": page_num,
                "text": text.strip(),
                "char_count": len(text),
            })
    return pages


def _extract_with_pypdf(PdfReader, pdf_path: Path) -> List[Dict[str, Any]]:
    """Extract text using pypdf as a fallback."""
    pages = []
    reader = PdfReader(str(pdf_path))
    logger.info("pypdf: reading %d pages from '%s'", len(reader.pages), pdf_path.name)
    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        pages.append({
            "page_number": page_num,
            "text": text.strip(),
            "char_count": len(text),
        })
    return pages


def get_pdf_metadata(pdf_path: Path) -> Dict[str, Any]:
    """
    Return basic metadata about a PDF (title, author, page count, etc.).

    Parameters
    ----------
    pdf_path : Path

    Returns
    -------
    Dict[str, Any]
        Metadata dictionary; values may be None if the PDF has no metadata.
    """
    fitz = _try_import_fitz()
    if fitz:
        with fitz.open(str(pdf_path)) as doc:
            meta = doc.metadata or {}
            return {
                "title":     meta.get("title") or pdf_path.stem,
                "author":    meta.get("author"),
                "subject":   meta.get("subject"),
                "page_count": len(doc),
                "file_name": pdf_path.name,
                "file_size_kb": round(pdf_path.stat().st_size / 1024, 1),
            }

    PdfReader = _try_import_pypdf()
    if PdfReader:
        reader = PdfReader(str(pdf_path))
        info = reader.metadata or {}
        return {
            "title":      getattr(info, "title", None) or pdf_path.stem,
            "author":     getattr(info, "author", None),
            "subject":    getattr(info, "subject", None),
            "page_count": len(reader.pages),
            "file_name":  pdf_path.name,
            "file_size_kb": round(pdf_path.stat().st_size / 1024, 1),
        }

    return {"file_name": pdf_path.name, "page_count": "unknown"}


def pages_to_full_text(pages: List[Dict[str, Any]]) -> str:
    """
    Concatenate all page texts into a single string (useful for quick summaries).

    Parameters
    ----------
    pages : List[Dict[str, Any]]
        Output of extract_text_from_pdf.

    Returns
    -------
    str
        Full document text with page separators.
    """
    parts = [f"[Page {p['page_number']}]\n{p['text']}" for p in pages if p["text"]]
    return "\n\n".join(parts)
