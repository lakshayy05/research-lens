"""
text_chunking.py
----------------
Responsible for splitting raw PDF text into chunks that are:
  - Small enough to fit in the embedding model's context window
  - Large enough to contain meaningful context
  - Slightly overlapping so no idea gets cut mid-sentence

Uses LangChain's RecursiveCharacterTextSplitter which tries to
split on paragraphs -> sentences -> words, preserving semantic
boundaries as much as possible.
"""

import logging
from typing import List, Dict, Any

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from config import CHUNK_SIZE, CHUNK_OVERLAP

logger = logging.getLogger(__name__)


# -- Core chunker --------------------------------------------------------------

def build_splitter(
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> RecursiveCharacterTextSplitter:
    """
    Construct a RecursiveCharacterTextSplitter with sensible academic-text
    separators (section breaks, double newlines, single newlines, spaces).

    Parameters
    ----------
    chunk_size    : int  Maximum characters per chunk.
    chunk_overlap : int  Characters shared between adjacent chunks.

    Returns
    -------
    RecursiveCharacterTextSplitter
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        # Ordered from coarsest to finest split boundary
        separators=[
            "\n\n\n",   # major section breaks
            "\n\n",     # paragraphs
            "\n",       # single newlines
            ". ",       # sentences
            " ",        # words
            "",         # characters (last resort)
        ],
    )


def chunk_pages(
    pages: List[Dict[str, Any]],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[Document]:
    """
    Take the page-level records from pdf_loader and produce a flat list of
    LangChain Document objects, each representing one chunk.

    Metadata preserved per chunk:
      - source      : original PDF file name
      - page_number : which page the chunk came from
      - chunk_index : sequential index across the full document

    Parameters
    ----------
    pages         : List[Dict] as returned by pdf_loader.extract_text_from_pdf
    chunk_size    : int
    chunk_overlap : int

    Returns
    -------
    List[Document]
    """
    if not pages:
        logger.warning("chunk_pages received an empty pages list.")
        return []

    splitter = build_splitter(chunk_size, chunk_overlap)
    all_chunks: List[Document] = []
    chunk_index = 0

    for page in pages:
        page_text = page.get("text", "").strip()
        if not page_text:
            continue  # skip blank/image-only pages

        # Split this page's text into sub-chunks
        raw_chunks = splitter.split_text(page_text)

        for raw in raw_chunks:
            doc = Document(
                page_content=raw,
                metadata={
                    "page_number": page["page_number"],
                    "chunk_index": chunk_index,
                    "char_count":  len(raw),
                },
            )
            all_chunks.append(doc)
            chunk_index += 1

    logger.info(
        "Chunking complete: %d pages -> %d chunks (size=%d, overlap=%d)",
        len(pages), len(all_chunks), chunk_size, chunk_overlap,
    )
    return all_chunks


def attach_source_metadata(
    chunks: List[Document],
    source_name: str,
) -> List[Document]:
    """
    Attach the PDF filename to every chunk's metadata dict.
    Called after chunking so the file name is always available.

    Parameters
    ----------
    chunks      : List[Document]
    source_name : str  e.g. "attention_is_all_you_need.pdf"

    Returns
    -------
    List[Document]  (same objects, mutated in place)
    """
    for doc in chunks:
        doc.metadata["source"] = source_name
    return chunks


def get_chunk_stats(chunks: List[Document]) -> Dict[str, Any]:
    """
    Return simple statistics about a list of chunks (useful for debugging).

    Parameters
    ----------
    chunks : List[Document]

    Returns
    -------
    Dict[str, Any]
    """
    if not chunks:
        return {"total_chunks": 0}

    sizes = [len(c.page_content) for c in chunks]
    return {
        "total_chunks": len(chunks),
        "avg_chars":    round(sum(sizes) / len(sizes)),
        "min_chars":    min(sizes),
        "max_chars":    max(sizes),
        "total_chars":  sum(sizes),
    }