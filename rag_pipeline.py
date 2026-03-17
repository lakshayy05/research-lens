"""
rag_pipeline.py
---------------
Orchestrates the full Retrieval-Augmented Generation pipeline:

  1. Ingest a PDF   -> extract text -> chunk -> embed -> store in vector DB
  2. Answer queries -> retrieve relevant chunks -> build prompt -> call LLM

All heavy components (embeddings, vector store, LLM) are initialised
lazily and cached so the Streamlit app stays fast across reruns.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

from langchain_core.documents import Document
from langchain_mistralai import ChatMistralAI

from config import (
    MISTRAL_API_KEY,
    MISTRAL_MODEL,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
    TOP_K_RESULTS,
    VECTOR_DB_TYPE,
)
from pdf_loader    import extract_text_from_pdf, get_pdf_metadata, pages_to_full_text
from text_chunking import chunk_pages, attach_source_metadata, get_chunk_stats
from vector_store  import create_vector_store, load_vector_store, similarity_search
from prompts       import (
    RAG_QA_PROMPT,
    SUMMARISE_PROMPT,
    METHODOLOGY_PROMPT,
    CONTRIBUTIONS_PROMPT,
    ELI5_PROMPT,
    EXPLAIN_CONCEPT_PROMPT,
    EQUATION_PROMPT,
)

logger = logging.getLogger(__name__)


# -- LLM factory ---------------------------------------------------------------

def get_llm() -> ChatMistralAI:
    """
    Return a configured Mistral LLM instance.
    Raises ValueError if the API key is missing.
    """
    if not MISTRAL_API_KEY:
        raise ValueError(
            "MISTRAL_API_KEY is not set. "
            "Add it to your .env file or set it as an environment variable."
        )
    return ChatMistralAI(
        api_key=MISTRAL_API_KEY,
        model=MISTRAL_MODEL,
        temperature=LLM_TEMPERATURE,
        max_tokens=LLM_MAX_TOKENS,
    )


# -- Ingestion pipeline --------------------------------------------------------

def ingest_pdf(pdf_path: Path) -> Dict[str, Any]:
    """
    Full ingestion pipeline for a single PDF.

    Steps:
      1. Extract text page by page
      2. Chunk the text
      3. Generate and store embeddings

    Parameters
    ----------
    pdf_path : Path

    Returns
    -------
    Dict with keys:
      - vector_store : the built LangChain VectorStore
      - metadata     : PDF metadata dict
      - chunk_stats  : statistics about the chunking
      - full_text    : concatenated page text (for whole-doc summaries)
    """
    logger.info("-- Ingestion started: %s --", pdf_path.name)

    # Step 1: Extract text
    pages     = extract_text_from_pdf(pdf_path)
    metadata  = get_pdf_metadata(pdf_path)
    full_text = pages_to_full_text(pages)
    logger.info("Extracted %d pages, %d total chars", len(pages), len(full_text))

    # Step 2: Chunk
    chunks = chunk_pages(pages)
    chunks = attach_source_metadata(chunks, pdf_path.name)
    stats  = get_chunk_stats(chunks)
    logger.info("Chunk stats: %s", stats)

    # Step 3: Embed + store
    vs = create_vector_store(chunks, pdf_path.name, db_type=VECTOR_DB_TYPE)

    logger.info("-- Ingestion complete: %s --", pdf_path.name)
    return {
        "vector_store": vs,
        "metadata":     metadata,
        "chunk_stats":  stats,
        "full_text":    full_text,
    }


# -- Shared helper -------------------------------------------------------------

def _format_context(docs: List[Document]) -> str:
    """Concatenate retrieved chunks into a single context block for the LLM."""
    parts = []
    for i, doc in enumerate(docs, 1):
        page_info = doc.metadata.get("page_number", "?")
        parts.append(f"[Chunk {i} | Page {page_info}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


# -- Query pipeline ------------------------------------------------------------

def answer_question(
    vector_store,
    question: str,
    k: int = TOP_K_RESULTS,
) -> Dict[str, Any]:
    """
    Retrieve relevant context and ask the LLM to answer the question.

    Parameters
    ----------
    vector_store : LangChain VectorStore
    question     : str
    k            : int  Number of chunks to retrieve

    Returns
    -------
    Dict with keys:
      - answer   : str
      - sources  : List[Document]
      - context  : str  (formatted context sent to LLM)
    """
    docs    = similarity_search(vector_store, question, k=k)
    context = _format_context(docs)

    llm    = get_llm()
    chain  = RAG_QA_PROMPT | llm
    result = chain.invoke({"context": context, "question": question})

    return {
        "answer":  result.content.strip(),
        "sources": docs,
        "context": context,
    }


def summarise_paper(full_text: str, max_chars: int = 12_000) -> str:
    """
    Generate a structured summary of the full paper text.

    Parameters
    ----------
    full_text : str   Complete concatenated paper text
    max_chars : int   Truncate to this many chars to fit the LLM context window

    Returns
    -------
    str  Formatted summary
    """
    if len(full_text) > max_chars:
        half = max_chars // 2
        text = full_text[:half] + "\n\n[... middle of paper truncated ...]\n\n" + full_text[-half:]
    else:
        text = full_text

    llm    = get_llm()
    chain  = SUMMARISE_PROMPT | llm
    result = chain.invoke({"context": text})
    return result.content.strip()


def explain_methodology(vector_store) -> str:
    """Retrieve methodology-related chunks and explain them."""
    query   = "methodology experimental design algorithm approach method"
    docs    = similarity_search(vector_store, query, k=6)
    context = _format_context(docs)

    llm    = get_llm()
    chain  = METHODOLOGY_PROMPT | llm
    result = chain.invoke({"context": context})
    return result.content.strip()


def extract_contributions(vector_store) -> str:
    """Retrieve and list key contributions of the paper."""
    query   = "contribution novel approach proposed method key findings"
    docs    = similarity_search(vector_store, query, k=6)
    context = _format_context(docs)

    llm    = get_llm()
    chain  = CONTRIBUTIONS_PROMPT | llm
    result = chain.invoke({"context": context})
    return result.content.strip()


def explain_eli5(vector_store) -> str:
    """Generate a beginner-friendly, simple explanation of the paper."""
    query   = "main idea overview purpose of the paper"
    docs    = similarity_search(vector_store, query, k=5)
    context = _format_context(docs)

    llm    = get_llm()
    chain  = ELI5_PROMPT | llm
    result = chain.invoke({"context": context})
    return result.content.strip()


def explain_concept(vector_store, concept: str) -> str:
    """Explain a specific concept mentioned in the paper."""
    docs    = similarity_search(vector_store, concept, k=5)
    context = _format_context(docs)

    llm    = get_llm()
    chain  = EXPLAIN_CONCEPT_PROMPT | llm
    result = chain.invoke({"context": context, "concept": concept})
    return result.content.strip()


def explain_equation(vector_store, equation: str) -> str:
    """Explain a mathematical equation from the paper."""
    docs    = similarity_search(vector_store, equation, k=4)
    context = _format_context(docs)

    llm    = get_llm()
    chain  = EQUATION_PROMPT | llm
    result = chain.invoke({"context": context, "equation": equation})
    return result.content.strip()