"""
app.py
------
Streamlit frontend for the AI Research Paper Explainer.
"""

import logging
import hashlib
import re
from pathlib import Path

import streamlit as st

from config import APP_TITLE, APP_ICON, MISTRAL_API_KEY
from pdf_loader  import save_uploaded_pdf
from rag_pipeline import (
    ingest_pdf,
    answer_question,
    summarise_paper,
    explain_methodology,
    extract_contributions,
    explain_eli5,
    explain_concept,
    explain_equation,
)

# ── Inline helper functions (no utils folder needed) ─────────────────────────

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )

def file_hash(path: Path) -> str:
    """Short SHA-256 fingerprint of a file to detect re-uploads."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:12]

def estimate_reading_time(text: str, wpm: int = 200) -> str:
    words = len(text.split())
    minutes = max(1, round(words / wpm))
    return f"~{minutes} min read"

def truncate_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1] + "…"

def format_sources(docs, max_preview: int = 200) -> str:
    lines = []
    for i, doc in enumerate(docs, 1):
        page = doc.metadata.get("page_number", "?")
        preview = truncate_text(doc.page_content.replace("\n", " "), max_preview)
        lines.append(f"**Chunk {i}** (Page {page}): _{preview}_")
    return "\n\n".join(lines)

# ── Initialise ────────────────────────────────────────────────────────────────
setup_logging()
logger = logging.getLogger(__name__)

st.set_page_config(page_title=APP_TITLE, page_icon=APP_ICON, layout="wide")


# ── Session-state defaults ────────────────────────────────────────────────────
def _init_session():
    defaults = {
        "vector_store":  None,
        "metadata":      {},
        "full_text":     "",
        "chat_history":  [],
        "paper_hash":    None,
        "ingested":      False,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

_init_session()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Settings")

    api_key_input = st.text_input(
        "Mistral API Key",
        value=MISTRAL_API_KEY,
        type="password",
        help="Get your key at https://console.mistral.ai/",
    )
    if api_key_input:
        import os
        os.environ["MISTRAL_API_KEY"] = api_key_input

    st.divider()

    st.markdown("### 📊 Current Paper")
    if st.session_state["ingested"]:
        meta = st.session_state["metadata"]
        st.success(f"**{meta.get('title', 'Unknown')}**")
        st.caption(f"Pages: {meta.get('page_count', '?')} | Size: {meta.get('file_size_kb', '?')} KB")
        reading_time = estimate_reading_time(st.session_state["full_text"])
        st.caption(f"Estimated reading time: {reading_time}")

        if st.button("🗑️ Clear paper"):
            for key in ["vector_store", "metadata", "full_text",
                        "chat_history", "paper_hash", "ingested"]:
                st.session_state[key] = (
                    None  if key == "vector_store" else
                    {}    if key == "metadata"     else
                    []    if key == "chat_history"  else
                    False if key == "ingested"      else
                    ""
                )
            st.rerun()
    else:
        st.info("No paper loaded yet.")

    st.divider()
    st.markdown("### ℹ️ About")
    st.markdown(
        "This tool uses **Retrieval-Augmented Generation (RAG)** "
        "to answer questions about your research paper.\n\n"
        "Built with LangChain · Mistral · FAISS · Sentence-Transformers"
    )


# ── Main area ─────────────────────────────────────────────────────────────────
st.title(APP_TITLE)
st.caption("Upload any research paper PDF and ask questions about it in plain English.")

# ── Step 1: Upload ─────────────────────────────────────────────────────────────
st.header("1️⃣  Upload a Research Paper")

uploaded_file = st.file_uploader(
    "Choose a PDF file",
    type=["pdf"],
    help="The paper will be indexed for question answering.",
)

if uploaded_file is not None:
    temp_path    = save_uploaded_pdf(uploaded_file)
    current_hash = file_hash(temp_path)

    if current_hash != st.session_state["paper_hash"]:
        with st.spinner("🔍 Extracting and indexing your paper … (this may take 30–60 seconds)"):
            try:
                result = ingest_pdf(temp_path)
                st.session_state["vector_store"] = result["vector_store"]
                st.session_state["metadata"]     = result["metadata"]
                st.session_state["full_text"]    = result["full_text"]
                st.session_state["chunk_stats"]  = result["chunk_stats"]
                st.session_state["paper_hash"]   = current_hash
                st.session_state["ingested"]     = True
                st.session_state["chat_history"] = []
                st.success(
                    f"✅ Paper indexed! "
                    f"{result['chunk_stats']['total_chunks']} chunks created."
                )
            except Exception as e:
                st.error(f"❌ Failed to process the PDF: {e}")
                logger.exception("Ingestion error")
    else:
        st.info("ℹ️ This paper is already indexed. Ask away!")


# ── Step 2: Quick actions ──────────────────────────────────────────────────────
if st.session_state["ingested"]:
    st.divider()
    st.header("2️⃣  Quick Actions")

    col1, col2, col3, col4 = st.columns(4)

    def _run_action(label, fn, *args):
        with st.spinner(f"Generating: {label} …"):
            try:
                result = fn(*args)
                st.session_state["chat_history"].append(
                    {"role": "assistant", "content": f"**{label}**\n\n{result}"}
                )
            except Exception as e:
                st.error(f"Error: {e}")
                logger.exception("Quick action error: %s", label)

    with col1:
        if st.button("📋 Summarise"):
            _run_action("Paper Summary", summarise_paper, st.session_state["full_text"])
    with col2:
        if st.button("🔬 Methodology"):
            _run_action("Methodology", explain_methodology, st.session_state["vector_store"])
    with col3:
        if st.button("🏆 Contributions"):
            _run_action("Key Contributions", extract_contributions, st.session_state["vector_store"])
    with col4:
        if st.button("🧒 ELI5"):
            _run_action("Simple Explanation", explain_eli5, st.session_state["vector_store"])

    with st.expander("💡 Explain a specific concept or equation"):
        ce_col1, ce_col2 = st.columns(2)
        with ce_col1:
            concept_input = st.text_input("Enter a concept (e.g. 'attention mechanism')")
            if st.button("Explain Concept") and concept_input:
                _run_action(f"Concept: {concept_input}", explain_concept,
                            st.session_state["vector_store"], concept_input)
        with ce_col2:
            eq_input = st.text_input("Paste an equation (e.g. 'softmax(QK^T/√d_k)V')")
            if st.button("Explain Equation") and eq_input:
                _run_action(f"Equation: {eq_input}", explain_equation,
                            st.session_state["vector_store"], eq_input)

    # ── Step 3: Chat ───────────────────────────────────────────────────────────
    st.divider()
    st.header("3️⃣  Chat with the Paper")

    for msg in st.session_state["chat_history"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if user_input := st.chat_input("Ask anything about the paper …"):
        with st.chat_message("user"):
            st.markdown(user_input)
        st.session_state["chat_history"].append({"role": "user", "content": user_input})

        with st.chat_message("assistant"):
            with st.spinner("Thinking …"):
                try:
                    result = answer_question(
                        st.session_state["vector_store"],
                        user_input,
                    )
                    answer = result["answer"]
                    st.markdown(answer)

                    with st.expander("📚 View source chunks"):
                        st.markdown(format_sources(result["sources"]))

                    st.session_state["chat_history"].append(
                        {"role": "assistant", "content": answer}
                    )
                except Exception as e:
                    st.error(f"❌ Error: {e}")
                    logger.exception("Answer generation error")

    if st.session_state["chat_history"]:
        if st.button("🗑️ Clear chat history"):
            st.session_state["chat_history"] = []
            st.rerun()

else:
    st.info("👆 Upload a PDF above to get started.")
    with st.expander("📖 What can I ask?"):
        st.markdown("""
**Example questions after uploading a paper:**
- *Summarise this paper in simple terms*
- *What problem does this paper solve?*
- *Explain the methodology step by step*
- *What are the main contributions?*
- *What datasets were used?*
- *Explain equation [X] in plain English*
- *What are the limitations of this work?*
        """)