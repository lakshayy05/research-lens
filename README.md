# 📄 AI Research Paper Explainer

An AI-powered system that lets you upload any research paper (PDF), understand its content through simple explanations, and ask natural-language questions — all powered by Retrieval-Augmented Generation (RAG).

---

## ✨ Features

| Feature | Description |
|---|---|
| **PDF Upload** | Upload any research paper up to 50 MB |
| **Smart Summarisation** | Structured summary with problem, contributions, results |
| **Methodology Explainer** | Step-by-step breakdown of the research approach |
| **Key Contributions** | Auto-extracted list of novel contributions |
| **ELI5 Mode** | Explain the paper like you're talking to a 10-year-old |
| **Concept Explainer** | Ask about any specific concept in the paper |
| **Equation Explainer** | Understand any formula in plain English |
| **RAG Q&A Chat** | Ask any question; answers grounded in the actual paper |
| **Source Attribution** | Every answer shows which chunks it came from |

---

## 🏗️ System Architecture

```
┌──────────────────────────────────────────────────────────┐
│                   Streamlit Frontend (app.py)            │
│   Upload │ Quick Actions │ Chat Interface │ Settings     │
└────────────────────────┬─────────────────────────────────┘
                         │
         ┌───────────────▼───────────────┐
         │       RAG Pipeline            │
         │       (rag_pipeline.py)       │
         └──┬──────────────────────┬─────┘
            │  INGESTION           │  QUERY
            ▼                      ▼
   ┌────────────────┐    ┌─────────────────────┐
   │  pdf_loader    │    │  vector_store        │
   │  (PyMuPDF)     │    │  similarity_search   │
   └───────┬────────┘    └──────────┬──────────┘
           │                        │
   ┌───────▼────────┐    ┌──────────▼──────────┐
   │ text_chunking  │    │   prompts.py         │
   │ (LangChain     │    │   (RAG_QA, Summarise │
   │  Splitter)     │    │    Methodology …)    │
   └───────┬────────┘    └──────────┬──────────┘
           │                        │
   ┌───────▼────────┐    ┌──────────▼──────────┐
   │  embeddings    │    │   Mistral LLM        │
   │ (all-MiniLM)   │    │  (ChatMistralAI)     │
   └───────┬────────┘    └─────────────────────┘
           │
   ┌───────▼────────┐
   │  vector_store  │
   │  FAISS / Chroma│
   └────────────────┘
```

### How the RAG Pipeline Works

1. **Ingest** — User uploads a PDF.
2. **Extract** — PyMuPDF reads the text page-by-page.
3. **Chunk** — The text is split into ~800-character overlapping chunks so nothing gets lost at boundaries.
4. **Embed** — Each chunk is converted to a dense vector using `sentence-transformers/all-MiniLM-L6-v2`.
5. **Store** — All vectors are stored in a FAISS (or Chroma) vector index on disk.
6. **Query** — User asks a question. The question is also embedded.
7. **Retrieve** — The 5 most semantically similar chunks are fetched from the vector store.
8. **Generate** — The retrieved chunks + question are handed to Mistral as a structured prompt.
9. **Answer** — Mistral generates a grounded, beginner-friendly answer.

---

## 🗂️ Project Structure

```
research-paper-explainer/
│
├── app.py              ← Streamlit frontend (entry point)
├── config.py           ← All configuration & environment variables
├── pdf_loader.py       ← PDF ingestion and text extraction (PyMuPDF / pypdf)
├── text_chunking.py    ← Recursive text splitting into overlapping chunks
├── embeddings.py       ← HuggingFace sentence-transformer embeddings
├── vector_store.py     ← FAISS / Chroma vector DB creation and querying
├── rag_pipeline.py     ← Orchestrates the full ingestion + query pipeline
├── prompts.py          ← All LLM prompt templates
│
├── data/
│   ├── uploaded_papers/    ← Saved PDF files
│   └── vector_store/       ← Persisted FAISS indexes (one per paper)
│
├── utils/
│   └── helpers.py          ← Logging, text cleaning, Streamlit utilities
│
├── .env.example        ← Environment variable template
├── requirements.txt    ← Python dependencies
└── README.md
```

---

## 🚀 Setup & Running Locally

### Prerequisites

- Python 3.10 or 3.11
- A [Mistral AI API key](https://console.mistral.ai/)

### 1. Clone the repository

```bash
git clone https://github.com/yourname/research-paper-explainer.git
cd research-paper-explainer
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
# or
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> First run downloads the ~90 MB sentence-transformer model — this is cached locally afterward.

### 4. Configure environment variables

```bash
cp .env.example .env
# Edit .env and add your MISTRAL_API_KEY
```

### 5. Run the app

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`.

---

## ⚙️ Configuration Reference

All settings live in `config.py` and can be overridden via `.env`:

| Variable | Default | Description |
|---|---|---|
| `MISTRAL_API_KEY` | *(required)* | Your Mistral AI key |
| `MISTRAL_MODEL` | `mistral-large-latest` | Model to use |
| `LLM_TEMPERATURE` | `0.3` | Lower = more factual |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence-transformer model |
| `CHUNK_SIZE` | `800` | Characters per chunk |
| `CHUNK_OVERLAP` | `150` | Overlap between chunks |
| `TOP_K_RESULTS` | `5` | Chunks retrieved per query |
| `VECTOR_DB_TYPE` | `faiss` | `faiss` or `chroma` |

---

## 🔮 Future Improvements (Roadmap)

The modular architecture makes these easy to add:

- [ ] **Section-wise summarisation** — Abstract, Introduction, Methodology, Results separately
- [ ] **Key-term extraction** — Auto-glossary from the paper
- [ ] **Equation explanation** — LaTeX rendering with step-by-step breakdown
- [ ] **Figure / table explanation** — Multimodal support via vision LLMs
- [ ] **Multi-paper comparison** — Compare methodology and results across papers
- [ ] **Citation extraction** — Auto-parse the reference list
- [ ] **Paper recommendation** — Find related papers via semantic similarity
- [ ] **Voice interface** — Text-to-speech answers via ElevenLabs / gTTS
- [ ] **Research knowledge graph** — Visualise concept relationships with NetworkX
- [ ] **GPU acceleration** — Swap `faiss-cpu` → `faiss-gpu` for large corpora

---

## 🧪 Running Tests

```bash
# Install test dependencies
pip install pytest pytest-mock

# Run tests
pytest tests/ -v
```

---

## 📜 License

MIT License. Feel free to adapt for your own projects.
