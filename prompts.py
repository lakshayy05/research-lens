"""
prompts.py
----------
All LLM prompt templates in one place.

Design principles:
  - Every prompt explicitly tells the model to stay grounded in the
    provided context (reduces hallucination).
  - Language is kept simple so beginners benefit from the answers.
  - Each template uses LangChain's PromptTemplate so variables are
    injected safely.
  - Adding a new prompt = adding one entry to PROMPT_REGISTRY.
"""

from langchain_core.prompts import PromptTemplate


# -- Shared system instruction -------------------------------------------------
_SYSTEM_PREAMBLE = """You are an expert research assistant who explains complex academic papers \
to students and beginners in simple, clear, and friendly language.
Always base your answers strictly on the provided context from the paper.
If the context does not contain enough information to answer, say so honestly.
Never make up facts or add information not present in the context.
"""

# -- RAG QA prompt (main prompt used by the pipeline) -------------------------
RAG_QA_TEMPLATE = _SYSTEM_PREAMBLE + """
---
CONTEXT FROM THE PAPER:
{context}
---

QUESTION:
{question}

ANSWER (explain clearly, use simple language, use bullet points where helpful):"""

RAG_QA_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=RAG_QA_TEMPLATE,
)

# -- Summarisation prompt ------------------------------------------------------
SUMMARISE_TEMPLATE = _SYSTEM_PREAMBLE + """
---
FULL TEXT OF THE PAPER:
{context}
---

Please provide a comprehensive yet easy-to-understand summary of this research paper.
Structure your summary with these sections:

1. **One-line Summary** - What is this paper about in one sentence?
2. **Problem Being Solved** - What real-world problem does this paper address?
3. **Key Contributions** - What new ideas, methods, or findings does the paper introduce?
4. **Methodology** - How did the researchers approach the problem?
5. **Main Results** - What did they find or achieve?
6. **Limitations** - What are the weaknesses or open questions?
7. **Why It Matters** - What is the broader impact of this work?

Use plain language. Avoid jargon. If you must use a technical term, explain it."""

SUMMARISE_PROMPT = PromptTemplate(
    input_variables=["context"],
    template=SUMMARISE_TEMPLATE,
)

# -- Explain a specific concept / section -------------------------------------
EXPLAIN_CONCEPT_TEMPLATE = _SYSTEM_PREAMBLE + """
---
CONTEXT FROM THE PAPER:
{context}
---

Please explain the following concept from this paper in the simplest possible way:
"{concept}"

- Use an analogy if it helps.
- Break it down step by step.
- Assume the reader has no prior knowledge of this topic."""

EXPLAIN_CONCEPT_PROMPT = PromptTemplate(
    input_variables=["context", "concept"],
    template=EXPLAIN_CONCEPT_TEMPLATE,
)

# -- Methodology deep-dive ----------------------------------------------------
METHODOLOGY_TEMPLATE = _SYSTEM_PREAMBLE + """
---
CONTEXT FROM THE PAPER:
{context}
---

Explain the methodology of this research paper step by step.
Cover:
- What data or inputs were used
- What algorithms or models were applied
- How the experiments were designed
- How the results were measured

Use numbered steps and simple language."""

METHODOLOGY_PROMPT = PromptTemplate(
    input_variables=["context"],
    template=METHODOLOGY_TEMPLATE,
)

# -- Equation explainer -------------------------------------------------------
EQUATION_TEMPLATE = _SYSTEM_PREAMBLE + """
---
CONTEXT FROM THE PAPER:
{context}
---

The user wants to understand this equation or mathematical expression:
"{equation}"

Please:
1. Name each symbol and what it represents.
2. Explain what the equation is computing in plain English.
3. Give a simple numeric example if possible.
4. Explain why this equation is important to the paper's method."""

EQUATION_PROMPT = PromptTemplate(
    input_variables=["context", "equation"],
    template=EQUATION_TEMPLATE,
)

# -- Key contributions extractor ----------------------------------------------
CONTRIBUTIONS_TEMPLATE = _SYSTEM_PREAMBLE + """
---
CONTEXT FROM THE PAPER:
{context}
---

List the key contributions of this paper.
Format as a numbered list.
For each contribution, write:
  - What it is
  - Why it is new or significant
  - How it advances the field

Keep each point concise (2-3 sentences max)."""

CONTRIBUTIONS_PROMPT = PromptTemplate(
    input_variables=["context"],
    template=CONTRIBUTIONS_TEMPLATE,
)

# -- ELI5 (explain like I'm 5) ------------------------------------------------
ELI5_TEMPLATE = _SYSTEM_PREAMBLE + """
---
CONTEXT FROM THE PAPER:
{context}
---

Explain this research paper as if you were talking to a curious 10-year-old.
Use very simple words, everyday analogies, and short sentences.
Make it fun and engaging. Avoid all technical jargon."""

ELI5_PROMPT = PromptTemplate(
    input_variables=["context"],
    template=ELI5_TEMPLATE,
)


# -- Prompt registry - used by the UI to offer quick-action buttons -----------
PROMPT_REGISTRY = {
    "summarise":        {"prompt": SUMMARISE_PROMPT,        "label": "📋 Summarise the paper"},
    "methodology":      {"prompt": METHODOLOGY_PROMPT,      "label": "🔬 Explain methodology"},
    "contributions":    {"prompt": CONTRIBUTIONS_PROMPT,    "label": "🏆 Key contributions"},
    "eli5":             {"prompt": ELI5_PROMPT,             "label": "🧒 Explain like I'm 5"},
    "qa":               {"prompt": RAG_QA_PROMPT,           "label": "💬 Ask a question"},
    "explain_concept":  {"prompt": EXPLAIN_CONCEPT_PROMPT,  "label": "💡 Explain a concept"},
    "equation":         {"prompt": EQUATION_PROMPT,         "label": "➗ Explain an equation"},
}