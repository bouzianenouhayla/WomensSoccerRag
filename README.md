# Women's Soccer RAG

A RAG evaluation framework for women's football — comparing local vs API LLMs and local vs remote vector stores across consistent quality and latency metrics.

## Stack

- **FastAPI** backend with a lightweight chat UI
- **ChromaDB** local vector store
- **SentenceTransformer** embeddings (`all-MiniLM-L6-v2`)
- **Phi-3 mini** (GGUF via `llama-cpp-python`) — local LLM
- **Claude Haiku** (Anthropic API) — API LLM
- **LangSmith** tracing on all pipeline steps
- Evaluation harness with token F1 and retrieval recall metrics

## Setup

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in your API keys:

```
ANTHROPIC_API_KEY=...
LANGSMITH_API_KEY=...
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=womens-soccer-rag
```

Place your GGUF model under `models/` (e.g. `phi-3-mini.gguf`).

Ingest data into ChromaDB:

```bash
python ingestion/ingest.py
```

Run the app:

```bash
python main.py
```

## Evaluation

Run a single config and append results to `evaluation/results/all_runs.json`:

```bash
python evaluation/run_eval.py
```

Run all configs (local RAG, LLM-only, strict retrieval, Anthropic):

```bash
python evaluation/compare.py
```

## Tests

```bash
pytest
```

## Code Quality

Pre-commit hooks run ruff on every commit:

```bash
pip install pre-commit && pre-commit install
```
