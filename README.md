# Women's Soccer RAG

**A local Retrieval-Augmented Generation (RAG) system for answering questions about women's football, using open data, ChromaDB, and a small open-source LLM.**

Retrieval-Augmented Generation system focused on women's football rules, tournaments, and statistics.

## Stack

- Flask backend with a lightweight chat UI.
- ChromaDB vector store populated from open sources (IFAB laws PDF, StatsBomb JSON, Kaggle datasets).
- SentenceTransformer embeddings (`all-MiniLM-L6-v2`).
- Local LLM served through `llama-cpp-python` (e.g., Phi-3 mini or Mistral GGUF).
- Evaluation harness using Pydantic schemas and a small gold dataset.

## Setup

> **Python version:** Use CPython 3.11 or 3.12. Several dependencies (Chromadb, llama-cpp) are not yet compatible with experimental 3.14 builds shipped in this repo's default venv.

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python data/sources/fetch_statsbomb.py
python data/sources/fetch_kaggle.py  # requires Kaggle API credentials
python ingestion/ingest.py
python run.py
```

Place your local GGUF model under `models/` and update the path in `app/rag_pipeline.py` if needed.

## Evaluation

```bash
python evaluation/run_eval.py
```

## Tests

```bash
pytest
```
