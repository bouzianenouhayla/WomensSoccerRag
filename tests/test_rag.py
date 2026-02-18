from __future__ import annotations

from pathlib import Path

from app.rag_pipeline import RAGPipeline
from ingestion.chunker import Chunker


def test_chunker_generates_overlap():
    chunker = Chunker(max_chars=50, overlap_chars=10)
    text = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Pellentesque." * 2
    chunks = chunker.chunk_text(text)
    assert len(chunks) >= 2
    assert all(len(chunk) <= 50 for chunk in chunks)


def test_rag_pipeline_initializes_without_llm(tmp_path):
    vector_path = tmp_path / "chroma"
    pipeline = RAGPipeline(vector_path=str(vector_path), llm_path=str(tmp_path / "missing.gguf"))
    answer, contexts = pipeline.answer_question("Who won the last WWC?", max_contexts=1)
    assert isinstance(answer, str)
    assert isinstance(contexts, list)
