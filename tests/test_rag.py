from __future__ import annotations

from app.rag_pipeline import RAGPipeline
from ingestion.chunker import Chunker


def test_chunker_generates_overlap():
    chunker = Chunker(max_chars=50, overlap_chars=10)
    text = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Pellentesque." * 2
    chunks = chunker.chunk_text(text)
    assert len(chunks) >= 2
    assert all(len(chunk) <= 50 for chunk in chunks)


def test_rag_pipeline_initializes(tmp_path):
    pipeline = RAGPipeline(config_name="test")
    assert pipeline.config_name == "test"
    assert pipeline.use_retrieval is True
    assert pipeline.min_score == 0.0
