from typing import List

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3)
    max_contexts: int = Field(5, ge=1, le=20)
    backend: str = Field("local", pattern="^(local|anthropic)$")


class RetrievedContext(BaseModel):
    chunk_id: str
    text: str
    score: float


class PipelineResult(BaseModel):
    """Full output of one pipeline run, including timing and retrieval metadata."""

    answer: str
    contexts: List[RetrievedContext]
    config_name: str
    total_time_ms: float
    retrieval_time_ms: float
    llm_time_ms: float
    # False when pipeline runs in LLM-only baseline mode
    retrieval_used: bool = True
    # 0 means retrieval ran but nothing passed the score threshold
    chunks_retrieved: int = 0


class QueryResponse(BaseModel):
    """HTTP response shape — trimmed subset of PipelineResult."""

    answer: str
    contexts: List[RetrievedContext]
    config_name: str
    total_time_ms: float
