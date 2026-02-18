from typing import List
from pydantic import BaseModel, Field

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3)
    max_contexts: int = Field(5, ge=1, le=20)

class RetrievedContext(BaseModel):
    chunk_id: str
    text: str
    score: float

class QueryResponse(BaseModel):
    answer: str
    contexts: List[RetrievedContext]
    model: str
