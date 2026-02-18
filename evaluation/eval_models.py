from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field

class EvalSample(BaseModel):
    id: str
    question: str
    reference_answer: str
    tags: List[str] = Field(default_factory=list)
    context_ids: Optional[List[str]] = None

class EvalResult(BaseModel):
    sample_id: str
    answer: str
    reference_answer: str
    answer_f1: float
    retrieval_recall: float

class EvalRunSummary(BaseModel):
    results: List[EvalResult] = Field(default_factory=list)

    @property
    def average_f1(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.answer_f1 for r in self.results) / len(self.results)

    @property
    def average_retrieval_recall(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.retrieval_recall for r in self.results) / len(self.results)
