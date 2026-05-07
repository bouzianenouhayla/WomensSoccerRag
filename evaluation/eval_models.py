from typing import List, Optional

from pydantic import BaseModel, Field


class EvalSample(BaseModel):
    """A single evaluation question with its reference answer."""

    id: str
    question: str
    reference_answer: str
    tags: List[str] = Field(default_factory=list)
    context_ids: Optional[List[str]] = None


class EvalResult(BaseModel):
    """Result for a single question against a single pipeline config."""

    sample_id: str
    config_name: str
    question: str
    answer: str
    reference_answer: str
    answer_f1: float
    retrieval_recall: float
    total_time_ms: float
    retrieval_time_ms: float
    llm_time_ms: float


class EvalRunSummary(BaseModel):
    """Aggregated results for one full eval run across all questions."""

    config_name: str
    results: List[EvalResult] = Field(default_factory=list)

    @property
    def average_f1(self) -> float:
        """Mean token F1 across all samples."""
        if not self.results:
            return 0.0
        return sum(r.answer_f1 for r in self.results) / len(self.results)

    @property
    def average_retrieval_recall(self) -> float:
        """Mean retrieval recall across all samples."""
        if not self.results:
            return 0.0
        return sum(r.retrieval_recall for r in self.results) / len(self.results)

    @property
    def average_total_time_ms(self) -> float:
        """Mean total latency in milliseconds across all samples."""
        if not self.results:
            return 0.0
        return sum(r.total_time_ms for r in self.results) / len(self.results)

    @property
    def average_llm_time_ms(self) -> float:
        """Mean LLM generation latency in milliseconds across all samples."""
        if not self.results:
            return 0.0
        return sum(r.llm_time_ms for r in self.results) / len(self.results)
