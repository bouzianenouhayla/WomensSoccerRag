from abc import ABC, abstractmethod

from app.models import PipelineResult


class BasePipeline(ABC):
    """Contract that every pipeline strategy must satisfy."""

    config_name: str

    @abstractmethod
    def answer_question(self, question: str, **kwargs) -> PipelineResult:
        """Run the pipeline and return a structured result.

        Args:
            question: Natural language question from the user.
            **kwargs: Pipeline-specific options (e.g. max_contexts).

        Returns:
            PipelineResult with answer, contexts, config name, and timing.
        """
