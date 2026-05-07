from abc import ABC, abstractmethod


class BaseLLM(ABC):
    """Contract that every LLM backend must satisfy."""

    @abstractmethod
    def build_prompt(self, system: str, context: str, question: str) -> str | dict:
        """Format inputs into the structure this model expects.

        Args:
            system: System instruction string.
            context: Retrieved context text, or empty string if none.
            question: User question.

        Returns:
            Formatted prompt — a string for local models, a dict for API models.
        """

    @abstractmethod
    def generate(self, prompt: str | dict) -> str:
        """Generate a response from the formatted prompt.

        Args:
            prompt: Output of build_prompt().

        Returns:
            Generated response string.
        """
