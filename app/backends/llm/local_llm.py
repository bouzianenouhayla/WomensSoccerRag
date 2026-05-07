import logging
from pathlib import Path

from langsmith import traceable
from pydantic import BaseModel, Field

from .base import BaseLLM

try:
    from llama_cpp import Llama
except ImportError:
    Llama = None  # type: ignore

logger = logging.getLogger(__name__)


class LocalLLMConfig(BaseModel):
    """Validated configuration for a local GGUF model."""

    model_path: str = "models/Phi-3-mini-4k-instruct-q4.gguf"
    max_tokens: int = Field(512, gt=0)
    temperature: float = Field(0.2, ge=0.0, le=2.0)
    top_k: int = Field(40, gt=0)
    top_p: float = Field(0.9, gt=0.0, le=1.0)
    n_ctx: int = Field(4096, gt=0)


class LocalLLM(BaseLLM):
    """LLM backend that runs a GGUF model locally via llama_cpp."""

    def __init__(self, config: LocalLLMConfig | None = None) -> None:
        """Args:
        config: Model configuration. Defaults to Phi-3 mini at models/.
        """
        self.config = config or LocalLLMConfig()
        self._model = self._load(self.config)

    @staticmethod
    def _load(config: LocalLLMConfig) -> "Llama | None":
        """Load the GGUF model from disk.

        Args:
            config: Validated model configuration.

        Returns:
            Loaded Llama instance, or None if llama_cpp is missing or path not found.
        """
        if Llama is None:
            logger.warning("llama_cpp not installed — LLM disabled.")
            return None
        if not Path(config.model_path).exists():
            logger.warning("Model not found at %s — LLM disabled.", config.model_path)
            return None
        return Llama(
            model_path=config.model_path,
            n_ctx=config.n_ctx,
            n_threads=0,
            n_batch=512,
            logits_all=False,
            verbose=False,
        )

    def build_prompt(self, system: str, context: str, question: str) -> str:
        """Format inputs using the Phi-3 chat template.

        Args:
            system: System instruction string.
            context: Retrieved context text, or empty string if none.
            question: User question.

        Returns:
            Formatted prompt string with Phi-3 special tokens.
        """
        return (
            f"<|system|>\n{system}<|end|>\n"
            f"<|user|>\nContext:\n{context}\n\nQuestion: {question}<|end|>\n"
            f"<|assistant|>\n"
        )

    @traceable(name="LocalLLM.generate")
    def generate(self, prompt: str) -> str:
        """Run inference and return the generated text.

        Args:
            prompt: Formatted prompt string from build_prompt().

        Returns:
            Generated response string, or an error message if the model is unavailable.
        """
        if self._model is None:
            return "LLM not available. Check model path and llama_cpp installation."
        output = self._model(
            prompt,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            top_k=self.config.top_k,
            top_p=self.config.top_p,
            stop=["<|end|>", "<|user|>", "</s>"],
        )
        choices = output.get("choices", [])
        if not choices:
            return "No response generated."
        return choices[0].get("text", "").strip()
