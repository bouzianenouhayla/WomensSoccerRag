from functools import lru_cache
from typing import List

from sentence_transformers import SentenceTransformer

from .base import BaseEmbedder


class LocalEmbedder(BaseEmbedder):
    """Embedder backed by a local sentence-transformers model."""

    def __init__(
        self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    ) -> None:
        """Args:
        model_name: HuggingFace model ID to load.
        """
        self.model_name = model_name
        self._model = self._load_model(model_name)

    @staticmethod
    @lru_cache(maxsize=2)
    def _load_model(model_name: str) -> SentenceTransformer:
        """Load and cache a SentenceTransformer model.

        Args:
            model_name: HuggingFace model ID.

        Returns:
            Loaded SentenceTransformer instance.
        """
        return SentenceTransformer(model_name)

    def embed_text(self, text: str) -> List[float]:
        """Embed a single string into a normalised vector.

        Args:
            text: Input string to embed.

        Returns:
            List of floats representing the embedding.
        """
        return self._model.encode(text, normalize_embeddings=True).tolist()

    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        """Embed a list of strings into normalised vectors.

        Args:
            documents: List of strings to embed.

        Returns:
            List of embedding vectors, one per document.
        """
        return self._model.encode(documents, normalize_embeddings=True).tolist()


# Keep old name as alias so nothing else breaks
EmbeddingClient = LocalEmbedder
