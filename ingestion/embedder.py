from __future__ import annotations

from functools import lru_cache
from typing import List

from sentence_transformers import SentenceTransformer

class EmbeddingClient:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self._model = self._load_model(model_name)

    @staticmethod
    @lru_cache(maxsize=2)
    def _load_model(model_name: str) -> SentenceTransformer:
        return SentenceTransformer(model_name)

    def embed_text(self, text: str) -> List[float]:
        return self._model.encode(text, normalize_embeddings=True).tolist()

    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        return self._model.encode(documents, normalize_embeddings=True).tolist()
