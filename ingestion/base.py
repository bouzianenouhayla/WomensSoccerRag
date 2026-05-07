from abc import ABC, abstractmethod
from typing import Any, Dict, List


class BaseChunker(ABC):
    """Contract for all text chunking strategies."""

    @abstractmethod
    def chunk_text(self, text: str) -> List[str]:
        """Split a single text into chunks.

        Args:
            text: Raw input text to split.

        Returns:
            List of text chunk strings.
        """

    def chunk_many(self, texts: List[str]) -> List[str]:
        """Chunk multiple texts and flatten results.

        Args:
            texts: List of raw input texts.

        Returns:
            Flat list of all chunks across all inputs.
        """
        results: List[str] = []
        for text in texts:
            results.extend(self.chunk_text(text))
        return results


class BaseEmbedder(ABC):
    """Contract for all embedding backends."""

    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """Embed a single string into a vector.

        Args:
            text: Input string to embed.

        Returns:
            List of floats representing the embedding.
        """

    @abstractmethod
    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        """Embed a list of strings into vectors.

        Args:
            documents: List of strings to embed.

        Returns:
            List of embedding vectors, one per document.
        """


class BaseVectorStore(ABC):
    """Contract for all vector store backends."""

    @abstractmethod
    def add(
        self,
        documents: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]],
        ids: List[str],
    ) -> None:
        """Store documents with their embeddings.

        Args:
            documents: Raw text strings.
            embeddings: Corresponding embedding vectors.
            metadatas: Per-document metadata dicts.
            ids: Unique string ID for each document.
        """

    @abstractmethod
    def query(self, embedding: List[float], k: int) -> List[Dict[str, Any]]:
        """Return the k most similar documents to the given embedding.

        Args:
            embedding: Query vector to search against.
            k: Number of results to return.

        Returns:
            List of dicts, each with keys: chunk_id, text, score, metadata.
        """
