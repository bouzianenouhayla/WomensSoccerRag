from pathlib import Path
from typing import Any, Dict, List

import chromadb

from ingestion.base import BaseVectorStore

DEFAULT_PATH = "vectorstore/chroma_store"
DEFAULT_COLLECTION = "womens_soccer_rag"


class ChromaStore(BaseVectorStore):
    """Vector store backed by a local ChromaDB instance."""

    def __init__(
        self,
        path: str = DEFAULT_PATH,
        collection_name: str = DEFAULT_COLLECTION,
    ) -> None:
        """Args:
        path: Directory path for the persistent ChromaDB store.
        collection_name: Name of the ChromaDB collection to use.
        """
        Path(path).mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=path)
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

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
        self._collection.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids,
        )

    def query(self, embedding: List[float], k: int) -> List[Dict[str, Any]]:
        """Return the k most similar documents to the given embedding.

        Args:
            embedding: Query vector to search against.
            k: Number of results to return.

        Returns:
            List of dicts with keys: chunk_id, text, score, metadata.
            Score is cosine similarity (1.0 = identical, 0.0 = unrelated).
        """
        results = self._collection.query(
            query_embeddings=[embedding],
            n_results=k,
            include=["documents", "metadatas", "embeddings", "distances"],
        )
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        ids = results.get("ids", [[]])[0]

        return [
            {
                "chunk_id": chunk_id,
                "text": doc,
                "score": float(1 - dist) if dist is not None else 0.0,
                "metadata": meta or {},
            }
            for chunk_id, doc, meta, dist in zip(ids, documents, metadatas, distances)
        ]
