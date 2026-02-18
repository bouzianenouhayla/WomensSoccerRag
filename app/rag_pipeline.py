from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import chromadb
from chromadb.api import Collection

from ingestion.embedder import EmbeddingClient

try:
    from llama_cpp import Llama
except ImportError:  # pragma: no cover - optional dependency
    Llama = None  # type: ignore

DEFAULT_COLLECTION_NAME = "womens_soccer_rag"
DEFAULT_VECTOR_PATH = "vectorstore/chroma_store"
DEFAULT_LLM_PATH = "models/phi-3-mini-4k-instruct-q4.gguf"
DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions about women's football using the provided "
    "context. Cite key laws or data points from the context."
)


class RAGPipeline:
    def __init__(
        self,
        vector_path: str = DEFAULT_VECTOR_PATH,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        llm_path: str = DEFAULT_LLM_PATH,
        max_tokens: int = 512,
    ) -> None:
        self.embedder = EmbeddingClient(model_name=embedding_model)
        Path(vector_path).mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=vector_path)
        self.collection: Collection = client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self.llm = self._load_llm(llm_path)
        self.max_tokens = max_tokens

    def _load_llm(self, model_path: str):
        if Llama is None or not Path(model_path).exists():
            return None
        return Llama(
            model_path=model_path,
            n_ctx=4096,
            n_threads=0,
            n_batch=512,
            logits_all=False,
            verbose=False,
        )

    def retrieve(self, question: str, k: int = 5) -> List[Dict[str, Any]]:
        query_embedding = self.embedder.embed_text(question)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=["documents", "metadatas", "embeddings", "distances"],
        )
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        ids = results.get("ids", [[]])[0]
        contexts: List[Dict[str, Any]] = []
        for chunk_id, doc, meta, dist in zip(ids, documents, metadatas, distances):
            contexts.append(
                {
                    "chunk_id": chunk_id,
                    "text": doc,
                    "score": float(1 - dist) if dist is not None else 0.0,
                    "metadata": meta or {},
                }
            )
        return contexts

    def _build_prompt(self, question: str, contexts: List[Dict[str, Any]]) -> str:
        context_text = "\n\n".join(
            f"[Chunk {c['chunk_id']}] {c['text']}" for c in contexts
        )
        instructions = "Answer in <=3 sentences. Lead with the key fact and cite the chunk id when possible."
        return (
            f"{DEFAULT_SYSTEM_PROMPT}\n\nContext:\n{context_text}\n\n"
            f"Question: {question}\n{instructions}"
        )

    def _llm_answer(self, prompt: str) -> str:
        if self.llm is None:
            return "LLM model not configured. Provide a local GGUF model at models/ and install llama-cpp-python."
        output = self.llm(
            prompt,
            max_tokens=self.max_tokens,
            temperature=0.2,
            top_k=40,
            top_p=0.9,
            stop=["</s>", "User:"],
        )
        choices = output.get("choices", [])
        if not choices:
            return "No response generated."
        return choices[0].get("text", "").strip()

    def answer_question(self, question: str, max_contexts: int = 5):
        contexts = self.retrieve(question, k=max_contexts)
        prompt = self._build_prompt(question, contexts)
        answer = self._llm_answer(prompt)
        return answer, contexts
