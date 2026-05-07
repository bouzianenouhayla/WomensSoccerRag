import argparse
import hashlib
import json
from pathlib import Path
from typing import List, Tuple

import pandas as pd

from .base import BaseChunker, BaseEmbedder, BaseVectorStore
from .chunker import CharacterChunker
from .embedder import LocalEmbedder
from .vectorstores.chroma import ChromaStore

DATA_DIR = Path("data")
SOURCES_DIR = DATA_DIR / "sources"
RAG_CSV = DATA_DIR / "rag_output.csv"


def load_rules_blocks() -> List[str]:
    """Load text blocks from the IFAB laws CSV.

    Returns:
        List of raw text strings, one per CSV row.
    """
    if not RAG_CSV.exists():
        return []
    df = pd.read_csv(RAG_CSV)
    return df["text"].dropna().tolist()


def load_source_json() -> List[Tuple[str, str]]:
    """Load all JSON source files as (name, json_string) pairs.

    Returns:
        List of (source_name, json_string) tuples.
    """
    payloads: List[Tuple[str, str]] = []
    if not SOURCES_DIR.exists():
        return payloads
    for json_path in SOURCES_DIR.rglob("*.json"):
        with open(json_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        payloads.append((json_path.stem, json.dumps(data)))
    return payloads


def build_documents(chunker: BaseChunker) -> Tuple[List[str], List[dict]]:
    """Chunk all source documents and return texts with metadata.

    Args:
        chunker: Chunking strategy to apply to each source block.

    Returns:
        Tuple of (chunk_texts, metadatas), one entry per chunk.
    """
    documents: List[str] = []
    metadatas: List[dict] = []

    for block in load_rules_blocks():
        for chunk in chunker.chunk_text(block):
            documents.append(chunk)
            metadatas.append({"source": "ifab_laws_pdf"})

    for source_name, raw_json in load_source_json():
        for chunk in chunker.chunk_text(raw_json):
            documents.append(chunk)
            metadatas.append({"source": source_name})

    return documents, metadatas


def ingest(
    chunker: BaseChunker, embedder: BaseEmbedder, store: BaseVectorStore
) -> None:
    """Chunk, embed, and store all source documents.

    Args:
        chunker: Strategy for splitting documents into chunks.
        embedder: Model for converting chunks to vectors.
        store: Vector database to persist chunks in.
    """
    documents, metadatas = build_documents(chunker)
    if not documents:
        print("No documents to ingest.")
        return
    embeddings = embedder.embed_documents(documents)
    # Deterministic IDs: same text always gets the same ID across re-ingests
    ids = [
        "chunk-" + hashlib.sha256(doc.encode()).hexdigest()[:16] for doc in documents
    ]
    store.add(documents=documents, embeddings=embeddings, metadatas=metadatas, ids=ids)
    print(f"Ingested {len(documents)} chunks.")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments.

    Returns:
        Parsed namespace with max_chars and overlap fields.
    """
    parser = argparse.ArgumentParser(
        description="Ingest documents into a vector store."
    )
    parser.add_argument("--max-chars", type=int, default=800)
    parser.add_argument("--overlap", type=int, default=150)
    return parser.parse_args()


def main() -> None:
    """Entry point: parse args and run ingestion with default backends."""
    args = parse_args()
    chunker = CharacterChunker(max_chars=args.max_chars, overlap_chars=args.overlap)
    embedder = LocalEmbedder()
    store = ChromaStore()
    ingest(chunker=chunker, embedder=embedder, store=store)


if __name__ == "__main__":
    main()
