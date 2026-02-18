from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path
from typing import List, Tuple

import chromadb
import pandas as pd

from .chunker import Chunker
from .embedder import EmbeddingClient

DATA_DIR = Path("data")
SOURCES_DIR = DATA_DIR / "sources"
RAG_CSV = DATA_DIR / "rag_output.csv"
VECTOR_PATH = Path("vectorstore/chroma_store")
COLLECTION_NAME = "womens_soccer_rag"

def load_rules_blocks() -> List[str]:
    if not RAG_CSV.exists():
        return []
    df = pd.read_csv(RAG_CSV)
    return df["text"].dropna().tolist()

def load_source_json() -> List[Tuple[str, str]]:
    payloads: List[Tuple[str, str]] = []
    if not SOURCES_DIR.exists():
        return payloads
    for json_path in SOURCES_DIR.rglob("*.json"):
        with open(json_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        payloads.append((json_path.stem, json.dumps(data)))
    return payloads

def build_documents(chunker: Chunker) -> Tuple[List[str], List[dict]]:
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

def ingest_documents(documents: List[str], metadatas: List[dict]) -> None:
    if not documents:
        print("No documents to ingest.")
        return
    VECTOR_PATH.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(VECTOR_PATH))
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    embedder = EmbeddingClient()
    embeddings = embedder.embed_documents(documents)
    ids = [f"chunk-{uuid.uuid4()}" for _ in documents]
    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids,
        embeddings=embeddings,
    )
    print(f"Ingested {len(documents)} chunks into collection '{COLLECTION_NAME}'.")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest documents into ChromaDB.")
    parser.add_argument("--max-chars", type=int, default=800)
    parser.add_argument("--overlap", type=int, default=150)
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    chunker = Chunker(max_chars=args.max_chars, overlap_chars=args.overlap)
    documents, metadatas = build_documents(chunker)
    ingest_documents(documents, metadatas)

if __name__ == "__main__":
    main()
