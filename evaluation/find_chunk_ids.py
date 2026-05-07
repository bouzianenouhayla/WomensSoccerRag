import argparse
import sys

sys.path.insert(0, ".")

from ingestion.embedder import LocalEmbedder
from ingestion.vectorstores.chroma import ChromaStore


def find_ids(question: str, k: int = 5) -> None:
    """Query the vector store and print chunk IDs relevant to a question.

    Args:
        question: Natural language question to look up.
        k: Number of chunks to retrieve.
    """
    embedder = LocalEmbedder()
    store = ChromaStore()

    embedding = embedder.embed_text(question)
    results = store.query(embedding=embedding, k=k)

    if not results:
        print(
            "No results — is the vector store populated? Run: python -m ingestion.ingest"
        )
        return

    print(f"\nQuestion: {question!r}\n")
    print(f"{'chunk_id':<22}  {'score':>6}  text preview")
    print("-" * 80)
    for r in results:
        preview = r["text"].replace("\n", " ")[:60]
        print(f"{r['chunk_id']:<22}  {r['score']:>6.3f}  {preview}...")

    print("\n--- context_ids to copy ---")
    print([r["chunk_id"] for r in results])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("question", help="Question to look up")
    parser.add_argument("--k", type=int, default=5, help="Number of chunks to retrieve")
    args = parser.parse_args()
    find_ids(args.question, args.k)
