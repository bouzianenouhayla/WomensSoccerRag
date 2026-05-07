import argparse
import hashlib
import json
import logging
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_anthropic import ChatAnthropic
from langchain_huggingface import HuggingFaceEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.testset import TestsetGenerator

load_dotenv(Path(__file__).parent.parent / ".env")

DATA_DIR = Path("data")
SOURCES_DIR = DATA_DIR / "sources"
RAG_CSV = DATA_DIR / "rag_output.csv"
DEFAULT_OUT = Path("evaluation/eval_dataset.json")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

logger = logging.getLogger(__name__)


def load_langchain_docs() -> list[Document]:
    """Load all source text as LangChain Documents.

    Loads at block level so Ragas can control its own sampling window.

    Returns:
        List of LangChain Document objects with source metadata.
    """
    docs: list[Document] = []

    if RAG_CSV.exists():
        df = pd.read_csv(RAG_CSV)
        for _, row in df.iterrows():
            text = str(row.get("text", "")).strip()
            if text:
                docs.append(
                    Document(page_content=text, metadata={"source": "ifab_laws"})
                )

    if SOURCES_DIR.exists():
        for json_path in SOURCES_DIR.rglob("*.json"):
            with open(json_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            docs.append(
                Document(
                    page_content=json.dumps(data),
                    metadata={"source": json_path.stem},
                )
            )

    logger.info("Loaded %d source documents.", len(docs))
    return docs


def ragas_to_eval_samples(testset_df: pd.DataFrame) -> list[dict]:
    """Convert Ragas testset DataFrame to EvalSample-compatible dicts.

    Args:
        testset_df: DataFrame returned by TestsetGenerator.to_pandas().

    Returns:
        List of dicts matching the EvalSample schema.
    """
    samples = []
    for _, row in testset_df.iterrows():
        question = str(row.get("user_input", "")).strip()
        if not question:
            continue

        sample_id = "gen-" + hashlib.sha256(question.encode()).hexdigest()[:12]
        samples.append(
            {
                "id": sample_id,
                "question": question,
                "reference_answer": str(row.get("reference", "")).strip(),
                "context_ids": None,
                "tags": ["generated", str(row.get("synthesizer_name", "ragas"))],
            }
        )
    return samples


def generate(size: int, out_path: Path) -> None:
    """Generate an eval dataset from source documents using Ragas.

    Args:
        size: Number of question/answer pairs to generate.
        out_path: Path to write the resulting eval_dataset.json.
    """
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError("ANTHROPIC_API_KEY not set. Add it to your .env file.")

    docs = load_langchain_docs()
    if not docs:
        raise RuntimeError(
            "No source documents found. Check data/rag_output.csv exists."
        )

    generator_llm = LangchainLLMWrapper(
        ChatAnthropic(model="claude-haiku-4-5-20251001", api_key=api_key)
    )
    generator_embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    )

    generator = TestsetGenerator(
        llm=generator_llm,
        embedding_model=generator_embeddings,
    )

    logger.info("Generating %d eval samples...", size)
    testset = generator.generate_with_langchain_docs(docs, testset_size=size)
    df = testset.to_pandas()

    samples = ragas_to_eval_samples(df)
    logger.info("Generated %d samples.", len(samples))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(samples, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info("Saved to %s", out_path)

    print(f"\nGenerated {len(samples)} eval samples → {out_path}")
    for s in samples[:3]:
        print(f"  [{s['id']}] {s['question'][:80]}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=20)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    generate(size=args.size, out_path=args.out)
