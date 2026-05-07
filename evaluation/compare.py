"""Run a single pipeline config and append results to evaluation/results/all_runs.json.

Usage:
    python -m evaluation.compare --config rag          # RAG with no score filter
    python -m evaluation.compare --config llm-only     # LLM only, no retrieval
    python -m evaluation.compare --config rag-strict   # RAG with 0.5 score threshold
"""

import argparse
import logging

from app.backends.llm.anthropic_llm import AnthropicLLM
from app.rag_pipeline import RAGPipeline
from evaluation.run_eval import run

CONFIGS = {
    "rag": RAGPipeline(
        config_name="local-phi3-chroma",
        use_retrieval=True,
        min_score=0.0,
    ),
    "llm-only": RAGPipeline(
        config_name="local-phi3-llm-only",
        use_retrieval=False,
    ),
    "rag-strict": RAGPipeline(
        config_name="local-phi3-chroma-strict",
        use_retrieval=True,
        min_score=0.5,
    ),
    "anthropic-rag": RAGPipeline(
        llm=AnthropicLLM(),
        config_name="anthropic-haiku-chroma",
        use_retrieval=True,
        min_score=0.0,
    ),
    "anthropic-llm-only": RAGPipeline(
        llm=AnthropicLLM(),
        config_name="anthropic-haiku-llm-only",
        use_retrieval=False,
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run eval for one pipeline config.")
    parser.add_argument(
        "--config",
        choices=list(CONFIGS.keys()),
        required=True,
        help=f"Which config to evaluate. Options: {', '.join(CONFIGS.keys())}",
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    pipeline = CONFIGS[args.config]
    run(pipeline=pipeline)
