import argparse
import json
import logging
import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from ragas import evaluate
from ragas import EvaluationDataset, SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

from app.agent_pipeline import AgentPipeline
from app.backends.llm.anthropic_llm import AnthropicLLM
from app.backends.pipeline.base import BasePipeline
from app.rag_pipeline import RAGPipeline
from evaluation.eval_models import EvalSample

load_dotenv(Path(__file__).parent.parent / ".env")

EVAL_PATH = Path(__file__).with_name("eval_dataset.json")
RESULTS_DIR = Path(__file__).parent / "results"
logger = logging.getLogger(__name__)

CONFIGS: dict[str, BasePipeline] = {
    "local": RAGPipeline(config_name="local-phi3-chroma"),
    "anthropic": RAGPipeline(llm=AnthropicLLM(), config_name="anthropic-haiku-chroma"),
    "agent-laws": AgentPipeline(tools=["search_laws"]),
    "agent-stats": AgentPipeline(tools=["search_stats"]),
    "agent-all": AgentPipeline(tools=["search_laws", "search_stats"]),
}


def load_dataset() -> list[EvalSample]:
    """Load eval samples from eval_dataset.json.

    Returns:
        List of validated EvalSample objects.
    """
    with open(EVAL_PATH, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    return [EvalSample.model_validate(item) for item in raw]


def build_ragas_dataset(
    pipeline: BasePipeline,
    samples: list[EvalSample],
) -> tuple[EvaluationDataset, list[dict]]:
    """Run the pipeline on every sample and format results for Ragas.

    Args:
        pipeline: Any BasePipeline instance to evaluate.
        samples: Eval samples to run the pipeline against.

    Returns:
        Tuple of (Ragas EvaluationDataset, list of raw timing/answer dicts).
    """
    ragas_samples = []
    raw_results = []

    for sample in samples:
        result = pipeline.answer_question(sample.question)
        context_texts = [c.text for c in result.contexts]

        ragas_samples.append(
            SingleTurnSample(
                user_input=sample.question,
                response=result.answer,
                retrieved_contexts=context_texts if context_texts else [""],
                reference=sample.reference_answer,
            )
        )
        raw_results.append(
            {
                "sample_id": sample.id,
                "question": sample.question,
                "answer": result.answer,
                "reference_answer": sample.reference_answer,
                "expected_tool": sample.expected_tool,
                "total_time_ms": result.total_time_ms,
                "retrieval_time_ms": result.retrieval_time_ms,
                "llm_time_ms": result.llm_time_ms,
                "chunks_retrieved": result.chunks_retrieved,
            }
        )

    return EvaluationDataset(samples=ragas_samples), raw_results


def run(config: str = "local") -> None:
    """Evaluate a pipeline config with Ragas LLM-as-judge metrics.

    Args:
        config: Pipeline config name. One of the keys in CONFIGS.
                Anthropic Haiku is always used as the judge.
    """
    pipeline = CONFIGS[config]

    judge_llm = LangchainLLMWrapper(
        ChatAnthropic(
            model="claude-haiku-4-5-20251001",
            api_key=os.getenv("ANTHROPIC_API_KEY"),
        )
    )

    samples = load_dataset()
    logger.info("Running pipeline on %d samples...", len(samples))

    dataset, raw_results = build_ragas_dataset(pipeline, samples)

    logger.info("Running Ragas evaluation (LLM-as-judge)...")
    results = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=judge_llm,
    )

    scores = results.to_pandas()

    RESULTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"ragas_{pipeline.config_name}_{timestamp}.json"

    output = {
        "config_name": pipeline.config_name,
        "timestamp": timestamp,
        "summary": {
            "faithfulness": float(scores["faithfulness"].mean()),
            "answer_relevancy": float(scores["answer_relevancy"].mean()),
            "context_precision": float(scores["context_precision"].mean()),
            "context_recall": float(scores["context_recall"].mean()),
        },
        "per_sample": json.loads(scores.to_json(orient="records")),
        "pipeline_timing": raw_results,
    }

    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

    print(f"\n=== Ragas Results: {pipeline.config_name} ===")
    for metric, score in output["summary"].items():
        print(f"  {metric:<25} {score:.3f}")
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        choices=list(CONFIGS.keys()),
        default="local",
        help=f"Which pipeline to evaluate. Options: {', '.join(CONFIGS.keys())}",
    )
    args = parser.parse_args()
    run(config=args.config)
