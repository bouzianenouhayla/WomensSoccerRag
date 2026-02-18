from __future__ import annotations

import json
from pathlib import Path

from app.rag_pipeline import RAGPipeline
from evaluation.eval_models import EvalResult, EvalRunSummary, EvalSample

EVAL_PATH = Path(__file__).with_name("eval_dataset.json")

def token_f1(prediction: str, reference: str) -> float:
    pred_tokens = prediction.lower().split()
    ref_tokens = reference.lower().split()
    common = set(pred_tokens) & set(ref_tokens)
    if not pred_tokens or not ref_tokens:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)

def compute_retrieval_recall(context_ids, retrieved_contexts) -> float:
    if not context_ids:
        return 1.0
    retrieved_ids = {ctx["chunk_id"] for ctx in retrieved_contexts}
    hits = len(set(context_ids) & retrieved_ids)
    return hits / len(context_ids)

def load_dataset() -> list[EvalSample]:
    with open(EVAL_PATH, "r", encoding="utf-8") as fh:
        raw_samples = json.load(fh)
    return [EvalSample.model_validate(item) for item in raw_samples]

def run() -> EvalRunSummary:
    samples = load_dataset()
    pipeline = RAGPipeline()
    results: list[EvalResult] = []
    for sample in samples:
        answer, contexts = pipeline.answer_question(sample.question)
        f1 = token_f1(answer, sample.reference_answer)
        recall = compute_retrieval_recall(sample.context_ids, contexts)
        results.append(
            EvalResult(
                sample_id=sample.id,
                answer=answer,
                reference_answer=sample.reference_answer,
                answer_f1=f1,
                retrieval_recall=recall,
            )
        )
    summary = EvalRunSummary(results=results)
    print(
        f"Evaluated {len(results)} samples | F1={summary.average_f1:.3f} | "
        f"Retrieval Recall={summary.average_retrieval_recall:.3f}"
    )
    return summary

if __name__ == "__main__":
    run()
