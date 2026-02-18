from __future__ import annotations

import json
from pathlib import Path

from evaluation.eval_models import EvalSample


def test_eval_dataset_schema():
    path = Path(__file__).resolve().parents[1] / "evaluation" / "eval_dataset.json"
    with open(path, "r", encoding="utf-8") as fh:
        samples = json.load(fh)
    parsed = [EvalSample.model_validate(sample) for sample in samples]
    assert parsed, "Eval dataset should not be empty"
