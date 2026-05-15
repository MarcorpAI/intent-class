from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, f1_score


def compute_basic_metrics(eval_pred: Any) -> dict[str, float]:
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "f1_macro": float(f1_score(labels, predictions, average="macro")),
        "f1_weighted": float(f1_score(labels, predictions, average="weighted")),
    }


def classification_report_dict(
    labels: list[int],
    predictions: list[int],
    target_names: list[str],
) -> dict[str, Any]:
    return classification_report(
        labels,
        predictions,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )

