from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer

from training.config import DEFAULT_CONFIG
from training.data import label_mappings, prepare_splits, tokenize_splits
from training.metrics import classification_report_dict, compute_basic_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained intent classifier.")
    parser.add_argument("--model-path", default=str(DEFAULT_CONFIG.final_model_dir))
    parser.add_argument("--max-samples", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = DEFAULT_CONFIG
    model_path = Path(args.model_path)
    report_dir = config.report_dir
    report_dir.mkdir(parents=True, exist_ok=True)

    splits = prepare_splits(config, max_samples=args.max_samples)
    id2label, _ = label_mappings(splits)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenized = tokenize_splits(splits, tokenizer, config)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    trainer = Trainer(model=model, tokenizer=tokenizer, compute_metrics=compute_basic_metrics)
    output = trainer.predict(tokenized["test"])
    predictions = np.argmax(output.predictions, axis=-1)
    label_names = [id2label[index] for index in range(len(id2label))]
    report = classification_report_dict(
        labels=output.label_ids.tolist(),
        predictions=predictions.tolist(),
        target_names=label_names,
    )
    metrics = {
        "test_accuracy": float(report["accuracy"]),
        "test_f1_macro": float(report["macro avg"]["f1-score"]),
        "test_f1_weighted": float(report["weighted avg"]["f1-score"]),
    }

    (report_dir / "evaluation_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (report_dir / "evaluation_classification_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

