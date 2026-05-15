from __future__ import annotations

import argparse
import inspect
import json
import os
from pathlib import Path

import numpy as np
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from training.config import DEFAULT_CONFIG, TrainingConfig
from training.data import label_mappings, prepare_splits, tokenize_splits
from training.metrics import classification_report_dict, compute_basic_metrics
from training.trainer_compat import trainer_tokenizer_kwargs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune DistilBERT for support intent detection."
    )
    parser.add_argument("--max-samples", type=int, default=None, help="Limit rows for smoke tests.")
    parser.add_argument(
        "--model-id",
        default=os.getenv("HF_MODEL_ID"),
        help="Optional HF Hub repo id.",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push trained artifacts to HF Hub.",
    )
    parser.add_argument(
        "--no-push",
        action="store_true",
        help="Disable Hub push even if model id is set.",
    )
    parser.add_argument("--epochs", type=float, default=DEFAULT_CONFIG.num_train_epochs)
    parser.add_argument("--train-batch-size", type=int, default=DEFAULT_CONFIG.train_batch_size)
    parser.add_argument("--eval-batch-size", type=int, default=DEFAULT_CONFIG.eval_batch_size)
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> TrainingConfig:
    return TrainingConfig(
        num_train_epochs=args.epochs,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
    )


def should_push(args: argparse.Namespace) -> bool:
    return bool(args.push_to_hub and args.model_id and not args.no_push)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def training_arguments_kwargs(
    config: TrainingConfig, args: argparse.Namespace
) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "output_dir": str(config.output_dir),
        "learning_rate": config.learning_rate,
        "per_device_train_batch_size": config.train_batch_size,
        "per_device_eval_batch_size": config.eval_batch_size,
        "num_train_epochs": config.num_train_epochs,
        "weight_decay": config.weight_decay,
        "warmup_ratio": config.warmup_ratio,
        "save_strategy": "epoch",
        "load_best_model_at_end": True,
        "metric_for_best_model": config.metric_for_best_model,
        "greater_is_better": True,
        "logging_dir": "runs",
        "logging_strategy": "steps",
        "logging_steps": 50,
        "report_to": ["tensorboard"],
        "push_to_hub": should_push(args),
        "hub_model_id": args.model_id if should_push(args) else None,
        "seed": config.seed,
    }
    signature = inspect.signature(TrainingArguments)
    eval_key = "eval_strategy" if "eval_strategy" in signature.parameters else "evaluation_strategy"
    kwargs[eval_key] = "epoch"
    return kwargs


def train() -> dict[str, float]:
    args = parse_args()
    config = build_config(args)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.final_model_dir.mkdir(parents=True, exist_ok=True)
    config.report_dir.mkdir(parents=True, exist_ok=True)

    splits = prepare_splits(config, max_samples=args.max_samples)
    id2label, label2id = label_mappings(splits)

    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name)
    tokenized = tokenize_splits(splits, tokenizer, config)

    model = AutoModelForSequenceClassification.from_pretrained(
        config.base_model_name,
        num_labels=len(id2label),
        id2label=id2label,
        label2id=label2id,
    )

    training_args = TrainingArguments(**training_arguments_kwargs(config, args))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        compute_metrics=compute_basic_metrics,
        **trainer_tokenizer_kwargs(tokenizer),
    )

    trainer.train()
    validation_metrics = trainer.evaluate(tokenized["validation"])
    test_output = trainer.predict(tokenized["test"])
    test_predictions = np.argmax(test_output.predictions, axis=-1)

    label_names = [id2label[index] for index in range(len(id2label))]
    test_report = classification_report_dict(
        labels=test_output.label_ids.tolist(),
        predictions=test_predictions.tolist(),
        target_names=label_names,
    )
    test_metrics = {
        "test_accuracy": float(test_report["accuracy"]),
        "test_f1_macro": float(test_report["macro avg"]["f1-score"]),
        "test_f1_weighted": float(test_report["weighted avg"]["f1-score"]),
    }

    trainer.save_model(str(config.final_model_dir))
    tokenizer.save_pretrained(str(config.final_model_dir))
    write_json(config.report_dir / "validation_metrics.json", validation_metrics)
    write_json(config.report_dir / "test_metrics.json", test_metrics)
    write_json(config.report_dir / "classification_report.json", test_report)
    write_json(config.report_dir / "labels.json", label_names)

    if should_push(args):
        trainer.push_to_hub()

    return test_metrics


def main() -> None:
    metrics = train()
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
