from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from datasets import ClassLabel, Dataset, DatasetDict, load_dataset
from transformers import PreTrainedTokenizerBase

from training.config import TrainingConfig

TEXT_COLUMN_CANDIDATES = ("instruction", "text", "message", "utterance", "customer_message")
LABEL_COLUMN_CANDIDATES = ("intent", "label", "category")


def load_raw_dataset(config: TrainingConfig) -> Dataset:
    raw = load_dataset(config.dataset_name)
    if isinstance(raw, DatasetDict):
        if "train" in raw:
            return raw["train"]
        first_split = next(iter(raw.keys()))
        return raw[first_split]
    return raw


def _first_existing_column(dataset: Dataset, candidates: tuple[str, ...]) -> str:
    for column in candidates:
        if column in dataset.column_names:
            return column
    raise ValueError(
        f"None of the expected columns {candidates!r} exist. Found: {dataset.column_names!r}"
    )


def infer_columns(dataset: Dataset) -> tuple[str, str]:
    return (
        _first_existing_column(dataset, TEXT_COLUMN_CANDIDATES),
        _first_existing_column(dataset, LABEL_COLUMN_CANDIDATES),
    )


def prepare_splits(config: TrainingConfig, max_samples: int | None = None) -> DatasetDict:
    dataset = load_raw_dataset(config)
    text_column, label_column = infer_columns(dataset)

    if max_samples is not None:
        dataset = dataset.shuffle(seed=config.seed).select(range(min(max_samples, len(dataset))))

    if not isinstance(dataset.features[label_column], ClassLabel):
        dataset = dataset.class_encode_column(label_column)

    stratify_column = label_column if max_samples is None else None
    train_plus_validation = dataset.train_test_split(
        test_size=config.test_size,
        seed=config.seed,
        stratify_by_column=stratify_column,
    )
    validation_fraction = config.validation_size / (1.0 - config.test_size)
    train_validation = train_plus_validation["train"].train_test_split(
        test_size=validation_fraction,
        seed=config.seed,
        stratify_by_column=stratify_column,
    )

    splits = DatasetDict(
        {
            "train": train_validation["train"],
            "validation": train_validation["test"],
            "test": train_plus_validation["test"],
        }
    )
    return splits.rename_columns({text_column: "text", label_column: "labels"})


def label_mappings(splits: DatasetDict) -> tuple[dict[int, str], dict[str, int]]:
    label_feature = splits["train"].features["labels"]
    names = label_feature.names
    id2label = {index: label for index, label in enumerate(names)}
    label2id = {label: index for index, label in id2label.items()}
    return id2label, label2id


def tokenize_splits(
    splits: DatasetDict,
    tokenizer: PreTrainedTokenizerBase,
    config: TrainingConfig,
) -> DatasetDict:
    keep_columns = {"text", "labels"}
    remove_columns = [
        column for column in splits["train"].column_names if column not in keep_columns
    ]

    def tokenize_batch(batch: Mapping[str, list[Any]]) -> dict[str, Any]:
        return tokenizer(
            batch["text"],
            max_length=config.max_length,
            padding="max_length",
            truncation=True,
        )

    tokenized = splits.map(tokenize_batch, batched=True, remove_columns=remove_columns)
    return tokenized.with_format("torch")
