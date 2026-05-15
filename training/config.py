from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrainingConfig:
    dataset_name: str = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"
    base_model_name: str = "distilbert-base-uncased"
    output_dir: Path = Path("artifacts/checkpoints")
    final_model_dir: Path = Path("artifacts/intent-model")
    report_dir: Path = Path("reports")
    test_size: float = 0.10
    validation_size: float = 0.10
    seed: int = 42
    max_length: int = 128
    num_train_epochs: float = 4.0
    train_batch_size: int = 32
    eval_batch_size: int = 64
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.10
    metric_for_best_model: str = "f1_macro"


DEFAULT_CONFIG = TrainingConfig()

