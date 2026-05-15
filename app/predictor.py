from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_MODEL_ID = "artifacts/intent-model"


@dataclass(frozen=True)
class IntentPrediction:
    text: str
    intent: str
    confidence: float
    all_scores: dict[str, float]


class IntentPredictor:
    def __init__(self, model_id: str | None = None) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.model_id = model_id or os.getenv("INTENT_MODEL_ID", DEFAULT_MODEL_ID)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_id)
        self.model.to(self.device)
        self.model.eval()
        self.labels = self._labels_from_model()

    def _labels_from_model(self) -> list[str]:
        id2label = getattr(self.model.config, "id2label", None) or {}
        if not id2label:
            return []
        return [id2label[index] for index in sorted(id2label)]

    @classmethod
    def from_env(cls) -> IntentPredictor:
        model_id = os.getenv("INTENT_MODEL_ID", DEFAULT_MODEL_ID)
        if model_id == DEFAULT_MODEL_ID and not Path(model_id).exists():
            raise FileNotFoundError(
                "No local model artifact found at artifacts/intent-model. "
                "Train the model or set INTENT_MODEL_ID to a HuggingFace model id."
            )
        return cls(model_id=model_id)

    def predict(self, text: str) -> IntentPrediction:
        return self.predict_batch([text])[0]

    def predict_batch(self, texts: list[str]) -> list[IntentPrediction]:
        encoded = self.tokenizer(
            texts,
            max_length=128,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        with self.torch.inference_mode():
            logits = self.model(**encoded).logits
            probabilities = self.torch.softmax(logits, dim=-1).detach().cpu()

        predictions: list[IntentPrediction] = []
        for text, scores in zip(texts, probabilities, strict=True):
            top_index = int(self.torch.argmax(scores).item())
            labels = self.labels or [str(index) for index in range(scores.shape[0])]
            all_scores = {
                label: round(float(score), 6)
                for label, score in zip(labels, scores.tolist(), strict=True)
            }
            predictions.append(
                IntentPrediction(
                    text=text,
                    intent=labels[top_index],
                    confidence=round(float(scores[top_index]), 6),
                    all_scores=all_scores,
                )
            )
        return predictions
