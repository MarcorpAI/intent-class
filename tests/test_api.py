from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app
from app.predictor import IntentPrediction


class FakePredictor:
    model_id = "fake-model"
    labels = ["cancel_order", "track_order"]

    def predict(self, text: str) -> IntentPrediction:
        return IntentPrediction(
            text=text,
            intent="cancel_order",
            confidence=0.91,
            all_scores={"cancel_order": 0.91, "track_order": 0.09},
        )

    def predict_batch(self, texts: list[str]) -> list[IntentPrediction]:
        return [
            IntentPrediction(
                text=text,
                intent="track_order" if "where" in text.lower() else "cancel_order",
                confidence=0.88,
                all_scores={"cancel_order": 0.12, "track_order": 0.88},
            )
            for text in texts
        ]


def test_health_reports_loaded_model() -> None:
    with TestClient(app) as client:
        app.state.predictor = FakePredictor()
        app.state.model_error = None

        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "model_id": "fake-model",
        "labels": ["cancel_order", "track_order"],
        "loaded": True,
        "detail": None,
    }


def test_predict_returns_intent_scores() -> None:
    with TestClient(app) as client:
        app.state.predictor = FakePredictor()
        app.state.model_error = None

        response = client.post("/predict", json={"text": "Cancel my order"})

    assert response.status_code == 200
    assert response.json() == {
        "intent": "cancel_order",
        "confidence": 0.91,
        "all_scores": {"cancel_order": 0.91, "track_order": 0.09},
    }


def test_predict_batch_preserves_input_order() -> None:
    with TestClient(app) as client:
        app.state.predictor = FakePredictor()
        app.state.model_error = None

        response = client.post(
            "/predict/batch",
            json={"texts": ["Where is my order?", "Cancel this order"]},
        )

    assert response.status_code == 200
    assert response.json()["predictions"] == [
        {"text": "Where is my order?", "intent": "track_order", "confidence": 0.88},
        {"text": "Cancel this order", "intent": "cancel_order", "confidence": 0.88},
    ]


def test_predict_rejects_empty_text() -> None:
    with TestClient(app) as client:
        app.state.predictor = FakePredictor()
        app.state.model_error = None

        response = client.post("/predict", json={"text": ""})

    assert response.status_code == 422


def test_predict_returns_503_when_model_is_not_loaded() -> None:
    with TestClient(app) as client:
        app.state.predictor = None
        app.state.model_error = "model missing"

        response = client.post("/predict", json={"text": "Cancel my order"})

    assert response.status_code == 503
    assert response.json()["detail"] == "model missing"

