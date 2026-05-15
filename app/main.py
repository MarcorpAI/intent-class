from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request, status

from app.predictor import DEFAULT_MODEL_ID, IntentPredictor
from app.schemas import (
    BatchPrediction,
    HealthResponse,
    PredictBatchRequest,
    PredictBatchResponse,
    PredictRequest,
    PredictResponse,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        app.state.predictor = IntentPredictor.from_env()
        app.state.model_error = None
    except Exception as exc:  # noqa: BLE001 - health should report startup model failures.
        app.state.predictor = None
        app.state.model_error = str(exc)
    yield


app = FastAPI(
    title="Customer Support Intent Classifier",
    version="0.1.0",
    lifespan=lifespan,
)


def get_predictor(request: Request) -> IntentPredictor:
    predictor = getattr(request.app.state, "predictor", None)
    if predictor is None:
        detail = getattr(request.app.state, "model_error", None) or "Model is not loaded."
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=detail)
    return predictor


@app.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    predictor = getattr(request.app.state, "predictor", None)
    error = getattr(request.app.state, "model_error", None)
    if predictor is None:
        return HealthResponse(
            status="degraded",
            model_id=DEFAULT_MODEL_ID,
            labels=[],
            loaded=False,
            detail=error,
        )
    return HealthResponse(
        status="ok",
        model_id=predictor.model_id,
        labels=predictor.labels,
        loaded=True,
    )


@app.post("/predict", response_model=PredictResponse)
async def predict(payload: PredictRequest, request: Request) -> PredictResponse:
    prediction = get_predictor(request).predict(payload.text)
    return PredictResponse(
        intent=prediction.intent,
        confidence=prediction.confidence,
        all_scores=prediction.all_scores,
    )


@app.post("/predict/batch", response_model=PredictBatchResponse)
async def predict_batch(payload: PredictBatchRequest, request: Request) -> PredictBatchResponse:
    predictions = get_predictor(request).predict_batch(payload.texts)
    return PredictBatchResponse(
        predictions=[
            BatchPrediction(
                text=prediction.text,
                intent=prediction.intent,
                confidence=prediction.confidence,
            )
            for prediction in predictions
        ]
    )


@app.get("/", include_in_schema=False)
async def root() -> dict[str, Any]:
    return {"service": "intent-classifier", "health": "/health"}

