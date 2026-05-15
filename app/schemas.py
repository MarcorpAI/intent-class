from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field


NonEmptyText = Annotated[str, Field(min_length=1)]


class PredictRequest(BaseModel):
    text: NonEmptyText


class PredictBatchRequest(BaseModel):
    texts: Annotated[list[NonEmptyText], Field(min_length=1, max_length=128)]


class PredictResponse(BaseModel):
    intent: str
    confidence: float = Field(ge=0.0, le=1.0)
    all_scores: dict[str, float]


class BatchPrediction(BaseModel):
    text: str
    intent: str
    confidence: float = Field(ge=0.0, le=1.0)


class PredictBatchResponse(BaseModel):
    predictions: list[BatchPrediction]


class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    status: str
    model_id: str
    labels: list[str]
    loaded: bool
    detail: str | None = None

