"""Pydantic v2 schemas for FastAPI request / response payloads."""

from __future__ import annotations

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    version: str
    models: dict[str, str | None]


class TrialPredictRequest(BaseModel):
    epoch: list[list[float]] = Field(
        ...,
        description="EEG epoch as 2D array shape (n_channels, n_times). "
        "Default 22 channels x 1501 samples for BNCI2014_001.",
    )
    y_true: str | None = Field(default=None, description="Optional ground-truth label")
    accept_threshold: float | None = Field(default=None, ge=0.0, le=1.0)


class TrialPredictResponse(BaseModel):
    trial_idx: int
    y_true: str
    y_pred: str
    correct: bool
    confidence: float
    margin: float
    accepted: bool
    reject_reasons: list[str]
    timestamp_s: float


class BatchPredictResponse(BaseModel):
    n_trials: int
    accepted: int
    rejected: int
    accuracy_all: float | None = None
    accuracy_accepted: float | None = None
    rejection_breakdown: dict[str, int]
    results: list[TrialPredictResponse]


class CalibrateRequest(BaseModel):
    subject_id: int
    X: list[list[list[float]]] = Field(..., description="Calibration epochs (n_trials, n_ch, n_t)")
    y: list[str]
    X_eval: list[list[list[float]]] | None = None
    y_eval: list[str] | None = None
    accept_percentile: float = 30.0


class CalibrateResponse(BaseModel):
    subject_id: int
    version: str
    cal_threshold: float
    n_cal_trials: int
    metrics: dict[str, float]


class SessionAnalyzeRequest(BaseModel):
    features: dict[str, float]


class SessionAnalyzeResponse(BaseModel):
    cluster: int
    predicted_r: float
    silhouette: float | None = None


class EvalSessionStartResponse(BaseModel):
    session_id: str
    subject_id: int
    n_trials: int


class EvalTrialResponse(TrialPredictResponse):
    cursor: int = Field(..., description="Zero-based index of the trial just served")
    exhausted: bool = Field(..., description="True when this was the last trial in the eval set")
