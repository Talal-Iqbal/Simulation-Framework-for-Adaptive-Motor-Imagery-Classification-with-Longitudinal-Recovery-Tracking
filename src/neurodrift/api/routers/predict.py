"""Prediction endpoints: single trial JSON and batch .npz upload."""

from __future__ import annotations

import io
from collections import Counter

import numpy as np
from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile

from ...inference.engine import pseudo_online_engine
from ...observability.metrics import get_metrics
from ..deps import ModelBundle, get_model_bundle, get_registry
from ..schemas import (
    BatchPredictResponse,
    TrialPredictRequest,
    TrialPredictResponse,
)

router = APIRouter(prefix="/predict", tags=["predict"])


def _result_to_response(r) -> TrialPredictResponse:
    return TrialPredictResponse(
        trial_idx=r.trial_idx,
        y_true=r.y_true,
        y_pred=r.y_pred,
        correct=r.correct,
        confidence=r.confidence,
        margin=r.margin,
        accepted=r.accepted,
        reject_reasons=r.reject_reasons,
        timestamp_s=r.timestamp_s,
    )


def _require_models(bundle: ModelBundle, subject_id: int):
    if bundle.acceptance is None:
        raise HTTPException(status_code=503, detail="acceptance_model not loaded in registry")
    cal = bundle.get_calibration(subject_id, get_registry())
    if cal is None:
        raise HTTPException(
            status_code=404,
            detail=f"No calibration_subject_{subject_id} entry in registry. Run /calibrate/subject first.",
        )
    return cal


@router.post("/trial", response_model=TrialPredictResponse)
def predict_trial(
    payload: TrialPredictRequest,
    subject_id: int = Query(default=2, description="Calibrated subject id"),
    bundle: ModelBundle = Depends(get_model_bundle),
) -> TrialPredictResponse:
    epoch = np.asarray(payload.epoch, dtype=np.float64)
    if epoch.ndim != 2:
        raise HTTPException(status_code=400, detail="epoch must be 2D (n_channels, n_times)")

    cal = _require_models(bundle, subject_id)
    accept_threshold = payload.accept_threshold if payload.accept_threshold is not None else cal.cal_threshold

    X = epoch[np.newaxis, ...]
    y = np.array([payload.y_true]) if payload.y_true is not None else None

    results = pseudo_online_engine(
        cal.csp,
        cal.lda,
        bundle.rqe,
        bundle.acceptance.model,
        bundle.acceptance.scaler,
        X,
        y,
        accept_threshold=accept_threshold,
    )

    metrics = get_metrics()
    metrics.inc("neurodrift_predict_trial_total")
    if results[0].accepted:
        metrics.inc("neurodrift_predict_accepted_total")
    else:
        metrics.inc("neurodrift_predict_rejected_total")

    return _result_to_response(results[0])


@router.post("/upload", response_model=BatchPredictResponse)
async def predict_upload(
    file: UploadFile = File(..., description=".npz with arrays X (and optional y)"),
    subject_id: int = Query(default=2),
    accept_threshold: float | None = Query(default=None, ge=0.0, le=1.0),
    bundle: ModelBundle = Depends(get_model_bundle),
) -> BatchPredictResponse:
    raw = await file.read()
    try:
        arr = np.load(io.BytesIO(raw), allow_pickle=False)
        X = np.asarray(arr["X"], dtype=np.float64)
        y = np.asarray(arr["y"]).astype(str) if "y" in arr.files else None
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid .npz: {exc!r}")

    if X.ndim != 3:
        raise HTTPException(status_code=400, detail="X must be 3D (n_trials, n_channels, n_times)")

    cal = _require_models(bundle, subject_id)
    threshold = accept_threshold if accept_threshold is not None else cal.cal_threshold

    results = pseudo_online_engine(
        cal.csp,
        cal.lda,
        bundle.rqe,
        bundle.acceptance.model,
        bundle.acceptance.scaler,
        X,
        y,
        accept_threshold=threshold,
    )

    accepted = sum(1 for r in results if r.accepted)
    rejected = len(results) - accepted

    accuracy_all = accuracy_acc = None
    if y is not None and len(y) > 0:
        accuracy_all = float(np.mean([r.correct for r in results]))
        if accepted > 0:
            accuracy_acc = float(np.mean([r.correct for r in results if r.accepted]))

    breakdown: Counter = Counter()
    for r in results:
        for reason in r.reject_reasons:
            base = reason.split("(")[0]
            breakdown[base] += 1

    metrics = get_metrics()
    metrics.inc("neurodrift_predict_batch_total")
    metrics.inc("neurodrift_predict_trial_total", value=float(len(results)))
    metrics.inc("neurodrift_predict_accepted_total", value=float(accepted))
    metrics.inc("neurodrift_predict_rejected_total", value=float(rejected))

    return BatchPredictResponse(
        n_trials=len(results),
        accepted=accepted,
        rejected=rejected,
        accuracy_all=accuracy_all,
        accuracy_accepted=accuracy_acc,
        rejection_breakdown=dict(breakdown),
        results=[_result_to_response(r) for r in results],
    )
