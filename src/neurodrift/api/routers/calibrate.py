"""On-demand per-subject calibration endpoint."""

from __future__ import annotations

import numpy as np
from fastapi import APIRouter, Depends, HTTPException

from ...models.classifier import calibrate_subject
from ...registry.store import ModelRegistry, hash_array
from ..deps import ModelBundle, get_model_bundle, get_registry, reset_model_bundle
from ..schemas import CalibrateRequest, CalibrateResponse

router = APIRouter(prefix="/calibrate", tags=["calibrate"])


@router.post("/subject", response_model=CalibrateResponse)
def calibrate(
    payload: CalibrateRequest,
    bundle: ModelBundle = Depends(get_model_bundle),
    registry: ModelRegistry = Depends(get_registry),
) -> CalibrateResponse:
    if bundle.acceptance is None:
        raise HTTPException(
            status_code=503,
            detail="acceptance_model not loaded — run the training flow first",
        )

    X_cal = np.asarray(payload.X, dtype=np.float64)
    y_cal = np.asarray(payload.y, dtype=str)
    if X_cal.ndim != 3 or len(X_cal) != len(y_cal):
        raise HTTPException(status_code=400, detail="X must be 3D and len(X) == len(y)")

    if payload.X_eval is not None and payload.y_eval is not None:
        X_eval = np.asarray(payload.X_eval, dtype=np.float64)
        y_eval = np.asarray(payload.y_eval, dtype=str)
    else:
        X_eval = np.empty((0, *X_cal.shape[1:]), dtype=np.float64)
        y_eval = np.empty((0,), dtype=str)

    try:
        cal_artifacts, _ = calibrate_subject(
            X_cal,
            y_cal,
            X_eval,
            y_eval,
            bundle.rqe,
            bundle.acceptance,
            subject_id=payload.subject_id,
            accept_percentile=payload.accept_percentile,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Calibration failed: {exc!r}")

    entry = registry.save(
        f"calibration_subject_{payload.subject_id}",
        cal_artifacts,
        metrics=cal_artifacts.metrics,
        meta={"cal_threshold": cal_artifacts.cal_threshold, "via": "api"},
        train_data_hash=hash_array(X_cal),
    )
    reset_model_bundle()

    return CalibrateResponse(
        subject_id=payload.subject_id,
        version=entry.version,
        cal_threshold=cal_artifacts.cal_threshold,
        n_cal_trials=cal_artifacts.n_cal_trials,
        metrics=cal_artifacts.metrics,
    )
