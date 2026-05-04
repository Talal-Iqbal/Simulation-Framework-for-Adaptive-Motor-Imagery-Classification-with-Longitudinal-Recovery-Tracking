"""Session-level analysis: cluster id + predicted r."""

from __future__ import annotations

import numpy as np
from fastapi import APIRouter, Depends, HTTPException

from ...models.regressor import SESSION_FEATURE_NAMES
from ..deps import ModelBundle, get_model_bundle
from ..schemas import SessionAnalyzeRequest, SessionAnalyzeResponse

router = APIRouter(prefix="/analyze", tags=["analyze"])


@router.post("/session", response_model=SessionAnalyzeResponse)
def analyze_session(
    payload: SessionAnalyzeRequest,
    bundle: ModelBundle = Depends(get_model_bundle),
) -> SessionAnalyzeResponse:
    if bundle.regressor is None or bundle.kmeans is None:
        raise HTTPException(
            status_code=503,
            detail="Layer-7 regressor / kmeans not loaded — run the training flow first",
        )

    missing = [k for k in SESSION_FEATURE_NAMES if k not in payload.features]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing session features: {missing}")

    vec = np.array(
        [[payload.features[k] for k in SESSION_FEATURE_NAMES]], dtype=np.float64
    )

    pred_r = float(bundle.regressor.pipeline.predict(vec)[0])

    Xs = bundle.kmeans.scaler.transform(vec)
    cluster = int(bundle.kmeans.kmeans.predict(Xs)[0])

    silhouette = bundle.kmeans.metrics.get("silhouette")
    if silhouette is not None and (np.isnan(silhouette) or np.isinf(silhouette)):
        silhouette = None

    return SessionAnalyzeResponse(
        cluster=cluster,
        predicted_r=pred_r,
        silhouette=silhouette,
    )
