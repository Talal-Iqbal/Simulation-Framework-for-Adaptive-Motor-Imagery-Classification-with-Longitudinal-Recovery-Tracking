"""Liveness + loaded model versions endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from ... import __version__
from ..deps import ModelBundle, get_model_bundle
from ..schemas import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health(bundle: ModelBundle = Depends(get_model_bundle)) -> HealthResponse:
    cal_versions = {f"calibration_subject_{sid}": ver for sid, ver in bundle.calibration_version.items()}
    return HealthResponse(
        status="ok",
        version=__version__,
        models={
            "acceptance_model": bundle.acceptance_version,
            "session_r_regressor": bundle.regressor_version,
            "session_kmeans": bundle.kmeans_version,
            **cal_versions,
        },
    )
