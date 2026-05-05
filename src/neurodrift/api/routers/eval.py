"""Eval-stream endpoints: server-owned replay of real held-out MOABB trials.

Instead of the client sending random synthetic epochs to /predict/trial, the
client creates an eval session here and then pulls one real trial at a time.
The backend owns the eval data cursor so acceptance/rejection dynamics reflect
genuine held-out EEG signals.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Dict

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Path, Query

from ...data.ingest import load_held_out_subject
from ...inference.engine import pseudo_online_engine
from ...observability.metrics import get_metrics
from ..deps import ModelBundle, get_model_bundle, get_registry
from ..schemas import EvalSessionStartResponse, EvalTrialResponse

router = APIRouter(prefix="/eval", tags=["eval"])

# ---------------------------------------------------------------------------
# In-process session store (single-process dev; replace with Redis for prod)
# ---------------------------------------------------------------------------

@dataclass
class _EvalSession:
    subject_id: int
    X_eval: np.ndarray       # (144, n_ch, n_t)
    y_eval: np.ndarray       # (144,)
    cursor: int = 0
    trial_counter: int = 0   # monotonic index across the session


_sessions: Dict[str, _EvalSession] = {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _require_models(bundle: ModelBundle, subject_id: int):
    if bundle.acceptance is None:
        raise HTTPException(status_code=503, detail="acceptance_model not loaded in registry")
    cal = bundle.get_calibration(subject_id, get_registry())
    if cal is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No calibration_subject_{subject_id} in registry. "
                "Run /calibrate/subject first."
            ),
        )
    return cal


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/session/start", response_model=EvalSessionStartResponse)
def start_eval_session(
    subject_id: int = Query(default=2, description="Subject whose held-out eval trials to stream"),
    bundle: ModelBundle = Depends(get_model_bundle),
) -> EvalSessionStartResponse:
    """Load real held-out MOABB eval trials for *subject_id* and open a session.

    Returns a *session_id* that the client must pass to ``/eval/session/{id}/next``
    on every subsequent poll.
    """
    _require_models(bundle, subject_id)

    _, _, X_eval, y_eval = load_held_out_subject(subject_id)

    session_id = str(uuid.uuid4())
    _sessions[session_id] = _EvalSession(
        subject_id=subject_id,
        X_eval=X_eval,
        y_eval=y_eval,
    )

    return EvalSessionStartResponse(
        session_id=session_id,
        subject_id=subject_id,
        n_trials=int(X_eval.shape[0]),
    )


@router.get("/session/{session_id}/next", response_model=EvalTrialResponse)
def next_eval_trial(
    session_id: str = Path(..., description="Session ID from /eval/session/start"),
    bundle: ModelBundle = Depends(get_model_bundle),
) -> EvalTrialResponse:
    """Return the next real held-out trial result for the given eval session.

    Increments the internal cursor.  When all trials are exhausted the response
    has ``exhausted=true``; subsequent calls return 410 Gone.
    """
    session = _sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Eval session '{session_id}' not found.")

    if session.cursor >= len(session.X_eval):
        raise HTTPException(
            status_code=410,
            detail="Eval session exhausted. All held-out trials have been streamed.",
        )

    cal = _require_models(bundle, session.subject_id)

    idx = session.cursor
    X_single = session.X_eval[idx : idx + 1]          # (1, n_ch, n_t)
    y_single = session.y_eval[idx : idx + 1]

    results = pseudo_online_engine(
        cal.csp,
        cal.lda,
        bundle.rqe,
        bundle.acceptance.model,
        bundle.acceptance.scaler,
        X_single,
        y_single,
        accept_threshold=cal.cal_threshold,
    )

    session.cursor += 1
    session.trial_counter += 1
    exhausted = session.cursor >= len(session.X_eval)

    r = results[0]
    metrics = get_metrics()
    metrics.inc("neurodrift_predict_trial_total")
    if r.accepted:
        metrics.inc("neurodrift_predict_accepted_total")
    else:
        metrics.inc("neurodrift_predict_rejected_total")

    return EvalTrialResponse(
        trial_idx=session.trial_counter,
        y_true=r.y_true,
        y_pred=r.y_pred,
        correct=r.correct,
        confidence=r.confidence,
        margin=r.margin,
        accepted=r.accepted,
        reject_reasons=r.reject_reasons,
        timestamp_s=r.timestamp_s,
        cursor=idx,
        exhausted=exhausted,
    )
