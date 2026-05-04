"""Pseudo-online engine: offline/online equivalence (notebook invariant)."""

from __future__ import annotations

import numpy as np

from neurodrift.inference.engine import pseudo_online_engine
from neurodrift.models.classifier import task_view


def test_engine_runs(synthetic_epochs, rqe, trained_acceptance, calibrated_subject):
    X, y = synthetic_epochs
    results = pseudo_online_engine(
        calibrated_subject.csp,
        calibrated_subject.lda,
        rqe,
        trained_acceptance.model,
        trained_acceptance.scaler,
        X,
        y,
        accept_threshold=calibrated_subject.cal_threshold,
    )
    assert len(results) == len(X)
    for r in results:
        assert 0.0 <= r.confidence <= 1.0
        assert isinstance(r.accepted, (bool, np.bool_))


def test_engine_offline_online_equivalence(
    synthetic_epochs, rqe, trained_acceptance, calibrated_subject
):
    """Notebook invariant: per-trial loop predictions must match a batch pass."""
    X, y = synthetic_epochs

    online = pseudo_online_engine(
        calibrated_subject.csp,
        calibrated_subject.lda,
        rqe,
        trained_acceptance.model,
        trained_acceptance.scaler,
        X,
        y,
        accept_threshold=calibrated_subject.cal_threshold,
    )

    Xc = calibrated_subject.csp.transform(task_view(X))
    batch_pred = calibrated_subject.lda.predict(Xc)

    online_pred = np.array([r.y_pred for r in online])
    assert (online_pred == batch_pred.astype(str)).all()
