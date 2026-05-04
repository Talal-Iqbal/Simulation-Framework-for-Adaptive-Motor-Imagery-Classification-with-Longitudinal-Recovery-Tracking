"""DeepChecks model evaluation gate."""

from __future__ import annotations

import numpy as np
import pytest


def test_model_evaluation_runs(acceptance_dataset, trained_acceptance, rqe):
    from neurodrift.testing.model_validation import run_model_evaluation_suite

    summary = run_model_evaluation_suite(
        acceptance_dataset.F_all,
        acceptance_dataset.Y_all,
        trained_acceptance,
        rqe.feature_names,
        min_auc=0.0,
        min_accuracy=0.0,
    )
    assert summary["passed"] is True
    assert "auc" in summary
    assert "accuracy" in summary


def test_model_evaluation_fails_high_threshold(
    acceptance_dataset, trained_acceptance, rqe
):
    from neurodrift.testing.model_validation import run_model_evaluation_suite

    with pytest.raises(ValueError):
        run_model_evaluation_suite(
            acceptance_dataset.F_all,
            acceptance_dataset.Y_all,
            trained_acceptance,
            rqe.feature_names,
            min_auc=0.99,
            min_accuracy=0.99,
        )


def test_drift_detection_no_change():
    from neurodrift.testing.drift import detect_drift

    rng = np.random.RandomState(0)
    X = rng.randn(200, 4)
    feature_names = [f"f{i}" for i in range(4)]
    report = detect_drift(X, X, feature_names)
    assert report["any_drift"] is False
    assert len(report["drifted_features"]) == 0


def test_drift_detection_shifted():
    from neurodrift.testing.drift import detect_drift

    rng = np.random.RandomState(0)
    ref = rng.randn(200, 4)
    cur = rng.randn(200, 4) + 5.0
    feature_names = [f"f{i}" for i in range(4)]
    report = detect_drift(ref, cur, feature_names)
    assert report["any_drift"] is True
