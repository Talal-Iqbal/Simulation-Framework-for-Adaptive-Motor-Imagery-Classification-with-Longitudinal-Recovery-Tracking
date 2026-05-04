"""DeepChecks-driven data integrity smoke tests."""

from __future__ import annotations

import numpy as np
import pytest


def test_data_integrity_passes_clean(acceptance_dataset, rqe):
    from neurodrift.testing.data_validation import run_data_integrity_suite

    summary = run_data_integrity_suite(
        acceptance_dataset.F_all, acceptance_dataset.Y_all, rqe.feature_names
    )
    assert summary["passed"] is True
    assert summary["nulls"] == 0
    assert summary["inf_count"] == 0


def test_data_integrity_fails_with_single_class(rqe):
    from neurodrift.testing.data_validation import run_data_integrity_suite

    F = np.random.RandomState(0).randn(50, len(rqe.feature_names))
    Y = np.zeros(50)
    with pytest.raises(ValueError):
        run_data_integrity_suite(F, Y, rqe.feature_names)


def test_data_integrity_fails_with_nan(rqe):
    from neurodrift.testing.data_validation import run_data_integrity_suite

    F = np.random.RandomState(0).randn(50, len(rqe.feature_names))
    F[0, 0] = np.nan
    Y = np.r_[np.zeros(25), np.ones(25)]
    with pytest.raises(ValueError):
        run_data_integrity_suite(F, Y, rqe.feature_names)
