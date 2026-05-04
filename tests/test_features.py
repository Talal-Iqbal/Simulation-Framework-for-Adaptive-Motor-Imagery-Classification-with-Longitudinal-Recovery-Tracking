"""Feature extractor smoke tests."""

from __future__ import annotations

import numpy as np


def test_raw_quality_extractor_shape(synthetic_epochs, rqe):
    X, _ = synthetic_epochs
    F = rqe.extract_many(X)
    assert F.shape == (len(X), 8)
    assert F.dtype == np.float64


def test_raw_quality_extractor_feature_names(rqe):
    expected = {
        "peak_to_peak_max",
        "kurtosis_max",
        "baseline_p2p_max",
        "erd_mu_c3",
        "erd_mu_c4",
        "erd_lat_mu",
        "mu_ratio_motor",
        "motor_relative_power",
    }
    assert set(rqe.feature_names) == expected


def test_no_nan_in_features(synthetic_epochs, rqe):
    X, _ = synthetic_epochs
    F = rqe.extract_many(X)
    assert not np.isnan(F).any()
    assert not np.isinf(F).any()
