"""Train-vs-production feature drift detection.

Implements Population Stability Index (PSI) and Kolmogorov-Smirnov per-feature
drift detection. Designed to be called from a Prefect task to compare the
current training feature distribution against the most recent registry
manifest's reference distribution.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats as sp_stats


def compute_psi(
    expected: np.ndarray,
    actual: np.ndarray,
    n_bins: int = 10,
    eps: float = 1e-6,
) -> float:
    """Population Stability Index between two 1-D distributions."""
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    if len(expected) == 0 or len(actual) == 0:
        return 0.0

    breakpoints = np.linspace(
        min(expected.min(), actual.min()),
        max(expected.max(), actual.max()),
        n_bins + 1,
    )
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf

    exp_hist, _ = np.histogram(expected, bins=breakpoints)
    act_hist, _ = np.histogram(actual, bins=breakpoints)

    exp_pct = exp_hist / max(exp_hist.sum(), 1)
    act_pct = act_hist / max(act_hist.sum(), 1)

    exp_pct = np.where(exp_pct == 0, eps, exp_pct)
    act_pct = np.where(act_pct == 0, eps, act_pct)

    psi = float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))
    return psi


def detect_drift(
    reference: np.ndarray,
    current: np.ndarray,
    feature_names: list[str],
    psi_threshold: float = 0.2,
    ks_p_threshold: float = 0.01,
) -> dict[str, Any]:
    """Per-feature PSI + KS drift report.

    PSI: > 0.2 = significant shift (industry rule of thumb).
    KS:  p < 0.01 = distributions differ.
    """
    if reference.shape[1] != current.shape[1]:
        raise ValueError(
            f"Feature dimension mismatch: ref={reference.shape[1]} vs cur={current.shape[1]}"
        )

    per_feature: dict[str, dict[str, float]] = {}
    drifted: list[str] = []
    for j, name in enumerate(feature_names):
        psi = compute_psi(reference[:, j], current[:, j])
        ks_stat, ks_p = sp_stats.ks_2samp(reference[:, j], current[:, j])
        is_drifted = bool(psi > psi_threshold or ks_p < ks_p_threshold)
        per_feature[name] = {
            "psi": float(psi),
            "ks_statistic": float(ks_stat),
            "ks_pvalue": float(ks_p),
            "drifted": is_drifted,
        }
        if is_drifted:
            drifted.append(name)

    return {
        "drifted_features": drifted,
        "any_drift": len(drifted) > 0,
        "psi_threshold": psi_threshold,
        "ks_p_threshold": ks_p_threshold,
        "per_feature": per_feature,
    }
