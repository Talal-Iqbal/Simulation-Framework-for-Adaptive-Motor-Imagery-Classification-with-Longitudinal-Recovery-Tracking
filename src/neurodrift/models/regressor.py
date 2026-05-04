"""Layer 7 — Ridge regression on session-level feature vectors.

Predicts the latent `r` recovery factor (and individually the four physical
degradation parameters alpha, beta, gamma, delta_ms) from per-session
collapsed feature vectors. Refactored from `cell-build-panel`,
`cell-ridge-r`, and `cell-multioutput-ridge`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SESSION_FEATURE_NAMES: list[str] = [
    "erd_mu_c3_mean",
    "erd_mu_c4_mean",
    "erd_lat_mu_mean",
    "erd_lat_mu_abs_mean",
    "lda_conf_mean",
    "lda_margin_mean",
    "lda_margin_std",
    "mu_ratio_mean",
    "motor_relative_mean",
    "frob_dist_to_baseline",
    "intertrial_cov_var",
    "lda_session_acc",
    "p2p_max_mean",
]


def session_feature_vector(
    X_session: np.ndarray,
    y_session: np.ndarray,
    csp,
    lda,
    fe,
    baseline_cov: np.ndarray | None = None,
) -> tuple[dict[str, float], np.ndarray]:
    """Collapse a session of trials into one fixed-length feature vector.

    Mirrors the notebook's `session_feature_vector`. Adds three session-level
    quantities to the per-trial feature means: raw LDA accuracy, Frobenius
    drift from a reference covariance, and inter-trial covariance variance.
    """
    F = fe.extract_many(X_session)
    names = fe.feature_names

    def col(name: str) -> np.ndarray:
        return F[:, names.index(name)]

    X_task = X_session[..., fe.baseline_samples :]
    covs = np.array([np.cov(epoch) for epoch in X_task])
    mean_cov = covs.mean(axis=0)
    intertrial_cov_var = float(np.mean(np.var(covs, axis=0)))

    frob = 0.0 if baseline_cov is None else float(np.linalg.norm(mean_cov - baseline_cov, ord="fro"))

    y_pred = lda.predict(csp.transform(X_task))
    acc = float(np.mean(y_pred == y_session))

    feats = {
        "erd_mu_c3_mean": float(np.mean(col("erd_mu_c3"))),
        "erd_mu_c4_mean": float(np.mean(col("erd_mu_c4"))),
        "erd_lat_mu_mean": float(np.mean(col("erd_lat_mu"))),
        "erd_lat_mu_abs_mean": float(np.mean(np.abs(col("erd_lat_mu")))),
        "lda_conf_mean": float(np.mean(col("lda_proba_max"))),
        "lda_margin_mean": float(np.mean(col("lda_margin"))),
        "lda_margin_std": float(np.std(col("lda_margin"))),
        "mu_ratio_mean": float(np.mean(col("mu_ratio_motor"))),
        "motor_relative_mean": float(np.mean(col("motor_relative_power"))),
        "frob_dist_to_baseline": frob,
        "intertrial_cov_var": intertrial_cov_var,
        "lda_session_acc": acc,
        "p2p_max_mean": float(np.mean(col("peak_to_peak_max"))),
    }
    return feats, mean_cov


@dataclass
class RegressionArtifacts:
    """Standardised + Ridge pipeline for session r prediction."""

    pipeline: Pipeline
    feature_names: list[str]
    metrics: dict[str, float]
    pca: PCA | None = None


def train_session_r_regressor(
    panel: np.ndarray,
    r_values: np.ndarray,
    feature_names: list[str] | None = None,
    alpha: float = 1.0,
    use_pca: bool = False,
    pca_components: int = 5,
) -> RegressionArtifacts:
    """Train a Ridge regressor mapping session features -> latent r.

    Uses leave-one-out cross-validation (small panel sizes match the
    notebook's longitudinal experiments).
    """
    X = panel
    y = r_values
    feat_names = feature_names or SESSION_FEATURE_NAMES[: X.shape[1]]

    pca: PCA | None = None
    if use_pca and X.shape[1] > pca_components:
        pca = PCA(n_components=pca_components)
        X = pca.fit_transform(X)

    pipeline = Pipeline([("scale", StandardScaler()), ("ridge", Ridge(alpha=alpha))])
    pipeline.fit(X, y)

    loo = LeaveOneOut()
    preds = np.zeros_like(y, dtype=float)
    for tr, te in loo.split(X):
        pipeline_cv = Pipeline([("scale", StandardScaler()), ("ridge", Ridge(alpha=alpha))])
        pipeline_cv.fit(X[tr], y[tr])
        preds[te] = pipeline_cv.predict(X[te])

    r2_loo = float(r2_score(y, preds)) if len(y) > 1 else 0.0
    rmse_loo = float(np.sqrt(mean_squared_error(y, preds)))

    metrics = {
        "r2_loo": r2_loo,
        "rmse_loo": rmse_loo,
        "n_samples": int(len(y)),
        "n_features": int(X.shape[1]),
    }
    return RegressionArtifacts(pipeline=pipeline, feature_names=feat_names, metrics=metrics, pca=pca)
