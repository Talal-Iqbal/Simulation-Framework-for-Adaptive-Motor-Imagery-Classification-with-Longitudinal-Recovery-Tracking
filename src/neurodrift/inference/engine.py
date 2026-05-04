"""Pseudo-online trial-by-trial inference engine.

Refactored from `cell-engine-def` in the notebook. Uses the FROZEN GLOBAL
acceptance model (rqe + accept_scaler + accept_model) as the gate. Only
accepted trials are classified by the per-subject CSP+LDA. Rejection
reasons are derived from raw quality features only.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field

import numpy as np

from ..config import CLASSIFICATION_DELAY_S
from ..models.classifier import task_view


@dataclass
class TrialResult:
    """Per-trial outcome record."""

    trial_idx: int
    y_true: str
    y_pred: str
    correct: bool
    confidence: float
    margin: float
    accepted: bool
    reject_reasons: list[str] = field(default_factory=list)
    timestamp_s: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


def rejection_reasons(
    feat_vec: np.ndarray,
    feat_names: Sequence[str],
    p2p_thr: float = 150e-6,
    kurt_thr: float = 5.0,
    erd_lat_thr: float = 0.05,
    base_p2p_thr: float = 120e-6,
) -> list[str]:
    """Return rejection reason flags from an 8-d raw-quality feature vector."""
    idx = {n: i for i, n in enumerate(feat_names)}
    reasons: list[str] = []
    if feat_vec[idx["peak_to_peak_max"]] > p2p_thr:
        reasons.append(
            f"artifact_p2p({feat_vec[idx['peak_to_peak_max']] * 1e6:.0f}uV)"
        )
    if feat_vec[idx["kurtosis_max"]] > kurt_thr:
        reasons.append(f"artifact_kurt({feat_vec[idx['kurtosis_max']]:.1f})")
    if abs(feat_vec[idx["erd_lat_mu"]]) < erd_lat_thr:
        reasons.append(
            f"weak_lateralization({abs(feat_vec[idx['erd_lat_mu']]):.3f})"
        )
    if feat_vec[idx["baseline_p2p_max"]] > base_p2p_thr:
        reasons.append("noisy_baseline")
    return reasons


def pseudo_online_engine(
    csp,
    lda_model,
    raw_extractor,
    accept_model,
    accept_scaler,
    X: np.ndarray,
    y: np.ndarray | None = None,
    accept_threshold: float = 0.5,
    speed_factor: float = 0.0,
    verbose: bool = False,
) -> list[TrialResult]:
    """Causal trial-by-trial simulation matching the notebook implementation.

    With ``speed_factor=0.0`` the output exactly matches a batch forward pass
    on the same data (verified in the notebook's offline-equivalence check).
    """
    results: list[TrialResult] = []
    t_sim = 0.0

    for i in range(len(X)):
        epoch = X[i:i + 1]

        x_csp_i = csp.transform(task_view(epoch))
        pred_arr = lda_model.predict(x_csp_i)
        proba_arr = lda_model.predict_proba(x_csp_i)
        margin_val = float(abs(lda_model.decision_function(x_csp_i)[0]))

        y_pred_i = str(pred_arr[0])
        confidence_i = float(proba_arr[0].max())

        feat_raw = raw_extractor.extract(epoch[0])
        feat_s = accept_scaler.transform(feat_raw.reshape(1, -1))
        p_acc = float(accept_model.predict_proba(feat_s)[0, 1])
        accepted_i = p_acc >= accept_threshold

        reasons: list[str] = []
        if not accepted_i:
            reasons = rejection_reasons(feat_raw, raw_extractor.feature_names)

        t_sim += CLASSIFICATION_DELAY_S
        if speed_factor > 0:
            time.sleep(CLASSIFICATION_DELAY_S * speed_factor)

        y_true_i = str(y[i]) if y is not None else ""

        result = TrialResult(
            trial_idx=i,
            y_true=y_true_i,
            y_pred=y_pred_i,
            correct=(y_pred_i == y_true_i) if y is not None else False,
            confidence=confidence_i,
            margin=margin_val,
            accepted=accepted_i,
            reject_reasons=reasons,
            timestamp_s=t_sim,
        )
        results.append(result)

        if verbose and i % 20 == 0:
            gate_str = "ACCEPT" if accepted_i else "REJECT"
            print(
                f"  [{i:3d}] true={y_true_i:<12s} pred={y_pred_i:<12s} "
                f"conf={confidence_i:.2f} gate={gate_str}"
            )

    return results
