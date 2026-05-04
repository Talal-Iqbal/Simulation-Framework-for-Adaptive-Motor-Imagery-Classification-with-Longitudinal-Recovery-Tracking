"""Stage 2 — Per-subject CSP+LDA calibration with adaptive stopping.

Refactored from `cell-gate-calibration`, `cell-adaptive-stop`, and
`cell-fit-csp-lda`. The acceptance model gates the calibration pool by the
subject-adaptive top (100 - percentile)% rule before fitting CSP+LDA.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import classification_report

from ..config import BASELINE_SAMPLES, CSP_N_COMPONENTS
from .acceptance import AcceptanceArtifacts


def task_view(epochs: np.ndarray) -> np.ndarray:
    """Slice the imagery-only window. CSP/LDA must not see the rest baseline."""
    return epochs[..., BASELINE_SAMPLES:]


def adaptive_calibration_stop(
    X_pool: np.ndarray,
    y_pool: np.ndarray,
    min_trials: int = 48,
    step: int = 12,
    max_trials: int = 132,
    patience: int = 2,
    eps_acc: float = 0.005,
    eps_conf: float = 0.005,
) -> tuple[int, list[dict[str, float]]]:
    """Decide how many accepted trials to use for CSP+LDA fitting.

    Pure port of the notebook's `_adaptive_calibration_stop`.
    """
    n = len(X_pool)
    if n < min_trials + step:
        return min(min_trials, n), []

    history: list[dict[str, float]] = []
    stable = 0
    cut = min_trials

    for n_cal in range(min_trials, min(max_trials, n - step) + 1, step):
        X_cal, y_cal = X_pool[:n_cal], y_pool[:n_cal]
        X_val, y_val = X_pool[n_cal:], y_pool[n_cal:]
        if len(X_val) < 12:
            break

        csp_tmp = CSP(n_components=6, reg=None, log=True, norm_trace=False)
        Xc_cal = csp_tmp.fit_transform(task_view(X_cal), y_cal)
        ld_tmp = LDA()
        ld_tmp.fit(Xc_cal, y_cal)

        Xc_val = csp_tmp.transform(task_view(X_val))
        acc_val = float(np.mean(ld_tmp.predict(Xc_val) == y_val))
        conf_val = float(np.mean(np.max(ld_tmp.predict_proba(Xc_val), axis=1)))

        history.append(
            {"n_cal": int(n_cal), "n_val": int(len(X_val)), "val_acc": acc_val, "val_conf": conf_val}
        )

        if len(history) >= 2:
            da = history[-1]["val_acc"] - history[-2]["val_acc"]
            dc = history[-1]["val_conf"] - history[-2]["val_conf"]
            if abs(da) < eps_acc and abs(dc) < eps_conf:
                stable += 1
            else:
                stable = 0
            if stable >= patience:
                cut = int(n_cal)
                break
        cut = int(n_cal)

    return cut, history


@dataclass
class CalibrationArtifacts:
    """Per-subject CSP + LDA bundle plus calibration metadata."""

    csp: CSP
    lda: LDA
    subject_id: int
    cal_threshold: float
    n_cal_trials: int
    metrics: dict[str, float]
    history: list[dict[str, float]] = field(default_factory=list)


def gate_calibration_pool(
    X_pool: np.ndarray,
    y_pool: np.ndarray,
    rqe,
    accept_artifacts: AcceptanceArtifacts,
    accept_percentile: float = 30.0,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Apply the frozen acceptance gate to a calibration pool.

    Strategy: keep the top (100 - percentile)% of trials ranked by P(accept).
    Subject-adaptive — even a hard subject always provides enough trials.
    """
    F = rqe.extract_many(X_pool)
    F_s = accept_artifacts.scaler.transform(F)
    p_acc = accept_artifacts.model.predict_proba(F_s)[:, 1]
    threshold = float(np.percentile(p_acc, accept_percentile))
    mask = p_acc >= threshold
    return X_pool[mask], y_pool[mask], threshold


def calibrate_subject(
    X_cal_pool: np.ndarray,
    y_cal_pool: np.ndarray,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
    rqe,
    accept_artifacts: AcceptanceArtifacts,
    subject_id: int,
    accept_percentile: float = 30.0,
    min_per_class: int = 10,
    n_components: int = CSP_N_COMPONENTS,
) -> tuple[CalibrationArtifacts, dict[str, object]]:
    """Run the full calibration flow for one subject."""
    X_acc, y_acc, threshold = gate_calibration_pool(
        X_cal_pool, y_cal_pool, rqe, accept_artifacts, accept_percentile=accept_percentile
    )

    classes_acc, counts_acc = np.unique(y_acc, return_counts=True)
    used_adaptive = False

    if len(counts_acc) >= 2 and min(counts_acc) >= min_per_class and len(X_acc) >= 60:
        cut, history = adaptive_calibration_stop(X_acc, y_acc)
        X_train = X_acc[:cut]
        y_train = y_acc[:cut]
        used_adaptive = True
    else:
        X_train = X_acc
        y_train = y_acc
        cut = len(X_train)
        history = []

    csp = CSP(n_components=n_components, reg=None, log=True, norm_trace=False)
    Xc_train = csp.fit_transform(task_view(X_train), y_train)
    lda = LDA()
    lda.fit(Xc_train, y_train)

    in_sample_acc = float(np.mean(lda.predict(Xc_train) == y_train))

    eval_metrics: dict[str, float] = {}
    diagnostics: dict[str, object] = {
        "used_adaptive_stop": used_adaptive,
        "history": history,
        "accepted_class_counts": {str(c): int(n) for c, n in zip(classes_acc, counts_acc)},
        "n_cal_pool": int(len(X_cal_pool)),
        "n_accepted": int(len(X_acc)),
        "cal_threshold": float(threshold),
    }

    if X_eval is not None and y_eval is not None and len(X_eval) > 0:
        Xc_eval = csp.transform(task_view(X_eval))
        y_pred_eval = lda.predict(Xc_eval)
        eval_acc = float(np.mean(y_pred_eval == y_eval))
        eval_metrics = {"eval_accuracy": eval_acc}
        diagnostics["classification_report"] = classification_report(y_eval, y_pred_eval, output_dict=True)

    metrics = {
        "in_sample_accuracy": in_sample_acc,
        "n_train": int(len(X_train)),
        **eval_metrics,
    }

    artifacts = CalibrationArtifacts(
        csp=csp,
        lda=lda,
        subject_id=subject_id,
        cal_threshold=float(threshold),
        n_cal_trials=int(cut),
        metrics=metrics,
        history=history,
    )

    return artifacts, diagnostics
