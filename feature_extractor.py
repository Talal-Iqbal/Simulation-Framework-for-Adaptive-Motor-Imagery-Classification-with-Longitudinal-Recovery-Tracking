"""Feature extractor for the signal-acceptance model.

Single source of truth: the same function is called during training
(on clean + corrupted trials) and at deployment (per-trial in the
pseudo-engine loop). Keeping both paths on one implementation prevents
train/deploy skew.

This version is BASELINE-AWARE. Each input epoch is assumed to be the
concatenation of a pre-cue rest window and an imagery window along the
last axis, with the boundary at index `baseline_samples` (default 500
= 2 s at 250 Hz). The extractor splits internally and uses each window
for what it is good for: rest is the ERD reference and a pre-task
artifact flag; the task window drives CSP/LDA, kurtosis, p2p, and
power ratios.

The 15 features fall into four logical groups:

    classifier confidence       lda_margin, lda_proba_max
    CSP-space geometry          csp_logvar_0..3, mahal_to_predicted
    raw-signal artifact signals peak_to_peak_max (task), kurtosis_max (task),
                                baseline_p2p_max
    neurophysiological shape    erd_mu_c3, erd_mu_c4, erd_lat_mu,
                                mu_ratio_motor, motor_relative_power

Removed in this revision (low importance in the previous model):
    csp_logvar_2 -> kept (only marginal cost; preserves CSP basis as a unit)
    hjorth_mobility_motor -> dropped (≈ 0 weight, redundant with mu_ratio)
    lateralization_mu / lateralization_beta -> dropped (replaced by
        baseline-referenced ERD versions which strictly dominate them).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence

import numpy as np
from scipy.stats import kurtosis as _kurtosis


def _bandpower(x: np.ndarray, sfreq: float, fmin: float, fmax: float) -> float:
    """Mean spectral density in [fmin, fmax].

    Uses MEAN (not SUM) of bins so the result is comparable across
    windows of different length — important here because the rest
    window (2 s) and the task window (4 s) have different bin counts.
    """
    n = x.shape[-1]
    spec = np.fft.rfft(x, axis=-1)
    freqs = np.fft.rfftfreq(n, d=1.0 / sfreq)
    mask = (freqs >= fmin) & (freqs <= fmax)
    psd = (np.abs(spec) ** 2) / n
    return float(np.mean(psd[..., mask]))


@dataclass
class FeatureExtractor:
    """Extracts a fixed 15-d feature vector per (baseline + task) epoch."""

    csp: object
    lda: object
    ch_names: Sequence[str]
    sfreq: float = 250.0
    baseline_samples: int = 500
    feature_names: List[str] = field(default_factory=lambda: [
        "lda_margin",
        "lda_proba_max",
        "csp_logvar_0", "csp_logvar_1", "csp_logvar_2", "csp_logvar_3",
        "mahal_to_predicted",
        "peak_to_peak_max",
        "kurtosis_max",
        "erd_mu_c3",
        "erd_mu_c4",
        "erd_lat_mu",
        "mu_ratio_motor",
        "motor_relative_power",
        "baseline_p2p_max",
    ])

    def __post_init__(self):
        self.c3 = self.ch_names.index("C3")
        self.cz = self.ch_names.index("Cz")
        self.c4 = self.ch_names.index("C4")
        self.motor_ch = [self.c3, self.cz, self.c4]
        self._centroids = None
        self._inv_cov = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "FeatureExtractor":
        """X_train is the FULL (baseline+task) epoch tensor."""
        X_task = X_train[..., self.baseline_samples:]
        X_csp = self.csp.transform(X_task)
        classes = np.unique(y_train)
        self._centroids = {c: X_csp[y_train == c].mean(axis=0) for c in classes}
        pooled_cov = np.cov(X_csp.T) + 1e-6 * np.eye(X_csp.shape[1])
        self._inv_cov = np.linalg.inv(pooled_cov)
        return self

    def extract(self, epoch: np.ndarray) -> np.ndarray:
        """epoch: (n_channels, n_times) — full baseline+task. Returns 15-d array."""
        if self._centroids is None:
            raise RuntimeError("FeatureExtractor.fit(...) must be called first.")

        baseline = epoch[..., :self.baseline_samples]
        task = epoch[..., self.baseline_samples:]
        eps = 1e-12

        # ---- classifier confidence (CSP/LDA see ONLY the task slice) ----
        csp_feat = self.csp.transform(task[np.newaxis, ...])[0]
        dec = float(self.lda.decision_function([csp_feat])[0])
        proba = self.lda.predict_proba([csp_feat])[0]
        pred_class = self.lda.classes_[int(np.argmax(proba))]

        d = csp_feat - self._centroids[pred_class]
        mahal = float(np.sqrt(d @ self._inv_cov @ d))

        # ---- artifact signals ----
        p2p_per_ch_task = task.max(axis=-1) - task.min(axis=-1)
        p2p_max_task = float(np.max(p2p_per_ch_task))
        kurt_max_task = float(np.max(_kurtosis(task, axis=-1, fisher=True)))

        p2p_per_ch_base = baseline.max(axis=-1) - baseline.min(axis=-1)
        p2p_max_base = float(np.max(p2p_per_ch_base))

        # ---- baseline-referenced ERD on motor channels (μ band 8-13 Hz) ----
        rest_mu_c3 = _bandpower(baseline[self.c3], self.sfreq, 8, 13)
        rest_mu_c4 = _bandpower(baseline[self.c4], self.sfreq, 8, 13)
        task_mu_c3 = _bandpower(task[self.c3], self.sfreq, 8, 13)
        task_mu_c4 = _bandpower(task[self.c4], self.sfreq, 8, 13)
        erd_mu_c3 = (rest_mu_c3 - task_mu_c3) / (rest_mu_c3 + eps)
        erd_mu_c4 = (rest_mu_c4 - task_mu_c4) / (rest_mu_c4 + eps)
        # Positive: C3 desynchronizes more than C4 (right-hand MI signature).
        # Negative: C4 desynchronizes more (left-hand MI signature).
        erd_lat_mu = erd_mu_c3 - erd_mu_c4

        # ---- power-ratio shape (task window only) ----
        mu_motor = task_mu_c3 + task_mu_c4 + _bandpower(task[self.cz], self.sfreq, 8, 13)
        beta_motor = (
            _bandpower(task[self.c3], self.sfreq, 13, 30)
            + _bandpower(task[self.c4], self.sfreq, 13, 30)
            + _bandpower(task[self.cz], self.sfreq, 13, 30)
        )
        mu_ratio = mu_motor / (mu_motor + beta_motor + eps)

        motor_var = np.mean([np.var(task[ch]) for ch in self.motor_ch])
        global_var = np.mean(np.var(task, axis=-1))
        motor_rel = float(motor_var / (global_var + eps))

        return np.array([
            abs(dec),
            float(np.max(proba)),
            csp_feat[0], csp_feat[1], csp_feat[2], csp_feat[3],
            mahal,
            p2p_max_task,
            kurt_max_task,
            erd_mu_c3,
            erd_mu_c4,
            erd_lat_mu,
            mu_ratio,
            motor_rel,
            p2p_max_base,
        ], dtype=np.float64)

    def extract_many(self, X: np.ndarray) -> np.ndarray:
        return np.stack([self.extract(ep) for ep in X])


@dataclass
class RawQualityExtractor:
    """Extracts 8 raw-EEG quality/separability features per epoch.

    No CSP or LDA required — fully deployable before per-subject calibration.
    Used by the global acceptance model (Stage 1 training and all inference gates).

    Features fall into two groups:

        artifact signals       peak_to_peak_max (task), kurtosis_max (task),
                               baseline_p2p_max
        neurophysiological     erd_mu_c3, erd_mu_c4, erd_lat_mu,
          separability         mu_ratio_motor, motor_relative_power
    """

    ch_names: Sequence[str]
    sfreq: float = 250.0
    baseline_samples: int = 500
    feature_names: List[str] = field(default_factory=lambda: [
        "peak_to_peak_max",
        "kurtosis_max",
        "baseline_p2p_max",
        "erd_mu_c3",
        "erd_mu_c4",
        "erd_lat_mu",
        "mu_ratio_motor",
        "motor_relative_power",
    ])

    def __post_init__(self):
        self.c3 = self.ch_names.index("C3")
        self.cz = self.ch_names.index("Cz")
        self.c4 = self.ch_names.index("C4")
        self.motor_ch = [self.c3, self.cz, self.c4]

    def extract(self, epoch: np.ndarray) -> np.ndarray:
        """epoch: (n_channels, n_times) — full baseline+task. Returns 8-d array."""
        baseline = epoch[..., :self.baseline_samples]
        task = epoch[..., self.baseline_samples:]
        eps = 1e-12

        # ---- artifact signals ----
        p2p_per_ch_task = task.max(axis=-1) - task.min(axis=-1)
        p2p_max_task = float(np.max(p2p_per_ch_task))
        kurt_max_task = float(np.max(_kurtosis(task, axis=-1, fisher=True)))
        p2p_per_ch_base = baseline.max(axis=-1) - baseline.min(axis=-1)
        p2p_max_base = float(np.max(p2p_per_ch_base))

        # ---- baseline-referenced ERD on motor channels (mu band 8-13 Hz) ----
        rest_mu_c3 = _bandpower(baseline[self.c3], self.sfreq, 8, 13)
        rest_mu_c4 = _bandpower(baseline[self.c4], self.sfreq, 8, 13)
        task_mu_c3 = _bandpower(task[self.c3], self.sfreq, 8, 13)
        task_mu_c4 = _bandpower(task[self.c4], self.sfreq, 8, 13)
        erd_mu_c3 = (rest_mu_c3 - task_mu_c3) / (rest_mu_c3 + eps)
        erd_mu_c4 = (rest_mu_c4 - task_mu_c4) / (rest_mu_c4 + eps)
        erd_lat_mu = erd_mu_c3 - erd_mu_c4

        # ---- power-ratio shape (task window only) ----
        mu_motor = (task_mu_c3 + task_mu_c4
                    + _bandpower(task[self.cz], self.sfreq, 8, 13))
        beta_motor = (
            _bandpower(task[self.c3], self.sfreq, 13, 30)
            + _bandpower(task[self.c4], self.sfreq, 13, 30)
            + _bandpower(task[self.cz], self.sfreq, 13, 30)
        )
        mu_ratio = mu_motor / (mu_motor + beta_motor + eps)

        motor_var = np.mean([np.var(task[ch]) for ch in self.motor_ch])
        global_var = np.mean(np.var(task, axis=-1))
        motor_rel = float(motor_var / (global_var + eps))

        return np.array([
            p2p_max_task,
            kurt_max_task,
            p2p_max_base,
            erd_mu_c3,
            erd_mu_c4,
            erd_lat_mu,
            mu_ratio,
            motor_rel,
        ], dtype=np.float64)

    def extract_many(self, X: np.ndarray) -> np.ndarray:
        return np.stack([self.extract(ep) for ep in X])
