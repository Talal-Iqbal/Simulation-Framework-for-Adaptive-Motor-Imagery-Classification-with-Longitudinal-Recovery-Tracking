"""Synthetic EEG fixture generator for CI / unit tests.

Produces epoch tensors with realistic shapes and lightly band-limited content
so feature extractors and the corruption engine run end-to-end. Not intended
for model accuracy tests — only smoke / contract tests.
"""

from __future__ import annotations

import numpy as np

from ..config import (
    BASELINE_SAMPLES,
    C3_IDX,
    C4_IDX,
    CH_NAMES,
    EPOCH_SAMPLES,
    N_CHANNELS,
    SFREQ,
)


def _bandlimited_noise(rng: np.random.Generator, n_samples: int, sfreq: float) -> np.ndarray:
    """Generate noise concentrated in 8-32 Hz (the bandpass MOABB applies)."""
    n = n_samples
    freqs = np.fft.rfftfreq(n, d=1.0 / sfreq)
    spec = rng.standard_normal(len(freqs)) + 1j * rng.standard_normal(len(freqs))
    mask = (freqs >= 4.0) & (freqs <= 32.0)
    spec[~mask] = 0.0
    sig = np.fft.irfft(spec, n=n)
    return sig.astype(np.float64)


def make_synthetic_epochs(
    n_trials: int = 32,
    seed: int = 0,
    class_separability: float = 0.8,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a small synthetic dataset of left/right MI trials.

    Class signal is encoded by a contralateral-ERD mu/beta amplitude difference
    on C3 vs C4, which mirrors the discriminative shape that real BNCI2014_001
    trials carry.
    """
    rng = np.random.default_rng(seed)
    X = np.zeros((n_trials, N_CHANNELS, EPOCH_SAMPLES), dtype=np.float64)
    y = np.empty(n_trials, dtype=object)
    classes = np.array(["left_hand", "right_hand"])

    for i in range(n_trials):
        cls = classes[i % 2]
        y[i] = cls
        for ch in range(N_CHANNELS):
            base = _bandlimited_noise(rng, BASELINE_SAMPLES, SFREQ) * 1e-5
            task = _bandlimited_noise(rng, EPOCH_SAMPLES - BASELINE_SAMPLES, SFREQ) * 1e-5
            if ch == C3_IDX:
                if cls == "right_hand":
                    task *= max(0.0, 1.0 - class_separability)
            elif ch == C4_IDX:
                if cls == "left_hand":
                    task *= max(0.0, 1.0 - class_separability)
            X[i, ch, :BASELINE_SAMPLES] = base
            X[i, ch, BASELINE_SAMPLES:] = task

    return X, y.astype(str)


def channel_names() -> list:
    return list(CH_NAMES)
