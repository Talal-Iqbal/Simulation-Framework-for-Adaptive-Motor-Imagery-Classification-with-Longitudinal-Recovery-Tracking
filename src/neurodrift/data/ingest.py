"""MOABB data ingestion for the BNCI2014_001 motor-imagery paradigm.

Mirrors the loading logic from the research notebook's `cell-multisubject-load`
and `cell-load-subj9` cells, bundled into reusable functions for Prefect tasks
and one-off CLI scripts.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np


def _build_paradigm():
    from moabb.datasets import BNCI2014_001
    from moabb.paradigms import LeftRightImagery

    paradigm = LeftRightImagery(tmin=-2)
    dataset = BNCI2014_001()
    return paradigm, dataset


def load_subjects(
    subject_ids: Sequence[int],
    trials_per_subject: int = 144,
) -> tuple[np.ndarray, np.ndarray]:
    """Load and concatenate trials from the given subjects.

    Each subject contributes `trials_per_subject` trials from session 1 (the
    calibration session in BNCI2014_001).

    Returns
    -------
    X : (n_total, n_channels, n_times) float64
    y : (n_total,) string labels in {"left_hand", "right_hand"}
    """
    paradigm, dataset = _build_paradigm()
    all_X, all_y = [], []
    for sid in subject_ids:
        Xs, ys, _ = paradigm.get_data(dataset, subjects=[sid])
        all_X.append(Xs[:trials_per_subject])
        all_y.append(ys[:trials_per_subject])
    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    return X, y


def load_held_out_subject(
    subject_id: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load held-out subject and split into calibration pool / eval session.

    Returns (X_cal_pool, y_cal_pool, X_eval, y_eval).
    """
    paradigm, dataset = _build_paradigm()
    X, y, _ = paradigm.get_data(dataset, subjects=[subject_id])
    return X[:144], y[:144], X[144:288], y[144:288]


def load_all(subject_ids: Iterable[int]) -> tuple[np.ndarray, np.ndarray]:
    """Load both sessions per subject (288 trials each), concatenated."""
    paradigm, dataset = _build_paradigm()
    Xs, ys = [], []
    for sid in subject_ids:
        X, y, _ = paradigm.get_data(dataset, subjects=[sid])
        Xs.append(X)
        ys.append(y)
    return np.concatenate(Xs, axis=0), np.concatenate(ys, axis=0)
