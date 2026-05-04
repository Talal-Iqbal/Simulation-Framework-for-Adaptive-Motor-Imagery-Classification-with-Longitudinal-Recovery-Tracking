"""Build the accept/reject training matrix from clean + corrupted trials.

Mirrors `cell-build-dataset` from the research notebook: positives are clean
trials with no artifact and meaningful ERD lateralization, borderline negatives
are clean-but-weak-lateralization trials, and hard negatives come from the
CorruptionEngine applied to all clean trials.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import ERDLAT_THRESHOLD, KURT_THRESHOLD


@dataclass
class AcceptanceDataset:
    """Container for the assembled accept/reject training data."""

    F_all: np.ndarray
    Y_all: np.ndarray
    src_type: np.ndarray
    p2p_cut: float
    basep2p_cut: float
    n_pos: int
    n_borderline: int
    n_bad: int


def build_acceptance_dataset(
    F_global: np.ndarray,
    bad_meta_global: list[dict],
    F_bad_global: np.ndarray,
    feature_names: list[str],
) -> AcceptanceDataset:
    """Assemble positives, borderline negatives, and hard negatives.

    Thresholds for artifact gating are derived from the 95th percentile of
    p2p across clean trials (per the notebook).
    """
    idx_p2p = feature_names.index("peak_to_peak_max")
    idx_kurt = feature_names.index("kurtosis_max")
    idx_basep2p = feature_names.index("baseline_p2p_max")
    idx_erdlat = feature_names.index("erd_lat_mu")

    p2p_cut = float(np.percentile(F_global[:, idx_p2p], 95))
    basep2p_cut = float(np.percentile(F_global[:, idx_basep2p], 95))

    mask_artifact_ok = (
        (F_global[:, idx_p2p] <= p2p_cut)
        & (F_global[:, idx_kurt] <= KURT_THRESHOLD)
        & (F_global[:, idx_basep2p] <= basep2p_cut)
    )
    mask_sep_ok = np.abs(F_global[:, idx_erdlat]) >= ERDLAT_THRESHOLD
    mask_positive = mask_artifact_ok & mask_sep_ok
    mask_borderline = mask_artifact_ok & ~mask_sep_ok

    F_pos = F_global[mask_positive]
    F_borderline = F_global[mask_borderline]

    F_neg = np.vstack([F_borderline, F_bad_global])
    F_all = np.vstack([F_pos, F_neg])
    Y_all = np.concatenate(
        [
            np.ones(len(F_pos)),
            np.zeros(len(F_borderline)),
            np.zeros(len(F_bad_global)),
        ]
    )
    src_type = np.concatenate(
        [
            np.array(["clean_good"] * len(F_pos)),
            np.array(["clean_borderline"] * len(F_borderline)),
            np.array([m["family"] for m in bad_meta_global]),
        ]
    )

    return AcceptanceDataset(
        F_all=F_all,
        Y_all=Y_all,
        src_type=src_type,
        p2p_cut=p2p_cut,
        basep2p_cut=basep2p_cut,
        n_pos=int(len(F_pos)),
        n_borderline=int(len(F_borderline)),
        n_bad=int(len(F_bad_global)),
    )


def train_test_split_indices(
    n: int, test_size: float = 0.2, random_state: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_state)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_test = int(round(n * test_size))
    return idx[n_test:], idx[:n_test]
