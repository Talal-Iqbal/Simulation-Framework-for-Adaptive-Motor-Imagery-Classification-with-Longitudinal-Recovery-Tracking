"""Seed the registry with synthetic artifacts so the API can boot in CI.

Trains tiny acceptance + calibration + Layer-7 models on synthetic EEG. Used
when no real MOABB data is available (CI smoke test, fresh local checkout).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from neurodrift.config import BASELINE_SAMPLES, CH_NAMES, SFREQ
from neurodrift.data.splits import build_acceptance_dataset
from neurodrift.data.synthetic import make_synthetic_epochs
from neurodrift.features import build_raw_quality_extractor
from neurodrift.models.acceptance import train_acceptance_model
from neurodrift.models.classifier import calibrate_subject
from neurodrift.models.clustering import cluster_sessions
from neurodrift.models.regressor import (
    SESSION_FEATURE_NAMES,
    session_feature_vector,
    train_session_r_regressor,
)
from neurodrift.registry.store import ModelRegistry, hash_array


def main() -> int:
    rng_seed = 0
    X_global, _y_global = make_synthetic_epochs(n_trials=64, seed=rng_seed)
    rqe = build_raw_quality_extractor()
    F_global = rqe.extract_many(X_global)

    from corruption_engine import CorruptionEngine

    eng = CorruptionEngine(sfreq=SFREQ, baseline_samples=BASELINE_SAMPLES, seed=rng_seed)
    X_bad, meta = eng.generate_dataset(X_global, n_per_epoch=2)
    F_bad = rqe.extract_many(X_bad)

    dataset = build_acceptance_dataset(F_global, meta, F_bad, rqe.feature_names)
    accept_artifacts, _ = train_acceptance_model(dataset, rqe.feature_names)

    X_cal, y_cal = make_synthetic_epochs(n_trials=64, seed=rng_seed + 1)
    X_eval, y_eval = make_synthetic_epochs(n_trials=32, seed=rng_seed + 2)

    cal_artifacts, _ = calibrate_subject(
        X_cal,
        y_cal,
        X_eval,
        y_eval,
        rqe,
        accept_artifacts,
        subject_id=2,
        accept_percentile=20.0,
        min_per_class=4,
    )

    from feature_extractor import FeatureExtractor

    fe = FeatureExtractor(csp=cal_artifacts.csp, lda=cal_artifacts.lda, ch_names=list(CH_NAMES))
    fe.fit(X_cal, y_cal)

    panel_rows: list[list[float]] = []
    r_values: list[float] = []
    rs = np.linspace(0.0, 1.0, 6)
    for r in rs:
        feats, _ = session_feature_vector(X_eval, y_eval, cal_artifacts.csp, cal_artifacts.lda, fe)
        feats["lda_session_acc"] = float(np.clip(0.5 + 0.4 * r, 0.0, 1.0))
        panel_rows.append([feats[k] for k in SESSION_FEATURE_NAMES])
        r_values.append(float(r))

    panel = np.array(panel_rows, dtype=float)
    r_arr = np.array(r_values, dtype=float)

    reg = train_session_r_regressor(panel, r_arr)
    clust = cluster_sessions(panel, SESSION_FEATURE_NAMES, n_clusters=3)

    registry = ModelRegistry()
    train_hash = hash_array(X_global)

    registry.save(
        "acceptance_model",
        accept_artifacts,
        metrics=accept_artifacts.metrics,
        meta={"frozen": True, "synthetic_seed": True},
        train_data_hash=train_hash,
    )
    registry.save(
        f"calibration_subject_{cal_artifacts.subject_id}",
        cal_artifacts,
        metrics=cal_artifacts.metrics,
        meta={"cal_threshold": cal_artifacts.cal_threshold, "synthetic_seed": True},
        train_data_hash=train_hash,
    )
    registry.save("session_r_regressor", reg, metrics=reg.metrics, train_data_hash=train_hash)
    registry.save("session_kmeans", clust, metrics=clust.metrics, train_data_hash=train_hash)

    print("Seeded synthetic registry artifacts:")
    print(f"  acceptance auc_test = {accept_artifacts.metrics.get('auc_test'):.3f}")
    print(f"  calibration in-sample acc = {cal_artifacts.metrics.get('in_sample_accuracy'):.3f}")
    print(f"  regressor R^2 (LOO) = {reg.metrics.get('r2_loo'):.3f}")
    print(f"  clustering silhouette = {clust.metrics.get('silhouette')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
