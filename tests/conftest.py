"""Test fixtures and PYTHONPATH bootstrap."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session", autouse=True)
def _isolated_registry(tmp_path_factory):
    """Use an isolated registry/data dir for the entire test session."""
    tmp = tmp_path_factory.mktemp("neurodrift-registry")
    os.environ["NEURODRIFT_REGISTRY_DIR"] = str(tmp / "registry")
    os.environ["NEURODRIFT_DATA_DIR"] = str(tmp / "data")
    os.environ["NEURODRIFT_HELD_OUT_SUBJECT"] = "2"
    yield tmp


@pytest.fixture
def synthetic_epochs():
    from neurodrift.data.synthetic import make_synthetic_epochs

    X, y = make_synthetic_epochs(n_trials=24, seed=0)
    return X, y


@pytest.fixture
def rqe():
    from neurodrift.features import build_raw_quality_extractor

    return build_raw_quality_extractor()


@pytest.fixture
def acceptance_dataset(synthetic_epochs, rqe):
    from corruption_engine import CorruptionEngine

    from neurodrift.config import BASELINE_SAMPLES, SFREQ
    from neurodrift.data.splits import build_acceptance_dataset

    X, _ = synthetic_epochs
    F_clean = rqe.extract_many(X)

    eng = CorruptionEngine(sfreq=SFREQ, baseline_samples=BASELINE_SAMPLES, seed=0)
    X_bad, meta = eng.generate_dataset(X, n_per_epoch=2)
    F_bad = rqe.extract_many(X_bad)

    return build_acceptance_dataset(F_clean, meta, F_bad, rqe.feature_names)


@pytest.fixture
def trained_acceptance(acceptance_dataset, rqe):
    from neurodrift.models.acceptance import train_acceptance_model

    artifacts, _ = train_acceptance_model(acceptance_dataset, rqe.feature_names)
    return artifacts


@pytest.fixture
def calibrated_subject(synthetic_epochs, rqe, trained_acceptance):
    from neurodrift.models.classifier import calibrate_subject

    X_cal, y_cal = synthetic_epochs
    X_eval, y_eval = synthetic_epochs
    artifacts, _ = calibrate_subject(
        X_cal, y_cal, X_eval, y_eval, rqe, trained_acceptance,
        subject_id=2, accept_percentile=20.0, min_per_class=2,
    )
    return artifacts
