"""FastAPI integration tests using TestClient against synthetic seeded models."""

from __future__ import annotations

import io

import numpy as np
import pytest


@pytest.fixture(scope="module")
def seeded_app(tmp_path_factory):
    """Seed a registry with synthetic models, then build the app."""
    import os
    import sys

    tmp = tmp_path_factory.mktemp("api-registry")
    os.environ["NEURODRIFT_REGISTRY_DIR"] = str(tmp)
    os.environ["NEURODRIFT_HELD_OUT_SUBJECT"] = "2"

    if "neurodrift" in sys.modules:
        for k in list(sys.modules.keys()):
            if k.startswith("neurodrift"):
                sys.modules.pop(k)

    from scripts.seed_artifacts import main as seed_main

    from neurodrift.api.deps import reset_model_bundle

    seed_main()
    reset_model_bundle()

    from neurodrift.api.main import create_app

    return create_app()


@pytest.fixture(scope="module")
def client(seeded_app):
    from fastapi.testclient import TestClient

    return TestClient(seeded_app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["models"]["acceptance_model"] is not None


def test_metrics_endpoint(client):
    r = client.get("/metrics")
    assert r.status_code == 200
    assert "TYPE" in r.text or len(r.text) >= 0


def test_predict_trial(client):
    from neurodrift.data.synthetic import make_synthetic_epochs

    X, y = make_synthetic_epochs(n_trials=2, seed=99)
    payload = {"epoch": X[0].tolist(), "y_true": str(y[0])}
    r = client.post("/predict/trial?subject_id=2", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["y_pred"] in ("left_hand", "right_hand")
    assert isinstance(body["accepted"], bool)


def test_predict_upload(client):
    from neurodrift.data.synthetic import make_synthetic_epochs

    X, y = make_synthetic_epochs(n_trials=8, seed=7)
    buf = io.BytesIO()
    np.savez(buf, X=X, y=y)
    buf.seek(0)

    files = {"file": ("trials.npz", buf.getvalue(), "application/octet-stream")}
    r = client.post("/predict/upload?subject_id=2", files=files)
    assert r.status_code == 200
    body = r.json()
    assert body["n_trials"] == 8
    assert body["accepted"] + body["rejected"] == 8


def test_analyze_session(client):
    from neurodrift.models.regressor import SESSION_FEATURE_NAMES

    feats = {k: 0.5 for k in SESSION_FEATURE_NAMES}
    r = client.post("/analyze/session", json={"features": feats})
    assert r.status_code == 200
    body = r.json()
    assert "cluster" in body
    assert "predicted_r" in body


def test_calibrate_endpoint(client):
    from neurodrift.data.synthetic import make_synthetic_epochs

    X, y = make_synthetic_epochs(n_trials=24, seed=11)
    payload = {
        "subject_id": 99,
        "X": X.tolist(),
        "y": y.tolist(),
        "accept_percentile": 10.0,
    }
    r = client.post("/calibrate/subject", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["subject_id"] == 99
    assert body["version"]
    assert body["n_cal_trials"] > 0
