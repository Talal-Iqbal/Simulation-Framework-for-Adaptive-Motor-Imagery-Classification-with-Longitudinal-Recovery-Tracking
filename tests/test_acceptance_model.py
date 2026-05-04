"""Stage 1 acceptance model training tests."""

from __future__ import annotations


def test_acceptance_model_trains(trained_acceptance):
    assert trained_acceptance.model is not None
    assert trained_acceptance.scaler is not None
    assert "auc_test" in trained_acceptance.metrics
    assert 0.0 <= trained_acceptance.metrics["auc_test"] <= 1.0


def test_acceptance_predict_proba_shape(synthetic_epochs, rqe, trained_acceptance):
    X, _ = synthetic_epochs
    F = rqe.extract_many(X)
    p = trained_acceptance.predict_proba(F)
    assert p.shape == (len(X),)
    assert ((p >= 0.0) & (p <= 1.0)).all()


def test_registry_save_load(trained_acceptance, tmp_path):
    from neurodrift.registry.store import ModelRegistry

    reg = ModelRegistry(tmp_path / "registry")
    entry = reg.save(
        "acceptance_model",
        trained_acceptance,
        metrics=trained_acceptance.metrics,
        train_data_hash="deadbeef",
    )
    assert (entry.path / "model.joblib").exists()
    assert (entry.path / "manifest.json").exists()

    latest = reg.latest("acceptance_model")
    assert latest is not None
    assert latest.version == entry.version
    loaded = latest.load()
    assert hasattr(loaded, "model")
    assert hasattr(loaded, "scaler")
