"""End-to-end Prefect training flow.

Stages:
  1. Ingest MOABB data for global training subjects + held-out subject
  2. Extract raw-quality features
  3. Apply CorruptionEngine to generate hard negatives
  4. Build acceptance dataset
  5. DeepChecks data integrity suite (gate)
  6. Train acceptance model (Stage 1)
  7. DeepChecks model performance suite (gate)
  8. Calibrate held-out subject (Stage 2: CSP+LDA)
  9. Layer 7 longitudinal: regression + clustering
 10. Save all artifacts to the local registry
 11. Email notification on success / failure
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..config import (
    BASELINE_SAMPLES,
    SFREQ,
    get_settings,
)
from ..data.ingest import load_held_out_subject, load_subjects
from ..data.splits import build_acceptance_dataset
from ..features import build_raw_quality_extractor
from ..models.acceptance import train_acceptance_model
from ..models.classifier import calibrate_subject
from ..models.clustering import cluster_sessions
from ..models.regressor import (
    SESSION_FEATURE_NAMES,
    session_feature_vector,
    train_session_r_regressor,
)
from ..observability.logging import configure_logging, get_logger
from ..registry.store import ModelRegistry, hash_array
from .notify import send_email_notification

try:
    from prefect import flow, task
except ImportError:
    def task(*args, **kwargs):
        def deco(fn):
            return fn
        if args and callable(args[0]):
            return args[0]
        return deco

    def flow(*args, **kwargs):
        def deco(fn):
            return fn
        if args and callable(args[0]):
            return args[0]
        return deco


log = get_logger(__name__)


@task(retries=3, retry_delay_seconds=30)
def ingest_global_pool(subject_ids: list[int]) -> dict[str, np.ndarray]:
    log.info("ingest_global_pool.start", subjects=subject_ids)
    X, y = load_subjects(subject_ids)
    log.info("ingest_global_pool.done", n_trials=int(len(X)))
    return {"X": X, "y": y}


@task(retries=3, retry_delay_seconds=30)
def ingest_held_out(subject_id: int) -> dict[str, np.ndarray]:
    log.info("ingest_held_out.start", subject=subject_id)
    X_cal, y_cal, X_eval, y_eval = load_held_out_subject(subject_id)
    return {"X_cal": X_cal, "y_cal": y_cal, "X_eval": X_eval, "y_eval": y_eval}


@task
def extract_raw_quality_features(X: np.ndarray) -> dict[str, Any]:
    rqe = build_raw_quality_extractor()
    F = rqe.extract_many(X)
    return {"F": F, "feature_names": rqe.feature_names, "rqe": rqe}


@task
def apply_corruption_engine(X: np.ndarray, n_per_epoch: int = 2, seed: int = 42) -> dict[str, Any]:
    from corruption_engine import CorruptionEngine

    engine = CorruptionEngine(sfreq=SFREQ, baseline_samples=BASELINE_SAMPLES, seed=seed)
    X_bad, meta = engine.generate_dataset(X, n_per_epoch=n_per_epoch)
    return {"X_bad": X_bad, "meta": meta}


@task
def build_dataset_task(
    F_global: np.ndarray,
    bad_meta: list[dict],
    F_bad: np.ndarray,
    feature_names: list[str],
):
    return build_acceptance_dataset(F_global, bad_meta, F_bad, feature_names)


@task
def deepchecks_data_integrity(F_all: np.ndarray, Y_all: np.ndarray, feature_names: list[str]) -> dict[str, Any]:
    from ..testing.data_validation import run_data_integrity_suite

    return run_data_integrity_suite(F_all, Y_all, feature_names)


@task
def train_acceptance_task(dataset, feature_names: list[str]) -> dict[str, Any]:
    artifacts, diagnostics = train_acceptance_model(dataset, feature_names)
    return {"artifacts": artifacts, "diagnostics": diagnostics}


@task
def deepchecks_model_perf(
    F_all: np.ndarray, Y_all: np.ndarray, artifacts, feature_names: list[str]
) -> dict[str, Any]:
    from ..testing.model_validation import run_model_evaluation_suite

    return run_model_evaluation_suite(F_all, Y_all, artifacts, feature_names)


@task
def calibrate_subject_task(
    X_cal: np.ndarray,
    y_cal: np.ndarray,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
    rqe,
    accept_artifacts,
    subject_id: int,
) -> dict[str, Any]:
    cal_artifacts, diagnostics = calibrate_subject(
        X_cal, y_cal, X_eval, y_eval, rqe, accept_artifacts, subject_id=subject_id
    )
    return {"artifacts": cal_artifacts, "diagnostics": diagnostics}


@task
def layer7_longitudinal(
    X_eval: np.ndarray,
    y_eval: np.ndarray,
    cal_artifacts,
    fe_factory,
) -> dict[str, Any]:
    """Run Layer 7 regression + clustering on a small synthetic longitudinal panel.

    Builds a degraded session series via DegradationModel and the chosen prior,
    then collapses each session into the standard feature vector.
    """
    from degradation_model import DegradationModel, prior_default

    csp = cal_artifacts.csp
    lda = cal_artifacts.lda

    fe = fe_factory(csp, lda)
    fe.fit(X_eval[: max(20, len(X_eval) // 2)], y_eval[: max(20, len(y_eval) // 2)])

    deg = DegradationModel(sfreq=SFREQ, baseline_samples=BASELINE_SAMPLES, seed=0)
    rs, sessions = deg.generate_longitudinal(
        X_eval, n_sessions=6, shape="linear", jitter=False, prior=prior_default
    )

    panel_rows: list[list[float]] = []
    r_values: list[float] = []
    for r, X_session in sessions:
        feats, _ = session_feature_vector(X_session, y_eval, csp, lda, fe)
        panel_rows.append([feats[k] for k in SESSION_FEATURE_NAMES])
        r_values.append(float(r))

    panel = np.array(panel_rows, dtype=float)
    r_arr = np.array(r_values, dtype=float)

    reg_artifacts = train_session_r_regressor(panel, r_arr)
    clust_artifacts = cluster_sessions(panel, SESSION_FEATURE_NAMES, n_clusters=3)

    return {
        "regression": reg_artifacts,
        "clustering": clust_artifacts,
        "panel": panel,
        "r_values": r_arr,
    }


@task
def save_all_artifacts(
    accept_artifacts,
    cal_artifacts,
    layer7,
    train_data_hash: str,
) -> dict[str, str]:
    registry = ModelRegistry()
    saved: dict[str, str] = {}

    saved["acceptance_model"] = registry.save(
        "acceptance_model",
        accept_artifacts,
        metrics=accept_artifacts.metrics,
        meta={"frozen": True},
        train_data_hash=train_data_hash,
    ).version

    saved[f"calibration_subject_{cal_artifacts.subject_id}"] = registry.save(
        f"calibration_subject_{cal_artifacts.subject_id}",
        cal_artifacts,
        metrics=cal_artifacts.metrics,
        meta={"cal_threshold": cal_artifacts.cal_threshold},
        train_data_hash=train_data_hash,
    ).version

    saved["session_r_regressor"] = registry.save(
        "session_r_regressor",
        layer7["regression"],
        metrics=layer7["regression"].metrics,
        train_data_hash=train_data_hash,
    ).version

    saved["session_kmeans"] = registry.save(
        "session_kmeans",
        layer7["clustering"],
        metrics=layer7["clustering"].metrics,
        train_data_hash=train_data_hash,
    ).version

    return saved


@flow(name="neurodrift-training")
def neurodrift_training_flow(
    global_subjects: list[int] | None = None,
    held_out_subject: int | None = None,
) -> dict[str, Any]:
    """Full end-to-end training flow."""
    configure_logging()
    settings = get_settings()
    settings.ensure_dirs()

    global_subjects = global_subjects or settings.global_train_subjects
    held_out_subject = held_out_subject or settings.held_out_subject

    try:
        global_pool = ingest_global_pool(global_subjects)
        held_out = ingest_held_out(held_out_subject)

        rq = extract_raw_quality_features(global_pool["X"])
        corrupt = apply_corruption_engine(global_pool["X"])
        F_bad = build_raw_quality_extractor().extract_many(corrupt["X_bad"])

        dataset = build_dataset_task(rq["F"], corrupt["meta"], F_bad, rq["feature_names"])
        data_diag = deepchecks_data_integrity(dataset.F_all, dataset.Y_all, rq["feature_names"])

        accept = train_acceptance_task(dataset, rq["feature_names"])
        accept_artifacts = accept["artifacts"]
        model_diag = deepchecks_model_perf(
            dataset.F_all, dataset.Y_all, accept_artifacts, rq["feature_names"]
        )

        cal = calibrate_subject_task(
            held_out["X_cal"],
            held_out["y_cal"],
            held_out["X_eval"],
            held_out["y_eval"],
            rq["rqe"],
            accept_artifacts,
            held_out_subject,
        )
        cal_artifacts = cal["artifacts"]

        from feature_extractor import FeatureExtractor

        from ..config import CH_NAMES

        def fe_factory(csp, lda):
            return FeatureExtractor(csp=csp, lda=lda, ch_names=list(CH_NAMES))

        layer7 = layer7_longitudinal(
            held_out["X_eval"], held_out["y_eval"], cal_artifacts, fe_factory
        )

        train_hash = hash_array(global_pool["X"])
        saved = save_all_artifacts(accept_artifacts, cal_artifacts, layer7, train_hash)

        send_email_notification(
            subject="[NeuroDrift] Training flow succeeded",
            body=(
                f"Acceptance AUC (test): {accept_artifacts.metrics.get('auc_test', 0):.3f}\n"
                f"Calibration in-sample acc: {cal_artifacts.metrics.get('in_sample_accuracy', 0):.3f}\n"
                f"Calibration eval acc: {cal_artifacts.metrics.get('eval_accuracy', 0):.3f}\n"
                f"Layer-7 regressor R^2 (LOO): {layer7['regression'].metrics.get('r2_loo', 0):.3f}\n"
                f"\nSaved registry versions: {saved}\n"
            ),
        )

        return {
            "saved": saved,
            "accept_metrics": accept_artifacts.metrics,
            "calibration_metrics": cal_artifacts.metrics,
            "layer7_metrics": {
                "regression": layer7["regression"].metrics,
                "clustering": layer7["clustering"].metrics,
            },
            "deepchecks": {"data": data_diag, "model": model_diag},
        }
    except Exception as exc:
        log.error("training_flow_failed", error=str(exc))
        send_email_notification(
            subject="[NeuroDrift] Training flow FAILED",
            body=f"NeuroDrift training flow raised: {exc!r}\nCheck Prefect logs for details.",
        )
        raise
