"""Stage 1 — Global Acceptance Model (frozen after training).

Refactored from the notebook's `cell-train-accept-model` cell. The acceptance
model classifies each EEG trial as accept (quality sufficient for downstream
MI classification) or reject. It uses 8 raw-EEG quality features only and is
never re-trained during calibration or inference.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from ..data.splits import AcceptanceDataset


@dataclass
class AcceptanceArtifacts:
    """Frozen acceptance-model bundle ready for joblib serialization."""

    model: LogisticRegression
    scaler: StandardScaler
    feature_names: list[str]
    metrics: dict[str, float]
    p2p_cut: float
    basep2p_cut: float

    def predict_proba(self, F: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(self.scaler.transform(F))[:, 1]


def train_acceptance_model(
    dataset: AcceptanceDataset,
    feature_names: list[str],
    test_size: float = 0.2,
    random_state: int = 0,
    C: float = 1.0,
    max_iter: int = 1000,
) -> tuple[AcceptanceArtifacts, dict[str, object]]:
    """Train + evaluate the frozen acceptance model.

    Returns the artifacts bundle plus a diagnostics dict (classification
    report, confusion matrix, per-source rejection rates, AUCs).
    """
    F_all = dataset.F_all
    Y_all = dataset.Y_all

    idx_all = np.arange(len(Y_all))
    idx_tr, idx_te, Y_tr, Y_te = train_test_split(
        idx_all, Y_all, test_size=test_size, stratify=Y_all, random_state=random_state
    )
    F_tr, F_te = F_all[idx_tr], F_all[idx_te]

    scaler = StandardScaler().fit(F_tr)
    F_tr_s = scaler.transform(F_tr)
    F_te_s = scaler.transform(F_te)

    model = LogisticRegression(
        C=C, class_weight="balanced", max_iter=max_iter, random_state=random_state
    ).fit(F_tr_s, Y_tr)

    p_tr = model.predict_proba(F_tr_s)[:, 1]
    p_te = model.predict_proba(F_te_s)[:, 1]
    auc_tr = float(roc_auc_score(Y_tr, p_tr))
    auc_te = float(roc_auc_score(Y_te, p_te))

    y_pred_te = model.predict(F_te_s)
    accuracy_te = float(np.mean(y_pred_te == Y_te))

    report = classification_report(
        Y_te, y_pred_te, target_names=["reject", "accept"], output_dict=True
    )
    cm = confusion_matrix(Y_te, y_pred_te).tolist()

    metrics = {
        "auc_train": auc_tr,
        "auc_test": auc_te,
        "accuracy_test": accuracy_te,
        "n_train": int(len(idx_tr)),
        "n_test": int(len(idx_te)),
    }

    artifacts = AcceptanceArtifacts(
        model=model,
        scaler=scaler,
        feature_names=feature_names,
        metrics=metrics,
        p2p_cut=dataset.p2p_cut,
        basep2p_cut=dataset.basep2p_cut,
    )

    diagnostics: dict[str, object] = {
        "metrics": metrics,
        "classification_report": report,
        "confusion_matrix": cm,
        "test_indices": idx_te.tolist(),
    }

    return artifacts, diagnostics
