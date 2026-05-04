"""DeepChecks model-performance suite + minimum-threshold gates."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

from ..observability.logging import get_logger

log = get_logger(__name__)

DEFAULT_MIN_AUC = 0.65
DEFAULT_MIN_ACCURACY = 0.55


def run_model_evaluation_suite(
    F: np.ndarray,
    Y: np.ndarray,
    accept_artifacts,
    feature_names: list[str],
    min_auc: float = DEFAULT_MIN_AUC,
    min_accuracy: float = DEFAULT_MIN_ACCURACY,
) -> dict[str, Any]:
    """Verify the acceptance model meets minimum thresholds."""
    F_tr, F_te, Y_tr, Y_te = train_test_split(F, Y, test_size=0.25, stratify=Y, random_state=0)
    model = accept_artifacts.model
    scaler = accept_artifacts.scaler

    F_te_s = scaler.transform(F_te)
    p_te = model.predict_proba(F_te_s)[:, 1]
    y_pred = (p_te >= 0.5).astype(int)

    auc = float(roc_auc_score(Y_te, p_te))
    acc = float(accuracy_score(Y_te, y_pred))
    f1 = float(f1_score(Y_te, y_pred, average="macro"))

    issues: list[str] = []
    if auc < min_auc:
        issues.append(f"AUC {auc:.3f} below minimum {min_auc}")
    if acc < min_accuracy:
        issues.append(f"Accuracy {acc:.3f} below minimum {min_accuracy}")

    summary: dict[str, Any] = {
        "passed": len(issues) == 0,
        "issues": issues,
        "auc": auc,
        "accuracy": acc,
        "f1_macro": f1,
        "min_auc": min_auc,
        "min_accuracy": min_accuracy,
    }

    try:
        from deepchecks.tabular import Dataset
        from deepchecks.tabular.suites import model_evaluation

        df_tr = pd.DataFrame(F_tr, columns=feature_names)
        df_tr["label"] = Y_tr.astype(int)
        df_te = pd.DataFrame(F_te, columns=feature_names)
        df_te["label"] = Y_te.astype(int)

        train_ds = Dataset(df_tr, label="label", cat_features=[])
        test_ds = Dataset(df_te, label="label", cat_features=[])

        class _ScaledModel:
            def __init__(self, scaler, model):
                self.scaler = scaler
                self.model = model
                self.classes_ = model.classes_

            def predict(self, X):
                arr = X.values if hasattr(X, "values") else X
                return self.model.predict(self.scaler.transform(arr))

            def predict_proba(self, X):
                arr = X.values if hasattr(X, "values") else X
                return self.model.predict_proba(self.scaler.transform(arr))

        suite = model_evaluation()
        result = suite.run(train_ds, test_ds, _ScaledModel(scaler, model))
        summary["deepchecks_summary"] = {
            "n_checks": len(result.results),
            "passed_conditions": int(sum(1 for r in result.results if r.passed_conditions())),
        }
        summary["deepchecks_available"] = True
    except Exception as exc:
        log.warning("deepchecks_model_eval_unavailable", error=str(exc))
        summary["deepchecks_available"] = False

    if not summary["passed"]:
        raise ValueError(f"Model evaluation gate failed: {summary['issues']}")

    return summary
