"""DeepChecks data-integrity suite.

If DeepChecks is unavailable the suite degrades to lightweight numpy/pandas
checks so CI on minimal images still validates basic invariants.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..observability.logging import get_logger

log = get_logger(__name__)


def _basic_integrity_checks(F: np.ndarray, Y: np.ndarray, feature_names: list[str]) -> dict[str, Any]:
    df = pd.DataFrame(F, columns=feature_names)
    nulls = int(df.isna().sum().sum())
    duplicates = int(df.duplicated().sum())
    inf_count = int(np.isinf(F).sum())

    classes, counts = np.unique(Y, return_counts=True)
    class_dist = {str(c): int(n) for c, n in zip(classes, counts)}

    issues: list[str] = []
    if nulls > 0:
        issues.append(f"{nulls} null values")
    if inf_count > 0:
        issues.append(f"{inf_count} infinite values")
    if len(classes) < 2:
        issues.append("only one class present")

    passed = len(issues) == 0
    return {
        "passed": passed,
        "issues": issues,
        "n_samples": int(len(F)),
        "n_features": int(F.shape[1]),
        "nulls": nulls,
        "duplicates": duplicates,
        "inf_count": inf_count,
        "class_distribution": class_dist,
    }


def run_data_integrity_suite(
    F: np.ndarray, Y: np.ndarray, feature_names: list[str]
) -> dict[str, Any]:
    """Run DeepChecks data-integrity suite (with graceful fallback)."""
    basic = _basic_integrity_checks(F, Y, feature_names)

    try:
        from deepchecks.tabular import Dataset
        from deepchecks.tabular.suites import data_integrity

        df = pd.DataFrame(F, columns=feature_names)
        df["label"] = Y.astype(int)
        ds = Dataset(df, label="label", cat_features=[])
        suite = data_integrity()
        result = suite.run(ds)
        basic["deepchecks_summary"] = {
            "n_checks": len(result.results),
            "passed_conditions": int(sum(1 for r in result.results if r.passed_conditions())),
            "total_conditions": int(sum(len(r.conditions_results) for r in result.results)),
        }
        basic["deepchecks_available"] = True
    except Exception as exc:
        log.warning("deepchecks_data_integrity_unavailable", error=str(exc))
        basic["deepchecks_available"] = False

    if not basic["passed"]:
        raise ValueError(f"Data integrity check failed: {basic['issues']}")

    return basic
