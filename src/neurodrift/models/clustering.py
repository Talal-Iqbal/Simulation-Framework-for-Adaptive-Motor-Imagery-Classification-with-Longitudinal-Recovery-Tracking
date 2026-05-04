"""Layer 7 — KMeans clustering of session feature vectors.

Refactored from `cell-kmeans`. Clusters session-level feature vectors into
discrete recovery / engagement stages.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


@dataclass
class ClusteringArtifacts:
    scaler: StandardScaler
    kmeans: KMeans
    feature_names: list[str]
    metrics: dict[str, float]


def cluster_sessions(
    panel: np.ndarray,
    feature_names: list[str],
    n_clusters: int = 3,
    random_state: int = 0,
) -> ClusteringArtifacts:
    """Standardise + cluster session feature vectors with KMeans."""
    scaler = StandardScaler()
    Xs = scaler.fit_transform(panel)

    n_clusters = min(n_clusters, max(2, len(panel) - 1))

    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = km.fit_predict(Xs)

    sil = float("nan")
    if len(set(labels)) > 1 and len(panel) > n_clusters:
        sil = float(silhouette_score(Xs, labels))

    metrics = {
        "n_clusters": int(n_clusters),
        "silhouette": sil,
        "inertia": float(km.inertia_),
    }
    return ClusteringArtifacts(
        scaler=scaler, kmeans=km, feature_names=feature_names, metrics=metrics
    )
