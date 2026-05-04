"""Local model registry: persisted joblib bundles + per-version manifest.

Layout::

    artifacts/registry/
        <name>/
            latest.json                  # pointer to current version
            <version>/
                model.joblib
                manifest.json            # metrics, training hash, timestamps
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib

from ..config import get_settings


@dataclass
class RegistryEntry:
    """A single saved model version."""

    name: str
    version: str
    path: Path
    manifest: dict[str, Any] = field(default_factory=dict)

    def load(self) -> Any:
        return joblib.load(self.path / "model.joblib")


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _hash_payload(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()[:16]


class ModelRegistry:
    """Filesystem-backed model registry. No network calls, no DB."""

    def __init__(self, root: Path | None = None) -> None:
        settings = get_settings()
        self.root = Path(root) if root is not None else settings.registry_dir
        self.root.mkdir(parents=True, exist_ok=True)

    def _model_dir(self, name: str) -> Path:
        d = self.root / name
        d.mkdir(parents=True, exist_ok=True)
        return d

    def save(
        self,
        name: str,
        artifact: Any,
        metrics: dict[str, Any] | None = None,
        meta: dict[str, Any] | None = None,
        train_data_hash: str | None = None,
    ) -> RegistryEntry:
        """Persist a model version under `<registry>/<name>/<version>/`."""
        timestamp = _utcnow_iso()
        version = timestamp.replace(":", "").replace("-", "")
        version_dir = self._model_dir(name) / version
        version_dir.mkdir(parents=True, exist_ok=True)

        joblib.dump(artifact, version_dir / "model.joblib")

        try:
            payload_bytes = (version_dir / "model.joblib").read_bytes()
            artifact_hash = _hash_payload(payload_bytes)
        except OSError:
            artifact_hash = ""

        manifest: dict[str, Any] = {
            "name": name,
            "version": version,
            "created_at": timestamp,
            "metrics": metrics or {},
            "meta": meta or {},
            "artifact_sha256": artifact_hash,
            "train_data_hash": train_data_hash or "",
        }
        (version_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

        latest = self._model_dir(name) / "latest.json"
        latest.write_text(json.dumps({"version": version, "path": str(version_dir)}, indent=2))

        return RegistryEntry(name=name, version=version, path=version_dir, manifest=manifest)

    def latest(self, name: str) -> RegistryEntry | None:
        latest_file = self._model_dir(name) / "latest.json"
        if not latest_file.exists():
            return None
        info = json.loads(latest_file.read_text())
        version_dir = Path(info["path"])
        manifest_path = version_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        return RegistryEntry(name=name, version=info["version"], path=version_dir, manifest=manifest)

    def list_versions(self, name: str) -> list[str]:
        d = self._model_dir(name)
        return sorted([p.name for p in d.iterdir() if p.is_dir()])

    def delete(self, name: str, version: str | None = None) -> None:
        """Delete a single version, or the entire model directory."""
        if version is None:
            shutil.rmtree(self._model_dir(name), ignore_errors=True)
        else:
            shutil.rmtree(self._model_dir(name) / version, ignore_errors=True)


def hash_array(arr) -> str:
    """Deterministic short hash for a numpy array (used for train-data hashes)."""
    import numpy as np

    a = np.ascontiguousarray(arr)
    return hashlib.sha256(a.tobytes()).hexdigest()[:16]


def hash_dict(d: dict[str, Any]) -> str:
    return _hash_payload(json.dumps(d, sort_keys=True, default=str).encode("utf-8"))
