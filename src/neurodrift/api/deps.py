"""Dependency providers: model loading, registry access, settings."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from ..config import Settings, get_settings
from ..features import build_raw_quality_extractor
from ..registry.store import ModelRegistry


@lru_cache(maxsize=1)
def get_cached_settings() -> Settings:
    return get_settings()


@lru_cache(maxsize=1)
def get_registry() -> ModelRegistry:
    return ModelRegistry(get_cached_settings().registry_dir)


def _load(registry: ModelRegistry, name: str) -> tuple[Any | None, str | None]:
    entry = registry.latest(name)
    if entry is None:
        return None, None
    return entry.load(), entry.version


class ModelBundle:
    """Cached models loaded from the registry, refreshed on demand."""

    def __init__(self) -> None:
        self.acceptance: Any | None = None
        self.acceptance_version: str | None = None
        self.calibration: dict[int, Any] = {}
        self.calibration_version: dict[int, str] = {}
        self.regressor: Any | None = None
        self.regressor_version: str | None = None
        self.kmeans: Any | None = None
        self.kmeans_version: str | None = None
        self.rqe = build_raw_quality_extractor()

    def load_all(self, registry: ModelRegistry, default_subject: int = 2) -> None:
        self.acceptance, self.acceptance_version = _load(registry, "acceptance_model")
        cal, cal_ver = _load(registry, f"calibration_subject_{default_subject}")
        if cal is not None:
            self.calibration[default_subject] = cal
            self.calibration_version[default_subject] = cal_ver  # type: ignore[assignment]
        self.regressor, self.regressor_version = _load(registry, "session_r_regressor")
        self.kmeans, self.kmeans_version = _load(registry, "session_kmeans")

    def get_calibration(self, subject_id: int, registry: ModelRegistry) -> Any | None:
        if subject_id in self.calibration:
            return self.calibration[subject_id]
        cal, ver = _load(registry, f"calibration_subject_{subject_id}")
        if cal is not None:
            self.calibration[subject_id] = cal
            self.calibration_version[subject_id] = ver  # type: ignore[assignment]
        return cal


@lru_cache(maxsize=1)
def get_model_bundle() -> ModelBundle:
    bundle = ModelBundle()
    bundle.load_all(get_registry(), default_subject=get_cached_settings().held_out_subject)
    return bundle


def reset_model_bundle() -> None:
    """Drop cached models — useful after a new registry write."""
    get_model_bundle.cache_clear()
    get_registry.cache_clear()
