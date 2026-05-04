"""Local model registry built on joblib + manifest.json."""

from .store import ModelRegistry, RegistryEntry

__all__ = ["ModelRegistry", "RegistryEntry"]
