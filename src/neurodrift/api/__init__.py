"""FastAPI service exposing NeuroDrift predictions and analytics."""

from .main import create_app

__all__ = ["create_app"]
