"""Data ingestion, splitting, and synthetic fixture utilities."""

from .ingest import load_held_out_subject, load_subjects
from .splits import build_acceptance_dataset, train_test_split_indices
from .synthetic import make_synthetic_epochs

__all__ = [
    "load_subjects",
    "load_held_out_subject",
    "build_acceptance_dataset",
    "train_test_split_indices",
    "make_synthetic_epochs",
]
