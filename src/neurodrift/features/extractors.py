"""Thin wrapper around the user-provided `feature_extractor.py`."""

from __future__ import annotations

from collections.abc import Sequence

from feature_extractor import FeatureExtractor, RawQualityExtractor

from ..config import BASELINE_SAMPLES, CH_NAMES, SFREQ


def build_raw_quality_extractor(
    ch_names: Sequence[str] | None = None,
    sfreq: float = SFREQ,
    baseline_samples: int = BASELINE_SAMPLES,
) -> RawQualityExtractor:
    """Construct the 8-d raw-quality extractor used by the acceptance gate."""
    return RawQualityExtractor(
        ch_names=list(ch_names) if ch_names is not None else list(CH_NAMES),
        sfreq=sfreq,
        baseline_samples=baseline_samples,
    )


__all__ = ["FeatureExtractor", "RawQualityExtractor", "build_raw_quality_extractor"]
