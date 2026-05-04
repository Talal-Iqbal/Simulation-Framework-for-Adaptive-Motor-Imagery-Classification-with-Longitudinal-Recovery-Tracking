"""Feature extraction adapters.

Re-exports the user-provided extractors from `feature_extractor.py` (project
root) so the rest of the package depends on a stable, internal import path.
"""

from .extractors import FeatureExtractor, RawQualityExtractor, build_raw_quality_extractor

__all__ = ["FeatureExtractor", "RawQualityExtractor", "build_raw_quality_extractor"]
