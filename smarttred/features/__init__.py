"""Feature engineering utilities for trading signals."""

from .feature_engine import FeatureEngine
from .target_generator import triple_barrier_labels

__all__ = ["FeatureEngine", "triple_barrier_labels"]
