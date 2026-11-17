"""Small supervised retinal classifiers. No weights download at inference time."""

from .networks import RetinalClassifier, build_model

__all__ = ["RetinalClassifier", "build_model"]
