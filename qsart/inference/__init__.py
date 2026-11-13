"""Artifact-backed CPU inference. Missing trained artifacts fail closed."""

from .engine import InferenceEngine

__all__ = ["InferenceEngine"]
