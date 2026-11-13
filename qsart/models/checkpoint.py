"""Validated, CPU-safe state-dictionary checkpoints; no pickled model objects."""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any

import torch

from .networks import build_model

FORMAT_VERSION = 1


class ArtifactError(RuntimeError):
    """A required supervised artifact is missing, invalid, or incompatible."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_checkpoint(path: str | Path, *, allow_software_test: bool = False
                    ) -> tuple[torch.nn.Module, dict[str, Any]]:
    path = Path(path)
    if not path.is_file():
        raise ArtifactError(f"Trained checkpoint not found: {path.name}")
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ArtifactError(f"Cannot read checkpoint {path.name}: {type(exc).__name__}") from exc
    if not isinstance(checkpoint, dict) or checkpoint.get("format_version") != FORMAT_VERSION:
        raise ArtifactError("Unsupported checkpoint schema")
    metadata = checkpoint.get("metadata", {})
    if not isinstance(metadata, dict) or metadata.get("trained") is not True:
        raise ArtifactError("Checkpoint has no verified supervised-training marker")
    purpose = metadata.get("data_purpose")
    if purpose not in {"real_research", "software_test"}:
        raise ArtifactError("Checkpoint data purpose is missing or unsupported")
    if purpose == "software_test" and not allow_software_test:
        raise ArtifactError("Software-test checkpoints are disabled for production inference")
    if int(metadata.get("optimizer_steps", 0)) < 1:
        raise ArtifactError("Checkpoint contains no completed optimizer updates")
    try:
        model = build_model(metadata["backbone"], metadata["heads"], pretrained=False)
        model.load_state_dict(checkpoint["model_state"], strict=True)
        if not all(torch.isfinite(tensor).all() for tensor in model.state_dict().values()):
            raise ValueError("Nonfinite model weights")
        for field in ("image_size", "normalization_mean", "normalization_std", "manifest_sha256",
                      "split_fingerprints", "model_version", "task"):
            if field not in metadata:
                raise ValueError(f"Missing {field}")
        if int(metadata["image_size"]) < 32:
            raise ValueError("Image size must be at least 32")
        if len(metadata["normalization_mean"]) != 3 or len(metadata["normalization_std"]) != 3:
            raise ValueError("RGB normalization requires three values")
        if not all(math.isfinite(float(value)) for value in
                   metadata["normalization_mean"] + metadata["normalization_std"]):
            raise ValueError("Normalization parameters must be finite")
        if any(float(value) <= 0 for value in metadata["normalization_std"]):
            raise ValueError("Normalization standard deviations must be positive")
    except Exception as exc:
        raise ArtifactError(f"Checkpoint metadata or state is invalid: {exc}") from exc
    model.eval()
    return model, checkpoint
