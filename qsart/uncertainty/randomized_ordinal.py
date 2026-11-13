"""Explicit-randomization ordinal conformal scores for development research.

Independently expresses the greedy adjacent-label construction in Hasan et al.
(2026), DOI 10.3390/diagnostics16162645. One supplied uniform per observation is
shared across all candidate labels: score = previous mass + (1-u)*p_candidate.
No random generator is created or advanced here. The caller must supply uniforms
independently of labels and model probabilities and keep calibration/evaluation
streams separate. A fixed seed permits replay; it is not another training seed.

This module does not change the production conformal sidecar or its schema.
Patient-max fitting targets simultaneous inclusion of all supplied patient views
only under exchangeable, independently sampled clusters and an identical view
sampling/randomization protocol. Image-pooled fitting of correlated patient views
has no justified iid-image guarantee. Neither construction guarantees coverage
after routing, under shift, or within each disease grade.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from qsart.uncertainty.conformal import (
    _alpha,
    _conformal_rank,
    _groups,
    _labels,
    _metadata,
    _probabilities,
    conformal_quantile,
)

_SCHEMA = "qsart.randomized_ordinal_research.v1"
_METHOD = "ordinal_greedy_randomized"
_TIES = "lowest_modal_grade;higher_adjacent_grade_on_equal_probability;inclusive_threshold"
_RANDOMIZATION = "one_explicit_uniform_per_observation_shared_across_candidate_labels"
_KEYS = {
    "schema", "method", "alpha", "num_classes", "n_images", "n_calibration_units",
    "unit", "rank", "qhat", "tie_policy", "randomization", "coverage_scope", "metadata",
}


def _uniforms(values: Any, n: int) -> np.ndarray:
    try:
        values = np.asarray(values)
        if values.dtype.kind not in "iuf":
            raise ValueError("Uniforms must be numeric")
        uniforms = values.astype(np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Uniforms must be an explicit numeric vector") from exc
    if uniforms.shape != (n,):
        raise ValueError("Uniforms require exactly one value per observation")
    if not np.isfinite(uniforms).all() or (uniforms < 0).any() or (uniforms > 1).any():
        raise ValueError("Uniforms must be finite and in [0,1]")
    return uniforms


def randomized_scores(probabilities: Any, uniforms: Any) -> np.ndarray:
    """Return N-by-K scores; a row uses the same supplied u for every grade.

    Adjacent greedy ordering starts at the lowest-index mode, and an equal
    adjacent probability selects the higher grade. Scores are monotone in that
    ordering, so inclusive threshold inversion produces an adjacent interval or
    an empty set. The stable previous-mass formula equals cumulative_mass-u*p.
    u=0 recovers deterministic full mass; u=1 recovers preceding mass.
    """
    probabilities = _probabilities(probabilities)
    uniforms = _uniforms(uniforms, len(probabilities))
    scores = np.empty_like(probabilities)
    for row_index, (row, uniform) in enumerate(zip(probabilities, uniforms, strict=True)):
        lower = upper = chosen = int(np.argmax(row))
        mass = 0.0
        while True:
            scores[row_index, chosen] = min(mass + (1.0 - uniform) * float(row[chosen]), 1.0)
            mass = min(mass + float(row[chosen]), 1.0)
            if lower == 0 and upper == len(row) - 1:
                break
            left, right = lower - 1, upper + 1
            if left < 0:
                chosen = right
            elif right >= len(row):
                chosen = left
            else:
                chosen = right if row[right] >= row[left] else left
            lower, upper = min(lower, chosen), max(upper, chosen)
    return scores


def fit_randomized(probabilities: Any, labels: Any, uniforms: Any, *, alpha: float = .1,
                   groups: Any = None, metadata: dict | None = None) -> dict:
    """Fit the exact conformal threshold using explicit calibration uniforms.

    Groups aggregate maximum true-label scores within patients. No label or
    patient identifier is stored, unless a caller improperly puts it in metadata.
    The supplied uniforms and their generation/provenance belong to the caller.
    """
    probabilities = _probabilities(probabilities)
    labels = _labels(labels, len(probabilities), probabilities.shape[1])
    alpha = _alpha(alpha)
    scores = randomized_scores(probabilities, uniforms)[np.arange(len(labels)), labels]
    if groups is not None:
        identifiers = _groups(groups, len(labels))
        maxima = {}
        for identifier, score in zip(identifiers, scores, strict=True):
            maxima[identifier] = max(maxima.get(identifier, -math.inf), float(score))
        scores = np.asarray(list(maxima.values()), dtype=np.float64)
    qhat = conformal_quantile(scores, alpha)
    artifact = {
        "schema": _SCHEMA, "method": _METHOD, "alpha": alpha,
        "num_classes": int(probabilities.shape[1]), "n_images": len(labels),
        "n_calibration_units": len(scores), "unit": "patient_max" if groups is not None else "image",
        "rank": _conformal_rank(len(scores), alpha), "qhat": None if math.isinf(qhat) else qhat,
        "tie_policy": _TIES, "randomization": _RANDOMIZATION,
        "coverage_scope": "exchangeable_patient_all_views" if groups is not None else "exchangeable_image",
        "metadata": _metadata({} if metadata is None else metadata),
    }
    return validate_randomized(artifact)


def validate_randomized(artifact: Any, *, expected_metadata: dict | None = None) -> dict:
    """Validate the research-only JSON schema and optional exact provenance."""
    if not isinstance(artifact, dict) or set(artifact) != _KEYS:
        raise ValueError("Unrecognized randomized ordinal artifact fields")
    if artifact["schema"] != _SCHEMA or artifact["method"] != _METHOD:
        raise ValueError("Unsupported randomized ordinal schema or method")
    alpha = _alpha(artifact["alpha"])
    for key in ("num_classes", "n_images", "n_calibration_units", "rank"):
        if type(artifact[key]) is not int or artifact[key] < 1:
            raise ValueError(f"Invalid randomized ordinal artifact {key}")
    n = artifact["n_calibration_units"]
    if artifact["num_classes"] < 2 or n > artifact["n_images"]:
        raise ValueError("Invalid randomized ordinal class or sample counts")
    if artifact["rank"] != _conformal_rank(n, alpha):
        raise ValueError("Conformal rank does not match sample count and alpha")
    qhat = artifact["qhat"]
    if qhat is None:
        if artifact["rank"] <= n:
            raise ValueError("Infinite threshold requires rank greater than calibration units")
    elif (isinstance(qhat, bool) or not isinstance(qhat, (int, float)) or not math.isfinite(qhat)
          or not 0 <= qhat <= 1 or artifact["rank"] > n):
        raise ValueError("Invalid randomized ordinal threshold")
    if not isinstance(artifact["unit"], str) or artifact["unit"] not in {"image", "patient_max"}:
        raise ValueError("Unknown calibration unit")
    scope = "exchangeable_patient_all_views" if artifact["unit"] == "patient_max" else "exchangeable_image"
    if artifact["coverage_scope"] != scope:
        raise ValueError("Coverage scope does not match calibration unit")
    if artifact["unit"] == "image" and n != artifact["n_images"]:
        raise ValueError("Image calibration count mismatch")
    if artifact["tie_policy"] != _TIES or artifact["randomization"] != _RANDOMIZATION:
        raise ValueError("Randomized ordinal construction differs from supported implementation")
    metadata = _metadata(artifact["metadata"])
    if expected_metadata is not None:
        for key, value in _metadata(expected_metadata).items():
            if key not in metadata or metadata[key] != value:
                raise ValueError(f"Randomized ordinal provenance mismatch: {key}")
    return {**artifact, "alpha": alpha, "metadata": metadata}


def predict_randomized(probabilities: Any, artifact: dict, uniforms: Any, *,
                       empty_policy: str = "argmax") -> list[list[int]]:
    """Invert explicit evaluation scores, with optional conservative modal fallback.

    Adding the mode when the raw set is empty cannot decrease coverage. Report
    raw empty counts separately; this enlargement changes score-event equality.
    """
    artifact = validate_randomized(artifact)
    if not isinstance(empty_policy, str) or empty_policy not in {"argmax", "allow"}:
        raise ValueError("Unknown empty-set policy")
    probabilities = _probabilities(probabilities)
    if probabilities.shape[1] != artifact["num_classes"]:
        raise ValueError("Probability columns differ from fitted class count")
    threshold = math.inf if artifact["qhat"] is None else artifact["qhat"]
    included = randomized_scores(probabilities, uniforms) <= threshold
    if empty_policy == "argmax":
        empty = ~included.any(axis=1)
        included[np.flatnonzero(empty), np.argmax(probabilities[empty], axis=1)] = True
    return [np.flatnonzero(row).tolist() for row in included]


def save_randomized(path: str | Path, artifact: dict) -> None:
    """Write safe deterministic JSON; caller controls destination/overwrite policy."""
    artifact = validate_randomized(artifact)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def load_randomized(path: str | Path, *, expected_metadata: dict | None = None) -> dict:
    def reject_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject_constant(value):
        raise ValueError(f"Non-finite JSON constant: {value}")

    artifact = json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=reject_duplicates,
                          parse_constant=reject_constant)
    return validate_randomized(artifact, expected_metadata=expected_metadata)
