"""Deterministic LAC and greedy contiguous ordinal split conformal predictors.

The ordinal ordering follows Algorithm 1 of Hasan et al. (2026), DOI
10.3390/diagnostics16162645, but uses full cumulative mass (their u=0) instead
of randomized mass. This is a documented deterministic adaptation, not an exact
replication. Both methods use an exact finite-sample order statistic. Optional
patient-max calibration targets simultaneous inclusion of all views of a future
exchangeable patient cluster, rather than treating correlated views as iid.

These constructions do not promise grade-conditional or shifted-domain coverage.
Classifier fitting, probability calibration and method/alpha selection must be
independent of the observations used to fit this conformal threshold. Callers
must bind and verify checkpoint, probability-transform and split identities in
metadata; this module cannot infer those identities from probability arrays.
"""
from __future__ import annotations

from decimal import Decimal, ROUND_CEILING
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

_SCHEMA = "qsart.split_conformal.v1"
_METHODS = {"lac", "ordinal_greedy"}
_TIES = "lowest_modal_grade;higher_adjacent_grade_on_equal_probability;inclusive_threshold"
_KEYS = {
    "schema", "method", "alpha", "num_classes", "n_images", "n_calibration_units",
    "unit", "rank", "qhat", "empty_policy", "tie_policy", "ordinal_randomization",
    "coverage_scope", "metadata",
}


def _alpha(value: Any) -> float:
    if isinstance(value, (bool, str)):
        raise ValueError("alpha must be a finite number strictly between zero and one")
    try:
        alpha = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Invalid alpha") from exc
    if not math.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must be a finite number strictly between zero and one")
    return alpha


def _probabilities(values: Any) -> np.ndarray:
    try:
        probabilities = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Probabilities must be a numeric matrix") from exc
    if probabilities.ndim != 2 or probabilities.shape[0] < 1 or probabilities.shape[1] < 2:
        raise ValueError("Probabilities require nonempty rows and at least two ordered classes")
    if not np.isfinite(probabilities).all() or (probabilities < 0).any() or (probabilities > 1).any():
        raise ValueError("Probabilities must be finite and in [0,1]")
    totals = probabilities.sum(axis=1)
    if not np.allclose(totals, 1.0, atol=1e-6, rtol=0):
        raise ValueError("Each probability row must sum to one")
    # Only floating-point normalization error is corrected, not arbitrary scores.
    return probabilities / totals[:, None]


def _labels(values: Any, n: int, num_classes: int) -> np.ndarray:
    labels = np.asarray(values)
    if labels.shape != (n,) or labels.dtype.kind not in "iu":
        raise ValueError("Labels must be a matching vector of integer class indices")
    if (labels < 0).any() or (labels >= num_classes).any():
        raise ValueError("Label outside the ordered class range")
    return labels.astype(np.int64)


def _groups(values: Any, n: int) -> list[str | int]:
    if isinstance(values, (str, bytes)):
        raise ValueError("Groups must contain one patient identifier per image")
    try:
        result = list(values)
    except TypeError as exc:
        raise ValueError("Groups must contain one patient identifier per image") from exc
    if len(result) != n:
        raise ValueError("Groups must contain one patient identifier per image")
    normalized = []
    for value in result:
        if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
            normalized.append(int(value))
        elif isinstance(value, str) and value.strip():
            normalized.append(value)
        else:
            raise ValueError("Patient identifiers must be nonempty strings or integers")
    return normalized


def _metadata(value: Any) -> dict:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError("Metadata must be a JSON object with string keys")
    try:
        encoded = json.dumps(value, allow_nan=False, sort_keys=True)
        return json.loads(encoded)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Metadata must contain finite JSON values") from exc


def _conformal_rank(n: int, alpha: float) -> int:
    # Treat the declared decimal alpha consistently with integer budget limits.
    # Binary rounding can otherwise turn an exact integer rank into its successor.
    value = Decimal(n + 1) * (Decimal(1) - Decimal(str(alpha)))
    return int(value.to_integral_value(rounding=ROUND_CEILING))


def conformal_quantile(scores: Any, alpha: float) -> float:
    """Return the kth score, k=ceil((n+1)*(1-alpha)); infinity if k>n.

    This deliberately uses the kth order statistic, not an interpolated quantile
    or np.quantile(scores, k/n, method='higher'), which can select the next rank.
    Scores equal to the threshold are included, so ties are conservative.
    """
    alpha = _alpha(alpha)
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 1 or not len(scores) or not np.isfinite(scores).all():
        raise ValueError("Calibration requires a nonempty vector of finite scores")
    rank = _conformal_rank(len(scores), alpha)
    return math.inf if rank > len(scores) else float(np.partition(scores, rank - 1)[rank - 1])


def _ordinal_scores(probabilities: np.ndarray) -> np.ndarray:
    scores = np.empty_like(probabilities)
    for row_index, row in enumerate(probabilities):
        lower = upper = int(np.argmax(row))
        mass = float(row[lower])
        scores[row_index, lower] = mass
        while lower > 0 or upper < len(row) - 1:
            left, right = lower - 1, upper + 1
            if left < 0:
                chosen = right
            elif right >= len(row):
                chosen = left
            else:
                # Matches the public paper code's right-neighbor tie convention.
                chosen = right if row[right] >= row[left] else left
            mass += float(row[chosen])
            scores[row_index, chosen] = min(mass, 1.0)
            lower, upper = min(lower, chosen), max(upper, chosen)
    return scores


def nonconformity_scores(probabilities: Any, *, method: str = "lac") -> np.ndarray:
    """Return n-by-K candidate-label scores, independent of the true labels."""
    probabilities = _probabilities(probabilities)
    if method == "lac":
        return 1 - probabilities
    if method == "ordinal_greedy":
        return _ordinal_scores(probabilities)
    raise ValueError(f"Unknown conformal method: {method}")


def fit_conformal(probabilities: Any, labels: Any, *, alpha: float = 0.1,
                  method: str = "lac", groups: Any = None,
                  metadata: dict | None = None, empty_policy: str = "argmax") -> dict:
    """Fit only a score threshold; no class labels or patient IDs enter the artifact.

    With groups, each calibration unit is the maximum true-label score across
    one patient's supplied views. The simultaneous event requires independent,
    exchangeable complete patient clusters under the same view-sampling protocol.
    Without groups, the usual exchangeable-image marginal scope is recorded.
    Forcing an empty set to contain the modal grade only adds labels and hence
    cannot reduce the coverage event. Use empty_policy='allow' for raw inversion.
    """
    probabilities = _probabilities(probabilities)
    labels = _labels(labels, len(probabilities), probabilities.shape[1])
    alpha = _alpha(alpha)
    scores = nonconformity_scores(probabilities, method=method)[np.arange(len(labels)), labels]
    if groups is not None:
        identifiers = _groups(groups, len(labels))
        maxima = {}
        for identifier, score in zip(identifiers, scores, strict=True):
            maxima[identifier] = max(maxima.get(identifier, -math.inf), float(score))
        scores = np.asarray(list(maxima.values()))
    qhat = conformal_quantile(scores, alpha)
    artifact = {
        "schema": _SCHEMA, "method": method, "alpha": alpha,
        "num_classes": int(probabilities.shape[1]), "n_images": int(len(labels)),
        "n_calibration_units": int(len(scores)), "unit": "patient_max" if groups is not None else "image",
        "rank": _conformal_rank(len(scores), alpha),
        # JSON null is explicitly +infinity, avoiding nonstandard Infinity/NaN.
        "qhat": None if math.isinf(qhat) else qhat,
        "empty_policy": empty_policy, "tie_policy": _TIES,
        "ordinal_randomization": "none_full_mass" if method == "ordinal_greedy" else "not_applicable",
        "coverage_scope": "exchangeable_patient_all_views" if groups is not None else "exchangeable_image",
        "metadata": _metadata({} if metadata is None else metadata),
    }
    return validate_artifact(artifact)


def validate_artifact(artifact: Any, *, expected_metadata: dict | None = None) -> dict:
    """Validate safe JSON contents and optional exact provenance field matches.

    No executable serialization is accepted. Expected metadata may be a subset
    of artifact metadata, allowing a caller to enforce all its required hashes.
    """
    if not isinstance(artifact, dict) or set(artifact) != _KEYS:
        raise ValueError("Unrecognized conformal artifact fields")
    if artifact["schema"] != _SCHEMA or artifact["method"] not in _METHODS:
        raise ValueError("Unsupported conformal schema or method")
    alpha = _alpha(artifact["alpha"])
    for key in ("num_classes", "n_images", "n_calibration_units", "rank"):
        if type(artifact[key]) is not int or artifact[key] < 1:
            raise ValueError(f"Invalid conformal artifact {key}")
    if artifact["num_classes"] < 2 or artifact["n_calibration_units"] > artifact["n_images"]:
        raise ValueError("Invalid conformal class or sample counts")
    n = artifact["n_calibration_units"]
    if artifact["rank"] != _conformal_rank(n, alpha):
        raise ValueError("Conformal rank does not match sample count and alpha")
    qhat = artifact["qhat"]
    if qhat is None:
        if artifact["rank"] <= n:
            raise ValueError("Infinite threshold requires rank greater than calibration units")
    elif (isinstance(qhat, bool) or not isinstance(qhat, (int, float)) or not math.isfinite(qhat)
          or not 0 <= qhat <= 1 or artifact["rank"] > n):
        raise ValueError("Invalid conformal threshold")
    if artifact["unit"] not in {"image", "patient_max"}:
        raise ValueError("Unknown calibration unit")
    scope = "exchangeable_patient_all_views" if artifact["unit"] == "patient_max" else "exchangeable_image"
    if artifact["coverage_scope"] != scope:
        raise ValueError("Coverage scope does not match calibration unit")
    if artifact["unit"] == "image" and n != artifact["n_images"]:
        raise ValueError("Image calibration count mismatch")
    randomization = "none_full_mass" if artifact["method"] == "ordinal_greedy" else "not_applicable"
    if artifact["ordinal_randomization"] != randomization or artifact["tie_policy"] != _TIES:
        raise ValueError("Conformal construction does not match supported implementation")
    if artifact["empty_policy"] not in {"argmax", "allow"}:
        raise ValueError("Unknown empty-set policy")
    metadata = _metadata(artifact["metadata"])
    if expected_metadata is not None:
        for key, value in _metadata(expected_metadata).items():
            if key not in metadata or metadata[key] != value:
                raise ValueError(f"Conformal provenance mismatch: {key}")
    return {**artifact, "metadata": metadata}


def predict_sets(probabilities: Any, artifact: dict) -> list[list[int]]:
    """Invert the fitted score with inclusive ties; return ascending grade lists."""
    artifact = validate_artifact(artifact)
    probabilities = _probabilities(probabilities)
    if probabilities.shape[1] != artifact["num_classes"]:
        raise ValueError("Probability columns differ from fitted class count")
    threshold = math.inf if artifact["qhat"] is None else artifact["qhat"]
    included = nonconformity_scores(probabilities, method=artifact["method"]) <= threshold
    if artifact["empty_policy"] == "argmax":
        empty = ~included.any(axis=1)
        included[np.flatnonzero(empty), np.argmax(probabilities[empty], axis=1)] = True
    return [np.flatnonzero(row).tolist() for row in included]


def save_conformal(path: str | Path, artifact: dict) -> None:
    """Write deterministic standards-compliant JSON; callers control overwrite policy."""
    validated = validate_artifact(artifact)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(validated, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def load_conformal(path: str | Path, *, expected_metadata: dict | None = None) -> dict:
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
    return validate_artifact(artifact, expected_metadata=expected_metadata)


def coverage_metrics(sets: Any, labels: Any, *, num_classes: int = 5, groups: Any = None) -> dict:
    """Empirical inclusion, grade denominators, size distribution and cluster event.

    These descriptive outcomes do not validate exchangeability assumptions.
    Absent grades have coverage null; an empty set has size zero and is counted
    separately from nonempty contiguous intervals.
    """
    if type(num_classes) is not int or num_classes < 2:
        raise ValueError("At least two ordered classes required")
    sets = list(sets)
    labels = _labels(labels, len(sets), num_classes)
    if not len(labels):
        raise ValueError("Coverage evaluation needs at least one observation")
    valid_sets = []
    for values in sets:
        values = list(values)
        if any(not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_))
               or not 0 <= value < num_classes for value in values):
            raise ValueError("Prediction sets require valid integer grades")
        if len(set(values)) != len(values):
            raise ValueError("Prediction sets cannot contain duplicate grades")
        valid_sets.append(sorted(int(value) for value in values))
    covered = np.asarray([int(label) in values for values, label in zip(valid_sets, labels, strict=True)])
    sizes = np.asarray([len(values) for values in valid_sets])
    contiguous = np.asarray([bool(values) and values[-1] - values[0] + 1 == len(values) for values in valid_sets])
    result = {
        "n_images": int(len(labels)), "marginal_coverage": float(covered.mean()),
        "mean_set_size": float(sizes.mean()), "empty_set_count": int((sizes == 0).sum()),
        "contiguous_nonempty_fraction": float(contiguous.mean()),
        "set_size_counts": {str(size): int((sizes == size).sum()) for size in range(num_classes + 1)},
        "per_class": {
            str(grade): {"n": int((labels == grade).sum()),
                         "coverage": float(covered[labels == grade].mean()) if (labels == grade).any() else None}
            for grade in range(num_classes)
        },
    }
    if groups is not None:
        identifiers = _groups(groups, len(labels))
        all_covered = {}
        for identifier, included in zip(identifiers, covered, strict=True):
            all_covered[identifier] = all_covered.get(identifier, True) and bool(included)
        result["n_patients"] = len(all_covered)
        result["patient_all_views_coverage"] = float(np.mean(list(all_covered.values())))
    return result
