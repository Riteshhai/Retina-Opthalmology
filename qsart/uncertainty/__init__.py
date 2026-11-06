"""Set-valued uncertainty with explicit calibration and provenance boundaries."""

from .conformal import (
    conformal_quantile,
    coverage_metrics,
    fit_conformal,
    load_conformal,
    nonconformity_scores,
    predict_sets,
    save_conformal,
    validate_artifact,
)

__all__ = [
    "conformal_quantile", "coverage_metrics", "fit_conformal", "load_conformal",
    "nonconformity_scores", "predict_sets", "save_conformal", "validate_artifact",
]
