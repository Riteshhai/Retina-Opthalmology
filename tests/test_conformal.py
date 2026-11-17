"""Synthetic probability fixtures verify software, not retinal model performance."""
import json
import math

import numpy as np
import pytest

from qsart.uncertainty import (
    conformal_quantile,
    coverage_metrics,
    fit_conformal,
    load_conformal,
    nonconformity_scores,
    predict_sets,
    save_conformal,
    validate_artifact,
)


def test_exact_finite_sample_rank_and_infinite_small_sample():
    scores = np.arange(10) / 10
    # k=ceil(11*.8)=9, so ninth element .8; numpy k/n higher returns .9.
    assert conformal_quantile(scores, .2) == .8
    assert conformal_quantile([.2, .2, .8, .9], .4) == .8
    assert math.isinf(conformal_quantile([.1, .2], .1))
    artifact = fit_conformal([[.8, .2], [.7, .3]], [0, 0], alpha=.1)
    assert artifact["qhat"] is None
    assert artifact["rank"] == 3
    assert predict_sets([[.99, .01]], artifact) == [[0, 1]]


@pytest.mark.parametrize("grouped", [False, True])
def test_decimal_alpha_integer_rank_boundary_survives_artifact_roundtrip(tmp_path, grouped):
    scores = np.arange(1, 10) / 10
    # Binary float arithmetic gives 10*(1-.7)=3.0000000000000004. The
    # declared decimal alpha instead requires the third score, not the fourth.
    assert conformal_quantile(scores, .7) == .3
    probabilities = np.column_stack((1 - scores, scores))
    groups = None
    if grouped:
        probabilities = np.repeat(probabilities, 2, axis=0)
        groups = np.repeat(np.arange(9), 2)
    artifact = fit_conformal(probabilities, np.zeros(len(probabilities), dtype=int),
                             alpha=.7, groups=groups, empty_policy="allow")
    assert artifact["n_calibration_units"] == 9
    assert artifact["rank"] == 3
    assert artifact["qhat"] == pytest.approx(.3)
    path = tmp_path / "boundary.json"
    save_conformal(path, artifact)
    assert load_conformal(path) == artifact
    with pytest.raises(ValueError, match="rank"):
        validate_artifact({**artifact, "rank": 4})


@pytest.mark.parametrize("alpha", [0, 1, -.01, float("nan"), float("inf"), True, ".1"])
def test_invalid_alpha_is_not_accepted(alpha):
    with pytest.raises(ValueError):
        conformal_quantile([.1, .2], alpha)


def test_inclusive_ties_and_conservative_empty_fallback():
    artifact = fit_conformal([[.8, .2]] * 9, [0] * 9, alpha=.5, empty_policy="allow")
    assert predict_sets([[.8, .2]], artifact) == [[0]]
    assert predict_sets([[.5, .5]], artifact) == [[]]
    nonempty = {**artifact, "empty_policy": "argmax"}
    assert predict_sets([[.5, .5]], nonempty) == [[0]]
    assert nonempty["qhat"] == artifact["qhat"]


def test_lac_can_return_disconnected_sets():
    artifact = fit_conformal([[.5, .1, .4]] * 9, [2] * 9, alpha=.5, empty_policy="allow")
    assert predict_sets([[.5, .1, .4]], artifact) == [[0, 2]]


def test_ordinal_greedy_order_adjacent_ties_and_modal_ties():
    probability = [[.15, .2, .3, .2, .15]]
    scores = nonconformity_scores(probability, method="ordinal_greedy")
    # mode2, right3 wins tie, then1, right4 wins tie, finally0.
    np.testing.assert_allclose(scores, [[1., .7, .3, .5, .85]])
    tied = nonconformity_scores([[.4, .4, .1, .1, 0]], method="ordinal_greedy")
    np.testing.assert_allclose(tied, [[.4, .8, .9, 1., 1.]])


def test_ordinal_inversion_matches_score_event_and_is_contiguous():
    rng = np.random.default_rng(6020)
    calibration = rng.dirichlet(np.ones(5), size=100)
    labels = rng.integers(0, 5, size=100)
    artifact = fit_conformal(calibration, labels, alpha=.1, method="ordinal_greedy", empty_policy="allow")
    evaluation = rng.dirichlet(np.ones(5), size=200)
    scores = nonconformity_scores(evaluation, method="ordinal_greedy")
    prediction_sets = predict_sets(evaluation, artifact)
    for row, values in enumerate(prediction_sets):
        assert values == np.flatnonzero(scores[row] <= artifact["qhat"]).tolist()
        assert not values or values == list(range(values[0], values[-1] + 1))
    metrics = coverage_metrics(prediction_sets, rng.integers(0, 5, size=200))
    assert sum(metrics["set_size_counts"].values()) == 200
    assert metrics["contiguous_nonempty_fraction"] == 1 - metrics["empty_set_count"] / 200


def test_exact_exchangeable_leave_one_out_coverage_event():
    # Exhaust all possible held-out positions, with the scoring model fixed.
    # This is a finite population software check, not an empirical clinical claim.
    probabilities = np.column_stack([1 - np.linspace(.05, .95, 11), np.linspace(.05, .95, 11)])
    successes = 0
    for held_out in range(11):
        train = np.arange(11) != held_out
        artifact = fit_conformal(probabilities[train], np.zeros(10, dtype=int), alpha=.2,
                                 empty_policy="allow")
        successes += 0 in predict_sets(probabilities[[held_out]], artifact)[0]
    assert successes == 9
    assert successes / 11 >= .8


def test_group_max_calibration_uses_patient_denominator():
    scores = np.array([.1, .2, .2, .3, .3, .4, .4, .9])
    probabilities = np.column_stack([1 - scores, scores])
    groups = ["p1", "p1", "p2", "p2", "p3", "p3", "p4", "p4"]
    artifact = fit_conformal(probabilities, np.zeros(8, dtype=int), alpha=.4, groups=groups)
    assert artifact["unit"] == "patient_max"
    assert artifact["n_images"] == 8 and artifact["n_calibration_units"] == 4
    assert artifact["rank"] == 3
    assert artifact["qhat"] == pytest.approx(.4)
    assert artifact["coverage_scope"] == "exchangeable_patient_all_views"
    assert "p1" not in json.dumps(artifact)
    # With one independent cluster and strict alpha, infinity is required.
    single = fit_conformal(probabilities, np.zeros(8, dtype=int), alpha=.1, groups=["p1"] * 8)
    assert single["qhat"] is None


def test_patient_simultaneous_coverage_differs_from_image_marginal():
    metrics = coverage_metrics([[0], [0], [1], [0]], [0, 0, 1, 1], num_classes=3,
                               groups=["a", "a", "b", "b"])
    assert metrics["marginal_coverage"] == .75
    assert metrics["patient_all_views_coverage"] == .5
    assert metrics["n_patients"] == 2
    assert metrics["per_class"]["2"] == {"n": 0, "coverage": None}


@pytest.mark.parametrize("probabilities", [[], [[.8, .8]], [[-.1, 1.1]], [[float("nan"), 0]],
                                           [[float("inf"), 0]], [[1]], [.8, .2]])
def test_invalid_probability_matrices(probabilities):
    with pytest.raises(ValueError):
        fit_conformal(probabilities, [0], alpha=.1)


@pytest.mark.parametrize("labels", [[-1], [2], [.5], [0.0], [True], ["0"], [0, 1]])
def test_labels_are_integer_class_indices(labels):
    with pytest.raises(ValueError):
        fit_conformal([[.8, .2]], labels)


@pytest.mark.parametrize("groups", [[], [None], [float("nan")], [""], "patient", [True]])
def test_invalid_or_missing_patient_identifiers(groups):
    with pytest.raises(ValueError):
        fit_conformal([[.8, .2]], [0], groups=groups)


def test_deterministic_safe_json_roundtrip_and_provenance_binding(tmp_path):
    metadata = {"dr_checkpoint_sha256": "a" * 64, "temperature_sha256": "b" * 64,
                "split_sha256": "c" * 64, "probability_path": "first_pass"}
    artifact = fit_conformal([[.8, .2]] * 12, [0] * 12, method="ordinal_greedy", metadata=metadata)
    first, second = tmp_path / "first.json", tmp_path / "second.json"
    save_conformal(first, artifact)
    loaded = load_conformal(first, expected_metadata=metadata)
    save_conformal(second, loaded)
    assert first.read_bytes() == second.read_bytes()
    assert predict_sets([[.3, .7]], loaded) == predict_sets([[.3, .7]], artifact)
    with pytest.raises(ValueError, match="provenance mismatch"):
        load_conformal(first, expected_metadata={"temperature_sha256": "d" * 64})
    with pytest.raises(ValueError, match="provenance mismatch"):
        load_conformal(first, expected_metadata={"missing_binding": "missing"})
    for key, invalid in [("qhat", float("nan")), ("rank", 4), ("n_images", True),
                         ("method", "unknown"), ("coverage_scope", "clinical_guarantee"),
                         ("ordinal_randomization", "randomized"), ("extra", "field")]:
        with pytest.raises(ValueError):
            validate_artifact({**artifact, key: invalid})


def test_json_nonstandard_numbers_and_duplicate_fields_rejected(tmp_path):
    path = tmp_path / "bad.json"
    for contents in ['{"qhat":NaN}', '{"schema":"x","schema":"y"}', '{"qhat":Infinity}']:
        path.write_text(contents, encoding="utf-8")
        with pytest.raises(ValueError):
            load_conformal(path)


def test_mismatching_class_count_and_invalid_sets_are_rejected():
    artifact = fit_conformal([[.8, .2]] * 10, [0] * 10)
    with pytest.raises(ValueError, match="columns"):
        predict_sets([[.3, .3, .4]], artifact)
    with pytest.raises(ValueError):
        coverage_metrics([[0, 0]], [0])
    with pytest.raises(ValueError):
        coverage_metrics([[5]], [0])
    with pytest.raises(ValueError):
        fit_conformal([[.8, .2]], [0], method="convex_hull")
    with pytest.raises(ValueError):
        fit_conformal([[.8, .2]], [0], metadata={"temperature": float("nan")})
