"""Synthetic mathematics fixtures; no model-performance or clinical evidence."""
import json

import numpy as np
import pytest

from qsart.uncertainty.conformal import nonconformity_scores
from qsart.uncertainty.randomized_ordinal import (
    fit_randomized,
    load_randomized,
    predict_randomized,
    randomized_scores,
    save_randomized,
    validate_randomized,
)


def test_manual_shared_uniform_scores_and_tie_order():
    probabilities = [[.15, .2, .3, .2, .15]]
    # Ordering:2,3,1,4,0. Full mass minus .5 times the candidate mass.
    np.testing.assert_allclose(randomized_scores(probabilities, [.5]), [[.925, .6, .15, .4, .775]])
    tied = [[.4, .4, .1, .1, 0.]]
    np.testing.assert_allclose(randomized_scores(tied, [.5]), [[.2, .6, .85, .95, 1.]])


def test_boundary_uniforms_recover_full_and_preceding_mass():
    probabilities = [[.15, .2, .3, .2, .15], [.4, .4, .1, .1, 0.]]
    deterministic = nonconformity_scores(probabilities, method="ordinal_greedy")
    np.testing.assert_array_equal(randomized_scores(probabilities, [0, 0]), deterministic)
    np.testing.assert_allclose(randomized_scores(probabilities, [1, 1]),
                               [[.85, .5, 0., .3, .7], [0., .4, .8, .9, 1.]])


def test_zero_probability_and_near_boundary_scores_stay_nested():
    probabilities = [[0., 0., 1., 0., 0.], [1e-16, .1, .7, .2-1e-16, 0.]]
    for uniform in [0., np.nextafter(0., 1.), .5, np.nextafter(1., 0.), 1.]:
        scores = randomized_scores(probabilities, [uniform, uniform])
        assert (scores >= 0).all() and (scores <= 1).all()
        artifact = fit_randomized(probabilities * 5, [2] * 10, [uniform] * 10)
        for threshold in np.unique(scores):
            sets = predict_randomized(probabilities, {**artifact, "qhat": float(threshold)},
                                      [uniform, uniform], empty_policy="allow")
            assert all(not values or values == list(range(values[0], values[-1] + 1)) for values in sets)


def test_score_inversion_monotonicity_and_contiguity_for_all_thresholds():
    rng = np.random.default_rng(2651)
    probabilities = rng.dirichlet(np.ones(5), size=100)
    uniforms = rng.uniform(size=100)
    scores = randomized_scores(probabilities, uniforms)
    deterministic = nonconformity_scores(probabilities, method="ordinal_greedy")
    # Deterministic cumulative mass defines the same greedy label ordering.
    order = np.argsort(deterministic, axis=1, kind="stable")
    assert (np.diff(np.take_along_axis(scores, order, axis=1), axis=1) >= 0).all()
    artifact = fit_randomized(probabilities, rng.integers(0, 5, size=100), uniforms)
    for threshold in np.linspace(0, 1, 21):
        sets = predict_randomized(probabilities, {**artifact, "qhat": float(threshold)},
                                  uniforms, empty_policy="allow")
        for row, values in enumerate(sets):
            assert values == np.flatnonzero(scores[row] <= threshold).tolist()
            assert not values or values == list(range(values[0], values[-1] + 1))


def test_shared_uniforms_preserve_nesting_between_declared_alpha_levels():
    rng = np.random.default_rng(900)
    probabilities = rng.dirichlet(np.ones(5), size=84)
    labels = rng.integers(0, 5, size=84)
    uniforms = rng.uniform(size=84)
    groups = np.repeat(np.arange(21), 4)
    ninety = fit_randomized(probabilities, labels, uniforms, alpha=.1, groups=groups)
    ninety_five = fit_randomized(probabilities, labels, uniforms, alpha=.05, groups=groups)
    assert ninety["rank"] == 20 and ninety_five["rank"] == 21
    assert ninety["qhat"] <= ninety_five["qhat"]
    smaller = predict_randomized(probabilities, ninety, uniforms)
    larger = predict_randomized(probabilities, ninety_five, uniforms)
    assert all(set(first).issubset(second) for first, second in zip(smaller, larger, strict=True))


def test_exact_quantile_inclusive_ties_and_infinite_small_sample():
    probabilities = np.column_stack([np.linspace(.5, .95, 10), np.linspace(.5, .05, 10)])
    artifact = fit_randomized(probabilities, np.zeros(10, dtype=int), np.zeros(10), alpha=.2)
    assert artifact["rank"] == 9
    assert artifact["qhat"] == pytest.approx(.9)
    assert predict_randomized([[.9, .1]], artifact, [0], empty_policy="allow") == [[0]]
    # At declared decimal alpha .7, k=ceil(10*.3)=3, not binary-rounding rank4.
    boundary = fit_randomized([[.8, .2]] * 9, [0] * 9, [0] * 9, alpha=.7)
    assert boundary["rank"] == 3
    small = fit_randomized([[.8, .2]], [0], [.5], alpha=.1)
    assert small["qhat"] is None and small["rank"] == 2
    assert predict_randomized([[.9, .1]], small, [.5]) == [[0, 1]]


def test_patient_max_uses_maximum_after_randomization_and_not_average():
    probabilities = [[.8, .2]] * 8
    uniforms = [1., .9, .8, .7, .6, .5, .4, 0.]
    artifact = fit_randomized(probabilities, [0] * 8, uniforms, alpha=.4,
                              groups=["a", "a", "b", "b", "c", "c", "d", "d"])
    # Group maxima .08,.24,.4,.8 -> exact third is .4.
    assert artifact["rank"] == 3
    assert artifact["n_images"] == 8 and artifact["n_calibration_units"] == 4
    assert artifact["qhat"] == pytest.approx(.4)
    assert artifact["coverage_scope"] == "exchangeable_patient_all_views"
    assert '"a"' not in json.dumps(artifact)
    one_group = fit_randomized(probabilities, [0] * 8, uniforms, groups=["same"] * 8)
    assert one_group["qhat"] is None


def test_argmax_fallback_enlarges_empty_raw_sets():
    artifact = fit_randomized([[.8, .2]] * 9, [0] * 9, [1] * 9, alpha=.5)
    assert artifact["qhat"] == 0
    assert predict_randomized([[.5, .5]], artifact, [0], empty_policy="allow") == [[]]
    assert predict_randomized([[.5, .5]], artifact, [0]) == [[0]]
    with pytest.raises(ValueError, match="empty-set"):
        predict_randomized([[.5, .5]], artifact, [0], empty_policy="unknown")


def test_no_implicit_rng_dependence_or_input_mutation(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Randomness must be explicitly supplied")

    monkeypatch.setattr(np.random, "default_rng", forbidden)
    monkeypatch.setattr(np.random, "uniform", forbidden)
    probabilities = np.array([[.8, .2]] * 10)
    uniforms = np.linspace(0, 1, 10)
    original_p, original_u = probabilities.copy(), uniforms.copy()
    artifact = fit_randomized(probabilities, [0] * 10, uniforms)
    first = predict_randomized(probabilities, artifact, uniforms)
    assert first == predict_randomized(probabilities, artifact, uniforms)
    np.testing.assert_array_equal(probabilities, original_p)
    np.testing.assert_array_equal(uniforms, original_u)


@pytest.mark.parametrize("uniforms", [None, [], [.1, .2], [[.1]], [float("nan")],
                                     [float("inf")], [-.1], [1.1], [True], [".1"]])
def test_invalid_uniforms_rejected(uniforms):
    with pytest.raises(ValueError):
        randomized_scores([[.8, .2]], uniforms)


@pytest.mark.parametrize("probabilities", [[], [[.8, .8]], [[-.1, 1.1]], [[float("nan"), 0]],
                                           [[float("inf"), 0]], [[1]], [.8, .2]])
def test_invalid_probabilities_rejected(probabilities):
    with pytest.raises(ValueError):
        randomized_scores(probabilities, [.5])


@pytest.mark.parametrize("labels", [[-1], [2], [.5], [0.0], [True], ["0"], [0, 1]])
def test_labels_require_matching_integer_indices(labels):
    with pytest.raises(ValueError):
        fit_randomized([[.8, .2]], labels, [.5])


@pytest.mark.parametrize("groups", [[], [None], [float("nan")], [""], "patient", [True]])
def test_invalid_groups_rejected(groups):
    with pytest.raises(ValueError):
        fit_randomized([[.8, .2]], [0], [.5], groups=groups)


@pytest.mark.parametrize("alpha", [0, 1, -.01, float("nan"), float("inf"), True, ".1"])
def test_invalid_alpha_rejected(alpha):
    with pytest.raises(ValueError):
        fit_randomized([[.8, .2]], [0], [.5], alpha=alpha)


def test_safe_schema_and_provenance_roundtrip(tmp_path):
    metadata = {"checkpoint_sha256": "a" * 64, "uniform_seed": 2026, "probability_path": "first_pass"}
    artifact = fit_randomized([[.8, .2]] * 10, [0] * 10, [.5] * 10, metadata=metadata)
    first, second = tmp_path / "first.json", tmp_path / "second.json"
    save_randomized(first, artifact)
    loaded = load_randomized(first, expected_metadata=metadata)
    save_randomized(second, loaded)
    assert first.read_bytes() == second.read_bytes()
    assert loaded == artifact
    assert predict_randomized([[.3, .7]], loaded, [.5]) == predict_randomized([[.3, .7]], artifact, [.5])
    with pytest.raises(ValueError, match="provenance"):
        validate_randomized(artifact, expected_metadata={"uniform_seed": 2027})
    with pytest.raises(ValueError, match="provenance"):
        load_randomized(first, expected_metadata={"missing": "key"})
    with pytest.raises(ValueError, match="columns"):
        predict_randomized([[.3, .3, .4]], loaded, [.5])


@pytest.mark.parametrize(("key", "invalid"), [
    ("schema", "qsart.split_conformal.v1"), ("method", "ordinal_greedy"),
    ("qhat", float("nan")), ("qhat", float("inf")), ("qhat", -1), ("qhat", 2),
    ("qhat", True), ("qhat", None), ("rank", 4), ("n_images", True),
    ("n_images", 1), ("n_calibration_units", 0), ("num_classes", 1),
    ("unit", "unknown"), ("unit", []), ("coverage_scope", "clinical_guarantee"),
    ("tie_policy", "random_ties"), ("randomization", "independent_per_candidate"),
    ("metadata", {"invalid": float("nan")}), ("extra", "field"),
])
def test_invalid_artifacts_rejected(key, invalid):
    artifact = fit_randomized([[.8, .2]] * 10, [0] * 10, [.5] * 10)
    with pytest.raises(ValueError):
        validate_randomized({**artifact, key: invalid})


def test_infinite_threshold_rejects_finite_value_when_rank_exceeds_count():
    artifact = fit_randomized([[.8, .2]], [0], [.5])
    with pytest.raises(ValueError, match="threshold"):
        validate_randomized({**artifact, "qhat": 1.0})


def test_json_duplicate_fields_and_nonstandard_constants_rejected(tmp_path):
    path = tmp_path / "bad.json"
    for content in ['{"qhat":NaN}', '{"schema":"x","schema":"y"}', '{"qhat":Infinity}']:
        path.write_text(content, encoding="utf-8")
        with pytest.raises(ValueError):
            load_randomized(path)


def test_finite_exchangeable_leave_one_patient_out_score_event():
    # Exhaust every held-out position, with model, scores and uniforms fixed.
    # This verifies the kth-rank construction, not a clinical coverage claim.
    p0 = np.linspace(.5, .95, 11)
    probabilities = np.repeat(np.column_stack([p0, 1-p0]), 2, axis=0)
    uniforms = np.tile([.1, .2], 11)
    groups = np.repeat(np.arange(11), 2)
    successes = 0
    for held_out in range(11):
        fitting = groups != held_out
        artifact = fit_randomized(probabilities[fitting], np.zeros(20, dtype=int), uniforms[fitting],
                                  alpha=.2, groups=groups[fitting])
        sets = predict_randomized(probabilities[~fitting], artifact, uniforms[~fitting],
                                   empty_policy="allow")
        successes += all(0 in values for values in sets)
    assert successes == 9
    assert successes / 11 >= .8
