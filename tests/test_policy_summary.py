"""Synthetic fixtures verify unavailable outcomes and public summary provenance."""

from copy import deepcopy
import hashlib
import json

import pytest

from scripts.summarize_policy_comparison import build_summary, make_status_plot


@pytest.fixture
def measured_report():
    metrics = {"coverage": 0.0, "review_count": 4, "recapture_count": 0,
               "auto_nonreferable_count": 0, "proxy_total_cost_per_image": .25}
    scenarios = []
    for variant in ("tuned", "default"):
        fit = ({"status": "tuned", "chosen_decision_review_price": .05,
                "candidates": [{"status": "evaluated"}]} if variant == "tuned" else
               {"status": "not_fitted", "decision_review_price": .25, "candidates": []})
        scenarios.append({
            "mode": "ordinal", "review_budget": 1.0, "recapture_budget": .4, "method": "dual_budget",
            "variant": variant, "status": "evaluated", "evaluated_folds": 1, "untunable_folds": 0,
            "evaluation_overload_folds": 0, "images_with_real_actions": 4, "patients_with_real_actions": 1,
            "fold_records": [{"fold": 0, "fit": fit,
                              "evaluation": {"status": "evaluated", "metrics": metrics}}],
            "pooled_metrics": metrics,
        })
    missing = deepcopy(scenarios[1])
    missing.update(mode="lac", status="incomplete", evaluated_folds=0, evaluation_overload_folds=1,
                   images_with_real_actions=0, patients_with_real_actions=0, pooled_metrics=None)
    missing["fold_records"][0]["evaluation"] = {"status": "overloaded"}
    scenarios.append(missing)
    interval = {"mean_delta": 0., "lower": 0., "upper": 0., "n_images": 4, "n_patients": 1,
                "repeats": 1000, "seed": 20261022, "direction": "left_minus_right",
                "method": "paired_patient_cluster_percentile_bootstrap", "confidence_level": .95,
                "weighting": "image_weighted_mean_with_whole_patient_resampling", "multiplicity_adjusted": False}
    def key(row):
        return [row[k] for k in ("mode", "review_budget", "recapture_budget", "method", "variant")]

    comparisons = [
        {"kind": "tuned_vs_default", "left": key(scenarios[0]), "right": key(scenarios[1]),
         "status": "evaluated", "matched_realized_workload_within_every_fold": True, "images": 4, "patients": 1,
         "left_fold_review_recapture_counts": [[4, 0]], "right_fold_review_recapture_counts": [[4, 0]],
         "paired_deltas": {"proxy_cost": interval, "automatic_indicator": deepcopy(interval)}},
        {"kind": "diagnostic", "left": key(scenarios[0]), "right": key(scenarios[2]),
         "status": "not_estimable", "matched_realized_workload_within_every_fold": None},
    ]
    return {
        "experiment_id": "synthetic_policy_fixture", "spec_version": "1.0", "dataset": "synthetic",
        "partition": "validation", "scope": "development_out_of_fold_policy", "images": 4, "patients": 1,
        "model_version": "synthetic", "scenarios": scenarios, "paired_comparisons": comparisons,
        "summary": {"scenarios": 3, "fully_evaluated_scenarios": 2, "incomplete_scenarios": 1,
                    "planned_tuning_candidates": 1, "paired_comparisons": 2, "estimable_pairs": 1},
        "folds": [{"fold": 0, "evaluation_images": 4, "evaluation_patients": 1}],
        "fixed_evaluation_costs": {"review": .25, "recapture": .2, "severity_weights": [0, 0, 3, 6, 10]},
        "decision_review_prices": [.05, .25], "limitations": ["Explicitly synthetic test fixture."],
        "manifest_sha256": "a" * 64, "cache_sha256": "b" * 64, "cache_report_sha256": "c" * 64,
        "artifact_sha256": {"dr": "d" * 64}, "protocol_sha256": "e" * 64, "source_commit": "f" * 40,
        "source_sha256": {"synthetic.py": "0" * 64}, "package_versions": {},
        "private_fold_assignment_sha256": "1" * 64, "private_output_sha256": "2" * 64, "elapsed_seconds": 1.,
    }


def test_observed_zero_is_distinct_from_unavailable_and_prices_are_preserved(measured_report):
    source_bytes = json.dumps(measured_report).encode()
    digest = hashlib.sha256(source_bytes).hexdigest()
    result = build_summary(measured_report, digest)
    assert result["source_report_sha256"] == digest
    assert result["scenarios"][0]["pooled_metrics"]["coverage"] == 0
    assert result["scenarios"][2]["pooled_metrics"] is None
    assert result["scenarios"][0]["chosen_decision_review_prices"] == [.05]
    assert result["scenarios"][1]["chosen_decision_review_prices"] == [.25]
    assert result["evaluation_ledger"]["review"] == .25
    assert json.dumps(measured_report).encode() == source_bytes


def test_unavailable_pair_keeps_null_deltas_and_workload_claim(measured_report):
    result = build_summary(measured_report, "0" * 64)
    assert result["paired_comparisons"][1]["estimable"] is False
    assert result["paired_comparisons"][1]["paired_deltas"] is None
    assert result["paired_comparisons"][1]["matched_realized_workload_within_every_fold"] is None
    assert result["summary"]["matched_realized_workload_pairs"] == 1
    assert result["summary"]["unmatched_realized_workload_pairs"] == 0
    assert result["summary"]["not_estimable_pairs"] == 1
    assert result["summary"]["matched_workload_pairs_with_nonzero_mean_delta"] == 0
    assert result["paired_bootstrap"]["multiplicity_adjusted"] is False


def test_private_fields_are_not_copied_to_dashboard(measured_report):
    measured_report["private_subjects"] = ["PRIVATE_SUBJECT"]
    measured_report["scenarios"][0]["patient_actions"] = ["PRIVATE_ACTION"]
    measured_report["scenarios"][0]["fold_records"][0]["fit"]["patient_id"] = "PRIVATE_ID"
    result = json.dumps(build_summary(measured_report, "0" * 64))
    assert "PRIVATE_" not in result
    assert "candidates" not in result.replace("planned_tuning_candidates", "")


@pytest.mark.parametrize("corruption", ["missing_pooled", "invented_pooled", "false_workload", "false_count",
                                         "wrong_fold_status", "invented_pair", "changed_bootstrap"])
def test_inconsistent_measured_outcomes_are_rejected(measured_report, corruption):
    if corruption == "missing_pooled":
        measured_report["scenarios"][0]["pooled_metrics"] = None
    elif corruption == "invented_pooled":
        measured_report["scenarios"][2]["pooled_metrics"] = {"coverage": 0}
    elif corruption == "false_workload":
        measured_report["paired_comparisons"][0]["right_fold_review_recapture_counts"] = [[3, 1]]
    elif corruption == "false_count":
        measured_report["summary"]["fully_evaluated_scenarios"] = 3
    elif corruption == "wrong_fold_status":
        measured_report["scenarios"][0]["fold_records"][0]["evaluation"]["status"] = "overloaded"
    elif corruption == "invented_pair":
        measured_report["paired_comparisons"][1]["paired_deltas"] = {"proxy_cost": 0}
    else:
        measured_report["paired_comparisons"][0]["paired_deltas"]["automatic_indicator"]["seed"] = 17
    with pytest.raises(ValueError):
        build_summary(measured_report, "0" * 64)


@pytest.mark.parametrize("digest", ["invalid", "A" * 64, "g" * 64])
def test_invalid_source_digest_is_rejected(measured_report, digest):
    with pytest.raises(ValueError, match="SHA-256"):
        build_summary(measured_report, digest)


def test_status_figure_is_written_from_complete_design(measured_report, tmp_path):
    # A full synthetic two-mode rectangle is needed for every displayed cell.
    missing_tuned = deepcopy(measured_report["scenarios"][2])
    missing_tuned["variant"] = "tuned"
    missing_tuned["fold_records"][0]["fit"] = {"status": "untunable", "candidates": []}
    missing_tuned["fold_records"][0]["evaluation"]["status"] = "untunable"
    missing_tuned.update(untunable_folds=1, evaluation_overload_folds=0)
    measured_report["scenarios"].append(missing_tuned)
    measured_report["summary"].update(scenarios=4, incomplete_scenarios=2)
    summary = build_summary(measured_report, "0" * 64)
    output = tmp_path / "status.png"
    make_status_plot(summary, output)
    assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert output.stat().st_size > 10_000
