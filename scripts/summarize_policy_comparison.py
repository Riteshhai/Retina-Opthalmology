"""Derive a compact public dashboard view without changing measured policy results."""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import numpy as np


IDENTITY_FIELDS = ("mode", "review_budget", "recapture_budget", "method", "variant")
BOOTSTRAP_FIELDS = (
    "repeats", "seed", "direction", "method", "confidence_level", "weighting", "multiplicity_adjusted",
)
DELTA_FIELDS = ("mean_delta", "lower", "upper", "n_images", "n_patients")
PROVENANCE_FIELDS = (
    "manifest_sha256", "cache_sha256", "cache_report_sha256", "artifact_sha256", "protocol_sha256",
    "source_commit", "source_sha256", "package_versions", "private_fold_assignment_sha256",
    "private_output_sha256", "elapsed_seconds",
)


def scenario_key(row):
    return tuple(row[field] for field in IDENTITY_FIELDS)


def _compact_scenario(row):
    result = {field: deepcopy(row[field]) for field in IDENTITY_FIELDS}
    for field in (
        "status", "evaluated_folds", "untunable_folds", "evaluation_overload_folds",
        "images_with_real_actions", "patients_with_real_actions",
    ):
        result[field] = row[field]
    complete = row["status"] == "evaluated"
    folds = row["fold_records"]
    if complete != all(fold["evaluation"]["status"] == "evaluated" for fold in folds):
        raise ValueError("Scenario status disagrees with evaluation fold statuses")
    if complete != (row.get("pooled_metrics") is not None):
        raise ValueError("Pooled metrics must exist only for complete scenarios")
    result["pooled_metrics"] = deepcopy(row.get("pooled_metrics"))
    result["chosen_decision_review_prices"] = []
    result["fold_statuses"] = []
    for fold in folds:
        fit, evaluation = fold["fit"], fold["evaluation"]
        if row["variant"] == "default":
            price = fit["decision_review_price"]
        else:
            price = fit.get("chosen_decision_review_price")
        result["chosen_decision_review_prices"].append(price)
        result["fold_statuses"].append({
            "fold": fold["fold"], "fit_status": fit["status"], "evaluation_status": evaluation["status"],
        })
    return result


def build_summary(report, source_report_sha256):
    """Allowlist public aggregates and check that unavailable outcomes remain null."""
    if report.get("scope") != "development_out_of_fold_policy":
        raise ValueError("Expected the development out-of-fold policy study")
    if len(source_report_sha256) != 64 or any(c not in "0123456789abcdef" for c in source_report_sha256):
        raise ValueError("Expected a lowercase SHA-256 digest of the measured report bytes")
    scenarios = [_compact_scenario(row) for row in report["scenarios"]]
    by_key = {scenario_key(row): row for row in scenarios}
    if len(by_key) != len(scenarios):
        raise ValueError("Duplicate scenario identity")
    comparisons, bootstrap_settings = [], None
    for pair in report["paired_comparisons"]:
        left, right = (by_key[tuple(pair[side])] for side in ("left", "right"))
        estimable = pair["status"] == "evaluated"
        if estimable != (left["status"] == right["status"] == "evaluated"):
            raise ValueError("Pair estimability disagrees with complete scenario outcomes")
        compact = {field: deepcopy(pair[field]) for field in ("kind", "left", "right", "status")}
        compact["estimable"] = estimable
        workload_match = pair["matched_realized_workload_within_every_fold"]
        compact["matched_realized_workload_within_every_fold"] = workload_match
        compact["images"] = pair.get("images")
        compact["patients"] = pair.get("patients")
        compact["left_fold_review_recapture_counts"] = pair.get("left_fold_review_recapture_counts")
        compact["right_fold_review_recapture_counts"] = pair.get("right_fold_review_recapture_counts")
        compact["paired_deltas"] = None
        if estimable:
            if not isinstance(workload_match, bool):
                raise ValueError("An estimable comparison needs a realized-workload match flag")
            if workload_match != (compact["left_fold_review_recapture_counts"] ==
                                  compact["right_fold_review_recapture_counts"]):
                raise ValueError("Realized-workload flag disagrees with per-fold counts")
            compact["paired_deltas"] = {}
            for metric, interval in pair["paired_deltas"].items():
                settings = {field: interval[field] for field in BOOTSTRAP_FIELDS}
                if bootstrap_settings is None:
                    bootstrap_settings = settings
                elif settings != bootstrap_settings:
                    raise ValueError("Cannot centralize inconsistent bootstrap settings")
                compact["paired_deltas"][metric] = {field: interval[field] for field in DELTA_FIELDS}
        elif workload_match is not None or pair.get("paired_deltas") is not None:
            raise ValueError("A non-estimable comparison cannot contain outcome deltas or a workload claim")
        comparisons.append(compact)

    evaluated = sum(row["status"] == "evaluated" for row in scenarios)
    estimable_pairs = sum(pair["estimable"] for pair in comparisons)
    candidate_statuses = Counter(
        candidate["status"] for row in report["scenarios"] for fold in row["fold_records"]
        for candidate in fold["fit"].get("candidates", [])
    )
    measured_counts = {
        "scenarios": len(scenarios), "fully_evaluated_scenarios": evaluated,
        "incomplete_scenarios": len(scenarios) - evaluated,
        "planned_tuning_candidates": sum(candidate_statuses.values()),
        "paired_comparisons": len(comparisons), "estimable_pairs": estimable_pairs,
    }
    if measured_counts != report["summary"]:
        raise ValueError("Measured report summary disagrees with scenario, pair or candidate counts")
    matched = [pair for pair in comparisons if pair["matched_realized_workload_within_every_fold"] is True]
    summary = {
        **measured_counts,
        "tuning_candidate_statuses": dict(candidate_statuses),
        "matched_realized_workload_pairs": len(matched),
        "unmatched_realized_workload_pairs": estimable_pairs - len(matched),
        "not_estimable_pairs": len(comparisons) - estimable_pairs,
        "matched_workload_pairs_with_nonzero_mean_delta": sum(
            any(value["mean_delta"] != 0 for value in pair["paired_deltas"].values()) for pair in matched
        ),
        "mode_status_counts": {
            mode: dict(Counter(row["status"] for row in scenarios if row["mode"] == mode))
            for mode in dict.fromkeys(row["mode"] for row in scenarios)
        },
    }
    result = {field: report[field] for field in (
        "experiment_id", "spec_version", "dataset", "partition", "scope", "images", "patients", "model_version",
    )}
    result.update({
        "schema_version": 1,
        "artifact_kind": "derived_dashboard_summary",
        "source_report": "reports/policy_comparison_v1/evaluation.json",
        "source_report_sha256": source_report_sha256,
        "population": {
            "dataset": report["dataset"], "partition": report["partition"], "scope": report["scope"],
            "images": report["images"], "patients": report["patients"], "folds": deepcopy(report["folds"]),
            "interpretation": "Already inspected model-selection patients; no fresh end-to-end evaluation.",
        },
        "summary": summary,
        "evaluation_ledger": deepcopy(report["fixed_evaluation_costs"]),
        "decision_review_prices": deepcopy(report["decision_review_prices"]),
        "scenarios": scenarios,
        "paired_comparisons": comparisons,
        "paired_bootstrap": bootstrap_settings,
        "provenance": {field: deepcopy(report[field]) for field in PROVENANCE_FIELDS},
        "limitations": deepcopy(report["limitations"]),
        "mode_interpretation": {
            "ordinal": "Current deployed conformal guard; research policy comparisons only.",
            "lac": "Conventional LAC sensitivity analysis; no deployment replacement.",
            "no_cp": "No-conformal diagnostic ablation; no deployment replacement.",
        },
        "status_interpretation": {
            "evaluated": "All excluded folds have real actions; pooled outcomes are available.",
            "incomplete": "One or more folds is untunable or overloaded; pooled outcomes are null.",
            "untunable": "No tuning candidate is feasible on the 112-image tuning cohort; no excluded-fold outcome.",
            "overloaded": "Allocation exceeds mandatory capacities in an excluded evaluation fold; no actions.",
        },
    })
    return result


def make_status_plot(summary, output):
    """Show all scenarios with a separate visual state for unavailable outcomes."""
    rows = summary["scenarios"]
    modes = list(dict.fromkeys(row["mode"] for row in rows))
    methods = list(dict.fromkeys(row["method"] for row in rows))
    budgets = list(dict.fromkeys((row["review_budget"], row["recapture_budget"]) for row in rows))
    lookup = {scenario_key(row): row for row in rows}
    fig, axes = plt.subplots(2, len(modes), figsize=(15.2, 9.2), squeeze=False)
    colors = ListedColormap(["#eee4d5", "#c6e5dc"])
    titles = {"ordinal": "Ordinal guard", "lac": "LAC diagnostic", "no_cp": "No-CP ablation"}
    for vindex, variant in enumerate(("tuned", "default")):
        for mindex, mode in enumerate(modes):
            axis = axes[vindex, mindex]
            records = [[lookup[(mode, review, recapture, method, variant)] for review, recapture in budgets]
                       for method in methods]
            values = np.array([[int(row["status"] == "evaluated") for row in group] for group in records])
            axis.imshow(values, cmap=colors, vmin=0, vmax=1, aspect="auto", interpolation="nearest")
            for y, group in enumerate(records):
                for x, row in enumerate(group):
                    label = (f'{row["pooled_metrics"]["coverage"]:.1%}'
                             if row["status"] == "evaluated" else "—")
                    axis.text(x, y, label, ha="center", va="center", fontsize=9, color="#203c3a")
            axis.set_xticks(range(len(budgets)), [f"{review:.2f} / {recapture:.2f}"
                                                for review, recapture in budgets], fontsize=8)
            axis.set_yticks(range(len(methods)), [method.replace("_", " ") for method in methods], fontsize=9)
            axis.set_title(f"{titles.get(mode, mode)} · {variant}", fontsize=12, pad=11)
            axis.set_xlabel("Review / recapture budget fraction", fontsize=9)
            axis.tick_params(length=0, pad=7)
            axis.set_xticks(np.arange(-.5, len(budgets), 1), minor=True)
            axis.set_yticks(np.arange(-.5, len(methods), 1), minor=True)
            axis.grid(which="minor", color="white", linewidth=2)
            axis.tick_params(which="minor", bottom=False, left=False)
            for spine in axis.spines.values():
                spine.set_visible(False)
    fig.suptitle("Policy feasibility and automatic non-referable coverage", fontsize=17, y=.985)
    fig.legend(handles=[Patch(facecolor="#c6e5dc", label="Evaluated: cell shows automatic coverage"),
                        Patch(facecolor="#eee4d5", label="Incomplete: — means no pooled outcome")],
               loc="lower center", bbox_to_anchor=(.5, .065), ncol=2, frameon=False, fontsize=10)
    fig.text(.5, .036,
             f'{summary["summary"]["fully_evaluated_scenarios"]} / {len(rows)} scenarios fully evaluated · '
             f'{summary["images"]} selection images / {summary["patients"]} patients · '
             "three patient folds; previously inspected development population",
             ha="center", fontsize=9)
    fig.text(.5, .014, "Budget fractions are caps; equal caps do not imply equal realized workloads. "
             "Zero coverage is an observed result, not missing data.", ha="center", fontsize=9)
    fig.subplots_adjust(left=.105, right=.985, top=.92, bottom=.15, wspace=.49, hspace=.37)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160, facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("reports/policy_comparison_v1/evaluation.json"))
    parser.add_argument("--output", type=Path, default=Path("reports/policy_comparison_v1/dashboard_summary.json"))
    parser.add_argument("--plot", type=Path, default=Path("reports/policy_comparison_v1/scenario_statuses.png"))
    args = parser.parse_args()
    measured_bytes = args.source.read_bytes()
    summary = build_summary(json.loads(measured_bytes), hashlib.sha256(measured_bytes).hexdigest())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    make_status_plot(summary, args.plot)
    print(json.dumps({"summary": summary["summary"], "source_report_sha256": summary["source_report_sha256"],
                      "output_bytes": args.output.stat().st_size}, indent=2))


if __name__ == "__main__":
    main()
