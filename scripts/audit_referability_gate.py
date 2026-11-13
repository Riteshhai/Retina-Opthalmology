"""Independently reconstruct the development referability experiment from bound inputs."""

import csv
from decimal import Decimal, ROUND_CEILING
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = "referability_gate_v1"
REPORTS = ROOT / "reports" / EXPERIMENT
WITNESS = ROOT / "artifacts" / EXPERIMENT / "sets_and_actions.json"
PROTOCOL = ROOT / "docs/research/REFERABILITY_GATE_V1_PROTOCOL.md"
INPUTS = {
    "ordinal_report": "reports/ordinal_study_v1/evaluation.json",
    "ordinal_cache": "artifacts/ordinal_study_v1/scores_and_sets.json",
    "robustness_report": "reports/robustness_v1/evaluation.json",
    "robustness_cache": "artifacts/robustness_v1/scores.json",
    "manifest": "data/processed/deepdrid/development_v1.csv",
    "dr": "checkpoints/development_v1/selected/dr.pt",
    "quality": "checkpoints/development_v1/selected/quality.pt",
    "calibration": "checkpoints/development_v1/selected/calibration.json",
    "conformal": "checkpoints/spec_v1/conformal_r1.json",
}
METHODS = ("ordinal_projected", "lac_projected", "binary_lac")
BUDGETS = ((0.30, 0.20), (0.60, 0.40), (0.90, 0.40), (1.0, 0.40))
ACTIONS = ("AUTO_NONREFERABLE", "RECAPTURE", "SPECIALIST_REVIEW")
WEIGHTS = np.array([0.0, 0.0, 3.0, 6.0, 10.0])
POLICY = {
    "review_budget": 0.30,
    "recapture_budget": 0.20,
    "minimum_quality": 0.50,
    "review_risk": 0.50,
    "uncertainty_threshold": 0.75,
    "referable_grade": 2,
    "severe_grade": 3,
    "severe_risk": 0.25,
    "severity_costs": WEIGHTS.tolist(),
    "review_cost": 0.25,
    "recapture_cost": 0.20,
    "uncertainty_weight": 0.50,
    "quality_weight": 1.0,
}


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def match(expected, actual, name):
    if isinstance(expected, dict):
        for key, value in expected.items():
            require(key in actual, f"{name}.{key} is missing")
            match(value, actual[key], f"{name}.{key}")
    elif isinstance(expected, float):
        require(
            isinstance(actual, (int, float))
            and not isinstance(actual, bool)
            and math.isclose(expected, actual, rel_tol=0.0, abs_tol=2e-12),
            f"{name} differs",
        )
    else:
        require(expected == actual, f"{name} differs")


def verify_bindings(report):
    bindings = {name: digest(ROOT / path) for name, path in INPUTS.items()}
    require(bindings == report["input_sha256"], "Frozen input file hashes differ")
    require(digest(PROTOCOL) == report["protocol_sha256"], "Protocol hash differs")
    require(digest(WITNESS) == report["private_output_sha256"], "Private output hash differs")
    commit = report["source_commit"]
    require(bool(re.fullmatch(r"[0-9a-f]{40}", commit)), "Invalid execution commit")
    sources = report["source_sha256"]
    require("scripts/run_referability_gate.py" in sources, "Missing execution source binding")
    for name, expected in sources.items():
        path = ROOT / name
        require(not Path(name).is_absolute() and path.resolve().is_relative_to(ROOT), "Unsafe source name")
        content = subprocess.check_output(["git", "show", f"{commit}:{name}"], cwd=ROOT)
        require(hashlib.sha256(content).hexdigest() == expected == digest(path), "Execution source differs")
    protocol_name = PROTOCOL.relative_to(ROOT).as_posix()
    protocol_bytes = subprocess.check_output(["git", "show", f"{commit}:{protocol_name}"], cwd=ROOT)
    require(
        hashlib.sha256(protocol_bytes).hexdigest() == report["protocol_sha256"], "Protocol was not locked"
    )
    auditor = Path(__file__).resolve()
    auditor_bytes = subprocess.check_output(
        ["git", "show", f"{commit}:{auditor.relative_to(ROOT).as_posix()}"], cwd=ROOT
    )
    require(
        hashlib.sha256(auditor_bytes).hexdigest() == digest(auditor), "Independent auditor was not locked"
    )
    ordinal_report = read(ROOT / INPUTS["ordinal_report"])
    robustness_report = read(ROOT / INPUTS["robustness_report"])
    require(
        ordinal_report["private_output_sha256"] == bindings["ordinal_cache"]
        and ordinal_report["selection_cache_sha256"] == bindings["robustness_cache"]
        and robustness_report["private_scores_sha256"] == bindings["robustness_cache"]
        and ordinal_report["selection_report_sha256"] == bindings["robustness_report"],
        "Prior report-to-cache bindings differ",
    )
    checkpoints = {key: bindings[key] for key in ("dr", "quality", "calibration", "conformal")}
    for previous in (ordinal_report, robustness_report):
        require(
            previous["artifact_sha256"] == checkpoints
            and previous["manifest_sha256"] == bindings["manifest"],
            "Prior model or manifest bindings differ",
        )
    return bindings


def cohort_inputs(ordinal, robust, sidecar, bindings):
    manifest = ROOT / INPUTS["manifest"]
    with manifest.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    require(
        {row["split"] for row in rows} == {"train", "validation", "calibration"}, "Unexpected manifest split"
    )
    cal = [row for row in rows if row["split"] == "calibration"]
    selected = [row for row in rows if row["split"] == "validation"]
    patients = sorted(
        {row["subject_id"] for row in cal},
        key=lambda value: (hashlib.sha256(f"spec-v1-conformal-20261020:{value}".encode()).hexdigest(), value),
    )
    require(len(patients) == 42, "Unexpected calibration patient count")
    temp = [row for row in cal if row["subject_id"] in patients[:21]]
    cp = [row for row in cal if row["subject_id"] in patients[21:]]
    require(
        len(temp) == len(cp) == 84
        and len(selected) == 168
        and len({row["subject_id"] for row in selected}) == 42,
        "Unexpected cohort size",
    )

    def image_path(row):
        return (manifest.parent / row["image_path"]).resolve()

    def split_hash(part):
        names = sorted(str(image_path(row)).casefold() for row in part)
        return hashlib.sha256("\n".join(names).encode()).hexdigest()

    for i, part in enumerate((temp, cp, selected)):
        for previous in (temp, cp, selected)[:i]:
            require(
                all(
                    {row[key] for row in part}.isdisjoint(row[key] for row in previous)
                    for key in ("subject_id", "sha256", "image_path")
                ),
                "Overlapping patient or image roles",
            )
        require(
            all(digest(image_path(row)) == row["sha256"] for row in part),
            "Bound development image bytes differ",
        )
    require(
        sidecar["split_sha256"] == split_hash(cp)
        and sidecar["temperature_fit_split_sha256"] == split_hash(temp)
        and sidecar["manifest_sha256"] == bindings["manifest"],
        "Independent calibration roles differ",
    )
    for field, key in (
        ("dr_checkpoint_sha256", "dr"),
        ("quality_checkpoint_sha256", "quality"),
        ("base_calibration_sha256", "calibration"),
    ):
        require(sidecar[field] == bindings[key], f"Sidecar {field} binding differs")
    temp_record = sidecar["independent_temperature"]
    temp_hash = hashlib.sha256(
        json.dumps(temp_record, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    require(temp_hash == sidecar["temperature_sha256"], "Stored temperature binding differs")
    for part, grades, subjects in (
        (cp, ordinal["calibration_grades"], ordinal["calibration_subject_ids"]),
        (selected, ordinal["selection_grades"], ordinal["selection_subject_ids"]),
        (selected, robust["original_grades"], robust["original_subject_ids"]),
    ):
        require(
            [int(row["dr_grade"]) for row in part] == grades
            and [row["subject_id"] for row in part] == list(map(str, subjects)),
            "Frozen score labels or patient ordering differ",
        )
    clean = robust["conditions"]["clean"]["rows"]
    require(
        len(clean) == len(selected) and all(not row["second_pass_activated"] for row in clean),
        "Selection predictions are not the frozen first pass",
    )
    return clean


def transform(rows, temperature):
    # Independent direct-power form of temperature scaling.
    p = np.clip(np.asarray(rows, float), 1e-30, 1.0) ** (1.0 / temperature)
    return p / p.sum(axis=1, keepdims=True)


def candidate_scores(probabilities, method):
    if method == "binary_lac":
        p = np.column_stack((probabilities[:, :2].sum(1), probabilities[:, 2:].sum(1)))
    else:
        p = probabilities
    p = p / p.sum(axis=1, keepdims=True)
    if method != "ordinal_projected":
        return 1.0 - p
    scores = np.empty_like(p)
    for index, row in enumerate(p):
        first = int(np.argmax(row))
        used = [first]
        while len(used) < len(row):
            neighbors = [k for k in (min(used) - 1, max(used) + 1) if 0 <= k < len(row)]
            used.append(max(neighbors, key=lambda k: (row[k], k)))
        scores[index, used] = np.minimum(np.cumsum(row[used]), 1.0)
    return scores


def threshold(scores, grades, groups, rank):
    truth = scores[np.arange(len(grades)), grades]
    patient_scores = sorted(float(truth[groups == group].max()) for group in np.unique(groups))
    require(len(patient_scores) == 21, "Unexpected number of threshold calibration units")
    return patient_scores[rank - 1]


def set_metrics(sets, grades, groups):
    binary = grades >= 2
    covered = np.array([int(label) in values for label, values in zip(binary, sets, strict=True)])
    sizes = np.array([len(values) for values in sets])

    def coverage_by_label(labels, count):
        result = {}
        for label in range(count):
            mask = labels == label
            result[str(label)] = {
                "n": int(mask.sum()),
                "coverage": float(covered[mask].mean()) if mask.any() else None,
            }
        return result

    return {
        "n_images": len(grades),
        "n_patients": len(np.unique(groups)),
        "marginal_coverage": float(covered.mean()),
        "patient_all_views_coverage": float(
            np.mean([covered[groups == group].all() for group in np.unique(groups)])
        ),
        "mean_set_size": float(sizes.mean()),
        "empty_set_count": int((sizes == 0).sum()),
        "contiguous_nonempty_fraction": float((sizes > 0).mean()),
        "set_size_counts": {str(size): int((sizes == size).sum()) for size in range(3)},
        "nonreferable_singleton_count": sets.count([0]),
        "referable_singleton_count": sets.count([1]),
        "per_class": coverage_by_label(binary, 2),
        "by_true_grade": coverage_by_label(grades, 5),
    }


def action_metrics(actions, grades):
    auto, recap, review = (actions == action for action in ACTIONS)
    referable, severe = grades >= 2, grades >= 3
    missed = referable & auto
    costs = {
        "proxy_missed_referable_cost": float(WEIGHTS[grades[missed]].sum()),
        "proxy_severe_recapture_cost": float(WEIGHTS[grades[severe & recap]].sum()),
        "proxy_review_cost": 0.25 * int(review.sum()),
        "proxy_recapture_cost": 0.20 * int(recap.sum()),
    }

    def ratio(numerator, denominator):
        return float(numerator / denominator) if denominator else None

    return {
        "n": len(grades),
        "unit": "image",
        "referable_count": int(referable.sum()),
        "severe_count": int(severe.sum()),
        "auto_nonreferable_count": int(auto.sum()),
        "review_count": int(review.sum()),
        "recapture_count": int(recap.sum()),
        "coverage": float(auto.mean()),
        "review_rate": float(review.mean()),
        "recapture_rate": float(recap.mean()),
        "missed_referable_auto": int(missed.sum()),
        "missed_referable_per_1000_all": 1000 * float(missed.mean()),
        "missed_referable_fraction_of_referable": ratio(missed.sum(), referable.sum()),
        "false_omission_fraction_of_auto": ratio(missed.sum(), auto.sum()),
        "missed_severe_auto": int((severe & auto).sum()),
        "referable_deferred_to_recapture": int((referable & recap).sum()),
        "severe_deferred_to_recapture": int((severe & recap).sum()),
        "severe_recapture_fraction_of_severe": ratio((severe & recap).sum(), severe.sum()),
        "proxy_total_cost_per_image": sum(costs.values()) / len(grades),
        **costs,
        "by_true_grade": {
            str(grade): {action: int(((grades == grade) & (actions == action)).sum()) for action in ACTIONS}
            for grade in range(5)
        },
    }


def audit_allocation(scenario, saved, budget, routes, costs, grades):
    review_budget, recap_budget = budget
    review_cap, recap_cap = (int(Decimal(str(value)) * len(grades)) for value in budget)
    pair = {"review_budget": review_budget, "recapture_budget": recap_budget}
    match(pair, scenario, "scenario budget")
    match(pair, saved, "private scenario budget")
    mandatory, poor, eligible = (routes == ACTIONS[k] for k in (2, 1, 0))
    needs_review, needs_recap = int(mandatory.sum()), int(poor.sum())
    needed_total = needs_review + needs_recap
    spill = max(0, needs_recap - recap_cap)
    shortfall = max(0, needs_review + spill - review_cap)
    diagnostics = {
        "n": len(grades),
        "needed_review": needs_review,
        "needed_recapture": needs_recap,
        "needed_recapture_or_review": needs_recap,
        "needed_recapture_after_available_review": max(0, needs_recap - max(0, review_cap - needs_review)),
        "needed_total": needed_total,
        "review_capacity": review_cap,
        "recapture_capacity": recap_cap,
        "total_capacity": review_cap + recap_cap,
        "mandatory_review_shortfall": max(0, needs_review - review_cap),
        "review_shortfall_after_recapture": shortfall,
        "total_shortfall": max(0, needed_total - review_cap - recap_cap),
        "eligible_auto_nonreferable": int(eligible.sum()),
        "unsafe_clearance_permitted": False,
    }
    match(diagnostics, scenario["diagnostics"], "budget diagnostics")
    if shortfall:
        require(
            scenario["status"] == "overloaded" and scenario["metrics"] is None and saved["actions"] is None,
            "Overloaded scenario reported actions or metrics",
        )
        return None
    require(scenario["status"] == "evaluated", "Feasible scenario was not evaluated")
    actions = np.asarray(saved["actions"])
    require(actions.shape == routes.shape and np.isin(actions, ACTIONS).all(), "Invalid allocation witness")
    auto, recap, review = (actions == action for action in ACTIONS)
    require(
        np.all(review[mandatory])
        and not (auto & ~eligible).any()
        and not (recap & ~poor).any()
        and review.sum() <= review_cap
        and recap.sum() <= recap_cap,
        "Allocation violates a guard or capacity",
    )
    match(
        {
            **action_metrics(actions, grades),
            "review_capacity": review_cap,
            "recapture_capacity": recap_cap,
            "review_budget_violation": False,
            "recapture_budget_violation": False,
        },
        scenario["metrics"],
        "allocation metrics",
    )

    # Start with pointwise actions. Each optional review replaces exactly one auto
    # or recapture, so the best savings can be chosen without the production LP.
    poor_savings = sorted(costs[poor, 1] - costs[poor, 2], reverse=True)
    eligible_savings = list(costs[eligible, 0] - costs[eligible, 2])
    optional = sorted(poor_savings[spill:] + eligible_savings, reverse=True)
    slots = review_cap - needs_review - spill
    baseline = costs[mandatory, 2].sum() + costs[poor, 1].sum() + costs[eligible, 0].sum()
    optimum = baseline - sum(poor_savings[:spill]) - sum(max(0.0, gain) for gain in optional[:slots])
    actual = costs[review, 2].sum() + costs[recap, 1].sum() + costs[auto, 0].sum()
    error = float(abs(actual - optimum))
    require(error <= 1e-7, "Allocation cost differs from independent exact optimum")
    return error


def run():
    evaluation = REPORTS / "evaluation.json"
    report, private = read(evaluation), read(WITNESS)
    require(
        report["experiment_id"] == EXPERIMENT and report["partition"] == "validation", "Unexpected experiment"
    )
    match(POLICY, report["policy"], "policy")
    match([list(pair) for pair in BUDGETS], report["budget_pairs"], "budget grid")
    bindings = verify_bindings(report)
    ordinal = read(ROOT / INPUTS["ordinal_cache"])
    robust = read(ROOT / INPUTS["robustness_cache"])
    sidecar = read(ROOT / INPUTS["conformal"])
    rows = cohort_inputs(ordinal, robust, sidecar, bindings)
    temperature = sidecar["independent_temperature"]["first_pass"]
    match(
        {
            "temperature_images": 84,
            "temperature_patients": 21,
            "conformal_images": 84,
            "conformal_patients": 21,
            "independent_first_pass_temperature": temperature,
            "temperature_fit_split_sha256": sidecar["temperature_fit_split_sha256"],
            "conformal_split_sha256": sidecar["split_sha256"],
        },
        report["calibration"],
        "calibration",
    )
    cp_p = transform(ordinal["calibration_raw_probabilities"], temperature)
    selected_p = transform([row["uncalibrated"]["grade_probabilities"] for row in rows], temperature)
    cp_g, cp_y = np.asarray(ordinal["calibration_subject_ids"]), np.asarray(ordinal["calibration_grades"])
    groups, grades = np.asarray(ordinal["selection_subject_ids"]), np.asarray(ordinal["selection_grades"])
    production_p = np.array([row["grade_probabilities"] for row in rows])
    quality = 1.0 - np.array([row["quality"]["poor_probability"] for row in rows])
    entropy = np.array([row["uncertainty"]["entropy"] for row in rows])
    poor = quality < 0.50
    baseline_review = (
        (production_p.argmax(1) >= 2)
        | (production_p[:, 2:].sum(1) >= 0.50)
        | (production_p[:, 3:].sum(1) >= 0.25)
        | ((entropy >= 0.75) & ~poor)
    )
    expected_miss = production_p @ WEIGHTS
    costs = np.column_stack(
        (
            expected_miss + (1.0 - quality) + 0.5 * entropy,
            0.20 + expected_miss + 0.5 * quality,
            np.full(len(rows), 0.25),
        )
    )
    cases = [(alpha, method) for alpha in (0.10, 0.05) for method in METHODS]
    require(len(report["fits"]) == len(private["fits"]) == len(cases), "Incomplete conformal matrix")
    errors = []
    for (alpha, method), fit, saved in zip(cases, report["fits"], private["fits"], strict=True):
        match({"alpha": alpha, "method": method}, saved, "private fit identity")
        patient_units = len(np.unique(cp_g))
        rank = int(
            (Decimal(patient_units + 1) * (1 - Decimal(str(alpha)))).to_integral_value(rounding=ROUND_CEILING)
        )
        cp_labels = (cp_y >= 2).astype(int) if method == "binary_lac" else cp_y
        qhat = threshold(candidate_scores(cp_p, method), cp_labels, cp_g, rank)
        match({"alpha": alpha, "method": method, "rank": rank, "qhat": qhat}, fit, "conformal fit")
        raw = [np.flatnonzero(row <= qhat).tolist() for row in candidate_scores(selected_p, method)]
        if method != "binary_lac":
            old_method = {"ordinal_projected": "ordinal_greedy", "lac_projected": "lac"}[method]
            old = sidecar["models"]["first_pass"][str(alpha)][old_method]
            match({"rank": rank, "qhat": qhat}, old, "sidecar threshold")
            historic = [
                record
                for record in ordinal["fits_and_sets"]
                if record["method"] == old_method
                and record["unit"] == "patient_max"
                and record["alpha"] == alpha
            ]
            require(
                len(historic) == 1 and raw == historic[0]["raw_sets"], "Historical raw five-grade sets differ"
            )
            if alpha == 0.10 and method == "ordinal_projected":
                require(
                    raw == [row["conformal"]["prediction_set"] for row in rows],
                    "Deployed ordinal grade sets differ before projection",
                )
            raw = [sorted({int(grade >= 2) for grade in values}) for values in raw]
        require(raw == saved["raw_binary_sets"], "Private binary sets differ")
        match(set_metrics(raw, grades, groups), fit["set_metrics"], "set metrics")
        guarded_review = baseline_review | np.array([not values or 1 in values for values in raw])
        routes = np.where(guarded_review, ACTIONS[2], np.where(poor, ACTIONS[1], ACTIONS[0]))
        require(routes.tolist() == saved["pointwise_actions"], "Pointwise action witness differs")
        match(action_metrics(routes, grades), fit["pointwise_metrics"], "pointwise metrics")
        require(
            len(fit["allocations"]) == len(saved["allocations"]) == len(BUDGETS), "Incomplete budget matrix"
        )
        for scenario, witness, budget in zip(fit["allocations"], saved["allocations"], BUDGETS, strict=True):
            errors.append(audit_allocation(scenario, witness, budget, routes, costs, grades))
    evaluated = [error for error in errors if error is not None]
    return {
        "audit_status": "passed",
        "experiment_id": EXPERIMENT,
        "source_commit": report["source_commit"],
        "evaluation_sha256": digest(evaluation),
        "auditor_sha256": digest(Path(__file__)),
        "private_output_sha256": report["private_output_sha256"],
        "protocol_sha256": digest(PROTOCOL),
        "calibration_fits_reconstructed": len(cases),
        "historical_five_grade_fits_reconstructed": 4,
        "binary_selection_sets_reconstructed": len(cases) * len(grades),
        "pointwise_actions_reconstructed": len(cases) * len(grades),
        "development_image_hashes_verified": 336,
        "allocation_scenarios_verified": len(errors),
        "evaluated_allocations": len(evaluated),
        "overloaded_allocations": len(errors) - len(evaluated),
        "max_absolute_allocation_objective_delta": max(evaluated, default=None),
    }


if __name__ == "__main__":
    target = REPORTS / "independent_audit.json"
    require(not target.exists(), "Independent audit already exists")
    receipt = run()
    with target.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(receipt, indent=2))
