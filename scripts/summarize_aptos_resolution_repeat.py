"""Publish the two-seed matched APTOS comparison from the completed local arms."""

from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qsart.models.checkpoint import sha256_file
from scripts.aptos_resolution_match import FIXED_EPOCH, compare_resolutions, epoch_probabilities
from scripts.aptos_resolution_repeat import ANCHORS, EXPERIMENT, ORIGINAL_SEED, SEED, read_anchor, verified_json
from scripts.run_aptos_384 import ROOT, publish_evaluation
from scripts.run_aptos_development import _grading, read_json


def load_arm(size):
    private = ROOT / "artifacts" / EXPERIMENT / f"{size}px"
    folder = ROOT / "checkpoints" / EXPERIMENT / f"seed{SEED}" / f"{size}px"
    report = read_json(private / "evaluation.json")
    lock = verified_json(private / "execution_lock.json", report["execution_lock_sha256"])
    result = verified_json(folder / "training_result.json", report["training_result_sha256"])
    manifest = private / "runtime_development.csv"
    if (
        not result["full_protocol_completed"] or result != report["run"] or result["epochs_executed"] < FIXED_EPOCH
        or lock["configuration"] != report["configuration"] or lock["configuration"]["seed"] != SEED
        or lock["configuration"]["image_size"] != size or lock["source_commit"] != report["source_commit"]
        or lock["frozen_input_bindings"] != report["frozen_input_bindings"]
        or sha256_file(manifest) != lock["runtime_manifest_sha256"]
        or sha256_file(folder / "best.pt") != report["selected_checkpoint_sha256"]
        or report["cache_sha256"] != lock["reused_input_cache_sha256"]
    ):
        raise ValueError("Expected a complete, provenance-bound seed-2027 arm")
    return report, lock, folder, manifest


def verify_matching_locks(low, high):
    if {**low["configuration"], "image_size": 384} != high["configuration"]:
        raise ValueError("The two arms must vary only input resolution")
    keys = (
        "source_commit", "frozen_input_bindings", "runtime_manifest_sha256", "runtime_calibration_roles_sha256",
        "fitting_selection_content_sha256", "imagenet_pretrained", "environment", "fresh_head_parameters_sha256",
    )
    if any(low[key] != high[key] for key in keys):
        raise ValueError("Seed-2027 arms differ in inputs, initialization or environment")


def epoch_state(folder, run, epoch):
    path = folder / f"epoch_{epoch:03}.pt"
    if sha256_file(path) != run["epoch_checkpoint_sha256"][path.name]:
        raise ValueError("Immutable epoch checkpoint differs from its digest")
    state = torch.load(path, map_location="cpu", weights_only=True)
    score = folder / f"selection_epoch_{epoch:03}.csv"
    if (state["epoch"] != epoch - 1 or state["metadata"]["optimizer_steps"] != 265 * epoch
            or sha256_file(score) != state["selection_scores_sha256"]):
        raise ValueError("Saved epoch, optimizer count or selection witness differs")
    return state, score


def score_arm(arm, folder, selection):
    run = arm["run"]
    selected = run["selected_epoch"]
    if arm["selected_checkpoint_sha256"] != run["epoch_checkpoint_sha256"][f"epoch_{selected:03}.pt"]:
        raise ValueError("The saved best checkpoint differs from the selected epoch")
    _, best = epoch_state(folder, run, selected)
    _, fixed = epoch_state(folder, run, FIXED_EPOCH)
    return SimpleNamespace(selected_probabilities=epoch_probabilities(best, selection),
                           epoch_probabilities=epoch_probabilities(fixed, selection))


def run_summary(seed, size, run, selected, fixed):
    return {
        "training_seed": seed, "image_size": size, "selected_grading": selected, "fixed_epoch_grading": fixed,
        **{key: run[key] for key in (
            "selected_epoch", "epochs_executed", "optimizer_steps", "stop_reason", "training_seconds",
            "sampled_process_and_windows_peak_rss_bytes", "history",
        )},
    }


def main():
    torch.set_num_threads(2)
    low, low_lock, low_folder, manifest = load_arm(224)
    high, high_lock, high_folder, _ = load_arm(384)
    verify_matching_locks(low_lock, high_lock)
    prior = {size: read_anchor(ROOT, size).report for size in ANCHORS}
    selection = pd.read_csv(manifest).loc[lambda frame: frame.split.eq("validation")]
    if len(selection) != 521 or selection.split_group_id.nunique() != 507:
        raise ValueError("Expected the frozen selection population")
    low_p, high_p = score_arm(low, low_folder, selection), score_arm(high, high_folder, selection)
    pairs, fixed = compare_resolutions(high_p, selection, low_p.selected_probabilities, low_p.epoch_probabilities)
    common = min(arm["run"]["epochs_executed"] for arm in (low, high))
    for epoch in range(1, common + 1):
        a, _ = epoch_state(low_folder, low["run"], epoch)
        b, _ = epoch_state(high_folder, high["run"], epoch)
        if any(not torch.equal(a[key], b[key]) for key in ("shuffle_rng", "augment_rng", "torch_rng")):
            raise ValueError("Common-epoch RNG states differ between resolutions")
    previous = prior[224]["paired_comparisons"]
    differences = {
        contrast: {metric: sorted((previous[contrast][metric]["delta"], pairs[contrast][metric]["delta"]))
                   for metric in ("accuracy", "macro_f1")}
        for contrast in pairs
    }
    runs = [run_summary(ORIGINAL_SEED, size, prior[size]["runs"][0], prior[size]["grading_before_temperature"],
                        prior[224]["fixed_epoch_comparison"][f"grading_{size}"]) for size in ANCHORS]
    for size, arm, scores in ((224, low, low_p), (384, high, high_p)):
        selected = _grading(selection.dr_grade.to_numpy(int), scores.selected_probabilities)
        if selected["accuracy"] != arm["grading_before_temperature"]["accuracy"]:
            raise ValueError("Selected replay and saved scores disagree on accuracy")
        runs.append(run_summary(SEED, size, arm["run"], selected, fixed[f"grading_{size}"]))
    report = {
        "schema_version": 1, "spec_version": "MASTER_SPEC_v1.0", "experiment_id": EXPERIMENT,
        "dataset": "APTOS_2019", "partition": "validation", "scope": "source_development_matched_resolution_repeat",
        "images": len(selection), "source_commit": low["source_commit"],
        "summary": {
            "training_seeds": [ORIGINAL_SEED, SEED], "new_training_runs": 2, "fixed_epoch": FIXED_EPOCH,
            "fixed_epoch_updates_per_arm": 1325, "common_repeat_epochs_with_matching_rng": common,
            "difference_ranges": differences, "between_seed_inference_performed": False,
        },
        "population": {"selection_images": len(selection), "fingerprint_components": 507,
                       "patient_identity_available": False, "reserved_test_inference_performed": False,
                       "calibration_inference_performed": False, "external_inference_performed": False},
        "runs": runs, "paired_comparisons": {"seed2026": previous, "seed2027": pairs},
        "original_evaluation_sha256": {str(size): info[1] for size, info in ANCHORS.items()},
        "repeat_arm_bindings": {str(size): {key: arm[key] for key in (
            "execution_lock_sha256", "training_result_sha256", "cache_sha256", "selected_checkpoint_sha256",
            "selected_replay_max_abs_delta", "preserved_artifact_count",
        )} for size, arm in ((224, low), (384, high))},
        "limitations": [
            "Both seeds reuse the development selection cohort that chose the checkpoints.",
            "Paired intervals resample fingerprint components with fixed predictions. Patient identities are unavailable.",
            "Differences across two seeds are descriptive. Training-seed uncertainty remains unestimated.",
            "This repeat measures grading. Temperature and conformal outcomes belong to the original studies.",
        ],
    }
    publish_evaluation(ROOT / "reports" / EXPERIMENT / "evaluation.json", report)
    print(f"Published {EXPERIMENT} with {common} common seed-2027 epochs")


if __name__ == "__main__":
    main()
