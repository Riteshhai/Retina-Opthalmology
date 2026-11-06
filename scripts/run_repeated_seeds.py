"""Execute the predeclared six development-only runs and publish aggregates.

Training requires --execute and a frozen source commit. Without --execute this
script only validates and summarizes the six already completed private runs.
Individual predictions, image paths and patient identifiers are never exported.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from dataclasses import asdict
import json
from pathlib import Path
import re
import subprocess
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, log_loss, recall_score
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qsart.data.development import defect_mapping_report
from qsart.evaluation import grade_metrics
from qsart.models.checkpoint import sha256_file
from qsart.training.data import read_manifest, split_fingerprint
from qsart.training.transfer import (HEAD_RNG_VERSION, TransferConfig, _input_content_hash,
                                     source_provenance, train_transfer)

SEEDS = (2026, 2027, 2028)
MANIFEST_SHA256 = "45e9f2d34ed54a12056b57a808d03cfaf29ccc970d148e9eb107b79cf5ac3e0d"
EXPECTED_COUNTS = {"train": 848, "validation": 168, "calibration": 168}
EXPECTED_PATIENTS = {"train": 212, "validation": 42, "calibration": 42}
QUALITY_HEADS = {"quality_grade": 2, "clarity_class": 5,
                 "field_definition_class": 5, "artifact_class": 6}
PRESERVED = {
    "selected_dr": "checkpoints/development_v1/selected/dr.pt",
    "selected_quality": "checkpoints/development_v1/selected/quality.pt",
    "selected_calibration": "checkpoints/development_v1/selected/calibration.json",
    "selected_model_card": "checkpoints/development_v1/selected/model_card.json",
}


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _configs(root: Path) -> dict[int, TransferConfig]:
    original = yaml.safe_load((root / "configs/development_v1.yaml").read_text(encoding="utf-8"))
    result = {}
    for seed in SEEDS:
        path = root / f"configs/repeated_seeds_v1_seed{seed}.yaml"
        values = yaml.safe_load(path.read_text(encoding="utf-8"))
        if values != {**original, "seed": seed}:
            raise ValueError("Repeated-seed configs may change only the declared seed")
        config = TransferConfig(**values)
        config.validate()
        result[seed] = config
    return result


def _validate_population(manifest: Path) -> tuple[pd.DataFrame, dict]:
    if sha256_file(manifest) != MANIFEST_SHA256:
        raise ValueError("The predeclared frozen development manifest has changed")
    frame = read_manifest(manifest)
    counts = {name: len(part) for name, part in frame.groupby("split")}
    patients = {name: int(part.subject_id.nunique()) for name, part in frame.groupby("split")}
    if counts != EXPECTED_COUNTS or patients != EXPECTED_PATIENTS:
        raise ValueError("Development image/patient counts differ from the frozen protocol")
    if not frame.official_split.eq("official_train").all():
        raise ValueError("Only official-training development records are accepted")
    selected = frame.loc[frame.split.isin(["train", "validation"])].copy()
    return frame, {
        "manifest_sha256": MANIFEST_SHA256,
        "split_counts": counts, "split_patient_counts": patients,
        "split_fingerprints": {name: split_fingerprint(part) for name, part in frame.groupby("split")},
        "fitting_and_selection_input_content_sha256": _input_content_hash(selected),
        "calibration_images_read_or_predicted": False,
    }


def _support(frame: pd.DataFrame, head: str, classes: int) -> dict:
    return {
        split: [int(frame.loc[frame.split.eq(split), head].eq(index).sum())
                for index in range(classes)]
        for split in ("train", "validation")
    }


def _verified_run(folder: Path, task: str, config: TransferConfig, frame: pd.DataFrame,
                  population: dict, provenance: dict, cache_root: Path) -> dict:
    checkpoint_path = folder / "best.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    metadata = checkpoint["metadata"]
    if metadata != _json(folder / "training_metadata.json"):
        raise ValueError("Checkpoint and exported training metadata differ")
    heads = {"dr_grade": 5} if task == "dr" else QUALITY_HEADS
    expected = {
        "trained": True, "data_purpose": "real_research", "task": task,
        "backbone": "mobilenet_v3_small", "confirmation_used": False,
        "training_config": asdict(config), "seed": config.seed, "heads": heads,
        "head_rng_version": HEAD_RNG_VERSION, "head_initialization_seed": config.seed,
        "manifest_sha256": population["manifest_sha256"],
        "split_counts": population["split_counts"],
        "split_fingerprints": population["split_fingerprints"],
        "source_sha256": provenance["source_sha256"],
        "original_train_images": 848, "train_feature_views": 1696,
        "selection_metric": "macro_f1_then_unweighted_nll", "backbone_frozen": True,
    }
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError("Completed run does not match the frozen seed/task/provenance design")
    code_provenance = metadata["code_provenance"]
    if (code_provenance["source_file_sha256"] != provenance["source_file_sha256"] or
            code_provenance["git_revision"] != provenance["git_revision"]):
        raise ValueError("Run source files or source revision differ from the frozen execution")
    cache = metadata["feature_cache_provenance"]
    if cache["metadata"]["input_content_sha256"] != population["fitting_and_selection_input_content_sha256"]:
        raise ValueError("Run feature cache uses different actual input image bytes")
    cache_path = cache_root / f"features_mobilenet_v3_small_{metadata['feature_cache_key']}.pt"
    if sha256_file(cache_path) != cache["cache_file_sha256"]:
        raise ValueError("Feature-cache file changed since training")
    cached = torch.load(cache_path, map_location="cpu", weights_only=True)
    if cached["metadata"] != cache["metadata"] or cached["cache_key"] != metadata["feature_cache_key"]:
        raise ValueError("Feature-cache metadata differs from the trained checkpoint")
    predictions = pd.read_csv(folder / "selection_predictions.csv").sort_values("manifest_row")
    selected = frame.loc[frame.split.eq("validation")]
    identities = predictions.manifest_row.to_numpy()
    if (predictions.manifest_row.duplicated().any() or len(predictions) != 168 or
            not np.array_equal(identities, selected.index.to_numpy())):
        raise ValueError("Selection prediction row identities differ from the frozen population")
    metrics, supports = {}, {}
    for head, classes in heads.items():
        truth_values = selected[head].fillna(-100).to_numpy(dtype=int)
        if not np.array_equal(predictions[f"true_{head}"].to_numpy(), truth_values):
            raise ValueError("Selection prediction labels differ from genuine manifest labels")
        probabilities = predictions[[f"{head}_p{i}" for i in range(classes)]].to_numpy(dtype=float)
        sums = probabilities.sum(axis=1)
        if (not np.isfinite(probabilities).all() or (probabilities < 0).any() or
                (probabilities > 1).any() or not np.allclose(sums, 1, rtol=0, atol=1e-5)):
            raise ValueError("Saved probability rows are invalid")
        probabilities /= sums[:, None]
        predicted = probabilities.argmax(1)
        if not np.array_equal(predicted, predictions[f"predicted_{head}"].to_numpy()):
            raise ValueError("Saved predicted grades disagree with probability argmax")
        mask = truth_values >= 0
        truth, probability, prediction = truth_values[mask], probabilities[mask], predicted[mask]
        values = {"accuracy": float(accuracy_score(truth, prediction)),
                  "macro_f1": float(f1_score(truth, prediction, labels=list(range(classes)),
                                            average="macro", zero_division=0)),
                  "nll": float(log_loss(truth, probability, labels=list(range(classes))))}
        if any(not np.isclose(value, metadata["selection_metrics"][head][name], atol=1e-5, rtol=0)
               for name, value in values.items()):
            raise ValueError("Computed metrics differ from checkpoint selection metrics")
        if task == "dr":
            all_metrics = grade_metrics(truth, probability)
            values.update({name: all_metrics[name] for name in (
                "quadratic_weighted_kappa", "ece", "brier", "referable_auroc", "referable_auprc")})
            values["referable_sensitivity"] = float(recall_score(truth >= 2, prediction >= 2, zero_division=0))
        metrics[head] = values
        supports[head] = _support(frame, head, classes)
    history = pd.read_csv(folder / "training_history.csv")
    result = _json(folder / "training_result.json")
    if (result["data_purpose"] != "real_research" or not result["backbone_unchanged"] or
            result["epochs_executed"] != len(history) or result["selected_epoch"] != metadata["epoch"] or
            result["selected_metrics"] != metadata["selection_metrics"]):
        raise ValueError("Training result/history differs from the selected checkpoint")
    elapsed = _json(folder / "execution.json")
    if (elapsed.get("source_commit") != provenance["git_revision"] or elapsed.get("task") != task or
            elapsed.get("seed") != config.seed or not np.isfinite(elapsed.get("elapsed_seconds", np.nan)) or
            elapsed["elapsed_seconds"] <= 0):
        raise ValueError("Execution record does not match the completed seed/task/source")
    return {
        "task": task, "seed": config.seed, "model_version": metadata["model_version"],
        "selected_epoch": metadata["epoch"], "epochs_executed": len(history),
        "optimizer_steps_at_selected_epoch": metadata["optimizer_steps"],
        "total_optimizer_steps": result["total_optimizer_steps"],
        "elapsed_seconds": elapsed["elapsed_seconds"],
        "extraction_seconds": cache["extraction_seconds"], "feature_cache_reused": cache["reused"],
        "metrics": metrics, "class_supports": supports,
        "training_source_provenance": code_provenance,
        "training_config": metadata["training_config"], "head_rng_version": metadata["head_rng_version"],
        "feature_cache_key": metadata["feature_cache_key"],
        "feature_cache_file_sha256": cache["cache_file_sha256"],
        "input_content_sha256": cache["metadata"]["input_content_sha256"],
        "pretrained_provenance": metadata["pretrained_provenance"],
        "torch_version": metadata["torch_version"], "torchvision_version": metadata["torchvision_version"],
        "artifact_sha256": {name: sha256_file(folder / name) for name in (
            "best.pt", "last.pt", "training_metadata.json", "training_history.csv",
            "selection_predictions.csv", "selection_metrics.json", "training_result.json", "execution.json")},
    }


def summarize_runs(runs: list[dict]) -> dict:
    result = {}
    for task in ("dr", "quality"):
        group = [run for run in runs if run["task"] == task]
        if sorted(run["seed"] for run in group) != list(SEEDS):
            raise ValueError("Exactly the three predeclared seeds are required for each task")
        if any(run["metrics"].keys() != group[0]["metrics"].keys() for run in group):
            raise ValueError("All seed runs must contain the same predefined output heads")
        result[task] = {}
        for head, metrics in group[0]["metrics"].items():
            result[task][head] = {}
            for name in metrics:
                values = np.asarray([run["metrics"][head][name] for run in group], dtype=float)
                if not np.isfinite(values).all():
                    raise ValueError("A repeated-seed summary outcome is missing or nonfinite")
                result[task][head][name] = {"mean": float(values.mean()),
                                          "sample_standard_deviation": float(values.std(ddof=1)), "seeds": 3}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    root = args.project_root.resolve()
    protocol = (root / args.protocol).resolve()
    if not protocol.is_relative_to(root) or not protocol.is_file():
        parser.error("Protocol must be an existing file inside the project")
    if not re.fullmatch(r"[0-9a-f]{40}", args.source_commit):
        parser.error("Use the full forty-character frozen source commit")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    if revision != args.source_commit:
        raise ValueError("Current HEAD differs from the requested frozen training-source commit")
    if args.execute and subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip():
        raise ValueError("Freeze and commit source/config/protocol changes before genuine training")
    configs = _configs(root)
    frame, population = _validate_population(root / "data/processed/deepdrid/development_v1.csv")
    provenance = source_provenance()
    preserved = {name: sha256_file(root / path) for name, path in PRESERVED.items()}
    source_files = {**provenance["source_file_sha256"],
                    "scripts/run_repeated_seeds.py": sha256_file(Path(__file__).resolve()),
                    str(protocol.relative_to(root)).replace("\\", "/"): sha256_file(protocol)}
    source_files.update({f"configs/repeated_seeds_v1_seed{seed}.yaml":
                         sha256_file(root / f"configs/repeated_seeds_v1_seed{seed}.yaml") for seed in SEEDS})
    cache_root = root / "artifacts/development_v1/features"
    checkpoint_root = root / "checkpoints/repeated_seeds_v1"
    plans = [(task, seed, checkpoint_root / f"{task}_seed{seed}")
             for task in ("dr", "quality") for seed in SEEDS]
    if args.execute and any(folder.exists() and any(folder.iterdir()) for _, _, folder in plans):
        raise ValueError("Refusing existing repeated-seed outputs; preserve partial/completed evidence")
    mappings = defect_mapping_report()
    runs = []
    for task, seed, folder in plans:
        if any(sha256_file(root / name) != digest for name, digest in source_files.items()):
            raise ValueError("Frozen source, protocol or config bytes changed during execution")
        if args.execute:
            log_root = root / "artifacts/repeated_seeds_v1"
            log_root.mkdir(parents=True, exist_ok=True)
            start = time.perf_counter()
            with (log_root / f"{task}_seed{seed}.log").open("w", encoding="utf-8") as log:
                with redirect_stdout(log):
                    train_transfer(root / "data/processed/deepdrid/development_v1.csv", folder, cache_root,
                                   task=task, config=configs[seed], quality_heads=QUALITY_HEADS,
                                   label_mappings=mappings if task == "quality" else {})
            (folder / "execution.json").write_text(json.dumps({
                "elapsed_seconds": time.perf_counter() - start, "source_commit": args.source_commit,
                "protocol_sha256": sha256_file(protocol), "task": task, "seed": seed,
            }, indent=2), encoding="utf-8")
        run = _verified_run(folder, task, configs[seed], frame, population, provenance, cache_root)
        runs.append(run)
        print(json.dumps({"task": task, "seed": seed, "metrics": run["metrics"],
                          "selected_epoch": run["selected_epoch"]}), flush=True)
    if (any(sha256_file(root / path) != preserved[name] for name, path in PRESERVED.items()) or
            any(sha256_file(root / name) != digest for name, digest in source_files.items())):
        raise ValueError("Preserved deployment artifacts or frozen study source bytes changed")
    if (len({run["feature_cache_key"] for run in runs}) != 1 or
            len({run["pretrained_provenance"]["backbone_state_sha256"] for run in runs}) != 1 or
            len({run["pretrained_provenance"]["file_sha256"] for run in runs}) != 1):
        raise ValueError("All six runs must share identical frozen features and official backbone weights")
    report = {
        "schema_version": 1, "experiment_id": "repeated_seeds_v1",
        "scope": "development_selection", "dataset": "DeepDRiD", "partition": "validation",
        "protocol_sha256": sha256_file(protocol), "source_commit": args.source_commit,
        "source_file_sha256": source_files, "population": population,
        "head_rng_version": HEAD_RNG_VERSION,
        "config_sha256": {str(seed): sha256_file(root / f"configs/repeated_seeds_v1_seed{seed}.yaml")
                          for seed in SEEDS},
        "runs": runs, "summary": summarize_runs(runs),
        "preserved_selected_artifact_sha256": preserved,
        "limitations": [
            "Three training seeds on one fixed development selection population; no cohort/domain variability estimate.",
            "New cache-independent head initialization procedure; historical seed2026 is excluded from this summary.",
            "Uncalibrated first-pass probabilities; no temperature/conformal fitting or consumed confirmation prediction.",
            "Mean and sample SD across all three seeds; no best-seed selection, confidence interval or deployment switch.",
            "Execution times include hashing/cache checks and shared-workstation activity; no matched speed comparison.",
        ],
    }
    serialized = json.dumps(report, indent=2, allow_nan=False)
    if str(root) in serialized or any(key in serialized for key in ('"image_path"', '"subject_id"', '"manifest_row"')):
        raise ValueError("Public aggregate report contains a forbidden private field")
    output = root / "reports/repeated_seeds_v1/summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(serialized, encoding="utf-8")
    print(json.dumps({"report": "reports/repeated_seeds_v1/summary.json", "runs": len(runs),
                      "summary": report["summary"]}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
