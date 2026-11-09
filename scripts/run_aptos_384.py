"""Run the locked APTOS direct-resize study at 384px or its matched 224px arm."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
from importlib import metadata as package_metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import uuid

import numpy as np
import pandas as pd
import psutil
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qsart.calibration.temperature import calibration_metrics, fit_temperature
from qsart.models.checkpoint import load_checkpoint, sha256_file
from qsart.training.aptos384 import Config, PREPROCESSING, evaluate, fit, prepare_store
from qsart.training.data import read_manifest, split_fingerprint
from qsart.training.finetune import batchnorm_buffer_hash, parameter_hash
from qsart.training.transfer import TransferConfig, _input_content_hash, initialize_transfer
from scripts.aptos_resolution_match import (
    EXPERIMENT as MATCHED_EXPERIMENT,
    FIXED_EPOCH,
    REFERENCE_REPORT_SHA256,
    compare_resolutions,
    epoch_probabilities,
    load_reference,
    verify_matching_source,
)
from scripts.inspect_datasets import default_aptos_root
from scripts.run_aptos_development import (
    CALIBRATION_UNIT,
    _bools,
    _grading,
    conformal_comparisons,
    read_json,
    validate_development,
)
from scripts.run_aptos_finetune import (
    BASELINE_SHA256,
    DEVELOPMENT_SHA256,
    paired_fingerprint_bootstrap,
    runtime_mapping,
)
from scripts.run_aptos_optimization import calibration_masks, ensure_aggregate, write_or_verify


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = "aptos_384_v1"
SCOPE = "source_development_384px_imagenet_finetuning"
CONFIG_PATH = ROOT / "configs/aptos_384_v1.json"
PROTOCOL = ROOT / "docs/research/APTOS_384_V1_PROTOCOL.md"
IMAGENET = ROOT / "artifacts/aptos_v1/features"
WEIGHT_FILE = IMAGENET / "torchhub/checkpoints/efficientnet_b0_rwightman-7f5810bc.pth"
WEIGHT_SHA256 = "7f5810bc96def8f7552d5b7e68d53c4786f81167d28291b21c0d90e1fca14934"
BASELINE_REPORT = ROOT / "reports/aptos_v1/evaluation.json"
PRIOR_REPORT = ROOT / "reports/aptos_optimization_v2/evaluation.json"
PRIOR_SHA256 = "66c49f12bf630493c9dc38180a73daeeae6a222523067f4ec262971b45d2310e"
COUNTS = {"train": 2115, "validation": 521, "calibration": 527}


def git(*args):
    return subprocess.check_output(
        ["git", "-c", f"safe.directory={ROOT.as_posix()}", "-c", "core.filemode=false", *args],
        cwd=ROOT,
        env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"},
    )


def source_bindings(commit, config_path, protocol):
    if (
        not commit
        or git("rev-parse", "HEAD").decode().strip() != commit
        or git("status", "--porcelain").strip()
    ):
        raise ValueError("Execution requires the exact declared clean source commit")
    paths = git("ls-files", "qsart/*.py", "api/*.py", "dashboard/*.py").decode().splitlines()
    paths += [
        Path(__file__).relative_to(ROOT).as_posix(),
        config_path.relative_to(ROOT).as_posix(),
        protocol.relative_to(ROOT).as_posix(),
        "Dockerfile",
        "requirements.txt",
        "scripts/run_aptos_384_docker.ps1",
        "scripts/aptos_resolution_match.py",
        "scripts/inspect_datasets.py",
        "scripts/run_aptos_development.py",
        "scripts/run_aptos_finetune.py",
        "scripts/run_aptos_optimization.py",
    ]
    hashes = {}
    for name in sorted(set(paths)):
        current = (ROOT / name).read_bytes()
        if current != git("show", f"{commit}:{name}"):
            raise ValueError(f"Committed execution bytes differ: {name}")
        hashes[name] = hashlib.sha256(current).hexdigest()
    return hashes


def preservation_snapshot():
    previous = read_json(ROOT / "artifacts/eyeq_hf_v1/preserved.json")
    for name, expected in previous.items():
        if sha256_file(ROOT / name) != expected:
            raise ValueError(f"Historical artifact changed: {name}")
    paths = git("ls-files", "reports").decode().splitlines()
    paths.extend(previous)
    paths += [
        WEIGHT_FILE.relative_to(ROOT).as_posix(),
        "data/manifests/aptos_v1/dataset_manifest.csv",
        "checkpoints/aptos_v1/dr_efficientnet_b0/selection_predictions.csv",
        "checkpoints/aptos_optimization_v2/seed2026/selection_predictions.csv",
    ]
    return {name: sha256_file(ROOT / name) for name in sorted(set(paths))}


def save(path, value, resume=False):
    write_or_verify(path, (json.dumps(value, indent=2, allow_nan=False) + "\n").encode(), resume)


def comparator_scores(report_path, report_sha, folder, validation, key):
    if sha256_file(report_path) != report_sha:
        raise ValueError("Frozen comparator report checksum differs")
    report = read_json(report_path)
    reference = (
        next(row for row in report["candidates"] if row["backbone"] == "efficientnet_b0")
        if key == "frozen"
        else report["runs"][0]
    )
    path = folder / "selection_predictions.csv"
    if sha256_file(path) != reference["artifact_sha256"][path.name]:
        raise ValueError("Frozen comparator score checksum differs")
    scores = pd.read_csv(path, float_precision="round_trip")
    if not np.array_equal(scores.manifest_row, validation.index) or not np.array_equal(
        scores.true_dr_grade, validation.dr_grade
    ):
        raise ValueError("Comparator scores do not match the frozen selection cohort")
    probabilities = scores[[f"dr_grade_p{i}" for i in range(5)]].to_numpy(float)
    if (
        not np.isfinite(probabilities).all()
        or (probabilities < 0).any()
        or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-5, rtol=0)
    ):
        raise ValueError("Invalid saved comparator probabilities")
    return reference, probabilities / probabilities.sum(axis=1, keepdims=True)


def initialize(config):
    if sha256_file(WEIGHT_FILE) != WEIGHT_SHA256:
        raise ValueError("Pinned ImageNet weight bytes differ")
    model, info = initialize_transfer(
        TransferConfig(backbone="efficientnet_b0"), IMAGENET, allow_download=False
    )
    torch.manual_seed(config.seed)
    model.heads["dr_grade"] = nn.Linear(model.heads["dr_grade"].in_features, 5)
    # Source URL remains in the private lock. Public reports bind its hash and weight enum.
    return model, info


def fixed_inputs(args):
    audit = read_json(args.data_audit)
    hashes = {
        key: sha256_file(path)
        for key, path in (
            ("manifest_sha256", args.full_manifest),
            ("development_manifest_sha256", args.manifest),
            ("calibration_roles_sha256", args.calibration_roles),
        )
    }
    if any(audit.get(key) != value for key, value in hashes.items()):
        raise ValueError("Frozen manifest hashes differ from the dataset audit")
    frame, roles = read_manifest(args.manifest), read_manifest(args.calibration_roles)
    validate_development(frame, roles)
    # Compare the serialized paths before rebinding them on either host OS.
    # The full manifest projection deliberately omits reserved-test labels.
    columns = ["image_id", "image_path", "split", "split_group_id", "sha256"]
    full = pd.read_csv(args.full_manifest, usecols=columns)[columns]
    development = pd.read_csv(args.manifest, usecols=columns)[columns]
    if (
        set(full.split) != {"train", "validation", "calibration", "test"}
        or full.image_id.duplicated().any()
        or full.groupby("split_group_id").split.nunique().gt(1).any()
        or not full.loc[full.split.ne("test")]
        .set_index("image_id")
        .sort_index()
        .equals(development.set_index("image_id").sort_index())
    ):
        raise ValueError("The frozen manifest has inconsistent development identities or split groups")
    return (
        frame,
        roles,
        {
            **hashes,
            "data_audit_sha256": sha256_file(args.data_audit),
            "archive_sha256": audit["archive_sha256"],
        },
    )


def run(args):
    matched = args.image_size == 224
    experiment = MATCHED_EXPERIMENT if matched else EXPERIMENT
    config_path = ROOT / "configs" / f"{experiment}.json" if matched else CONFIG_PATH
    protocol = ROOT / "docs/research/APTOS_MATCHED_RESOLUTION_V1_PROTOCOL.md" if matched else PROTOCOL
    scope = "source_development_matched_resolution_224_vs_384" if matched else SCOPE
    checkpoints = ROOT / "checkpoints" / experiment / "seed2026"
    private = ROOT / "artifacts" / experiment
    output = ROOT / "reports" / experiment / "evaluation.json"
    config = Config(image_size=args.image_size)
    if read_json(config_path) != asdict(config):
        raise ValueError("Configuration differs from the declared direct-resize protocol")
    frame, roles, inputs = fixed_inputs(args)
    if (
        inputs["development_manifest_sha256"] != DEVELOPMENT_SHA256
        or frame.split.value_counts().to_dict() != COUNTS
    ):
        raise ValueError("Expected the exact frozen APTOS development cohort")
    if sha256_file(args.aptos_root / "train.csv") != read_json(args.data_audit)["source_annotation_sha256"]:
        raise ValueError("Active source annotations differ from the audited source")
    runtime, runtime_roles = runtime_mapping(frame, roles, args.aptos_root)
    validation = runtime.loc[runtime.split.eq("validation")]
    reference = load_reference(ROOT, asdict(config), inputs, validation) if matched else None
    if reference is None:
        baseline, baseline_p = comparator_scores(
            BASELINE_REPORT,
            BASELINE_SHA256,
            ROOT / "checkpoints/aptos_v1/dr_efficientnet_b0",
            validation,
            "frozen",
        )
        prior, prior_p = comparator_scores(
            PRIOR_REPORT,
            PRIOR_SHA256,
            ROOT / "checkpoints/aptos_optimization_v2/seed2026",
            validation,
            "prior",
        )
    if sha256_file(WEIGHT_FILE) != WEIGHT_SHA256 or not all(Path(p).is_file() for p in runtime.image_path):
        raise ValueError("Expected pinned ImageNet weights and complete development image paths")
    if not args.execute:
        print(
            json.dumps(
                {
                    "status": "ready_inputs_only",
                    "experiment_id": experiment,
                    "partition_images": COUNTS,
                    "configuration": asdict(config),
                    "imagenet_sha256": WEIGHT_SHA256,
                    "training_or_inference_performed": False,
                },
                indent=2,
            ),
            flush=True,
        )
        return 0
    if output.exists() or (checkpoints.exists() and not args.resume):
        raise ValueError("Preserve existing experiment outputs")
    if psutil.virtual_memory().available < 2 * 1024**3 or shutil.disk_usage(ROOT).free < 5 * 1024**3:
        raise RuntimeError("Execution needs 2 GiB available RAM and 5 GiB available disk at launch")
    start = time.perf_counter()
    sources = source_bindings(args.source_commit, config_path, protocol)
    if reference:
        verify_matching_source(
            ROOT, reference, sources, lambda name: git("show", f"{reference.report['source_commit']}:{name}")
        )
    snapshot = preservation_snapshot()
    if reference:
        snapshot.update(reference.preserved)
    content = _input_content_hash(runtime)
    manifest = private / "runtime_development.csv"
    role_manifest = private / "runtime_calibration_roles.csv"
    manifest_bytes = runtime.to_csv(index=False, lineterminator="\n").encode()
    role_bytes = runtime_roles.to_csv(index=False, lineterminator="\n").encode()
    torch.set_num_threads(config.cpu_threads)
    model, pretrained = initialize(config)
    environment = {
        "system": platform.system(),
        "python": platform.python_version(),
        "packages": {
            name: package_metadata.version(name)
            for name in ("torch", "torchvision", "numpy", "pandas", "Pillow")
        },
    }
    lock = {
        "source_commit": args.source_commit,
        "source_file_sha256": sources,
        "configuration": asdict(config),
        "frozen_input_bindings": inputs,
        "runtime_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "runtime_calibration_roles_sha256": hashlib.sha256(role_bytes).hexdigest(),
        "development_content_sha256": content,
        "imagenet_pretrained": pretrained,
        "preserved_artifact_sha256": snapshot,
        "environment": environment,
    }
    if reference:
        for key in (
            "runtime_manifest_sha256",
            "runtime_calibration_roles_sha256",
            "development_content_sha256",
            "imagenet_pretrained",
            "environment",
        ):
            if lock[key] != reference.lock[key]:
                raise ValueError(f"Matched execution differs from the 384px reference: {key}")
        if (
            parameter_hash(model.backbone) != reference.initial_backbone_sha256
            or batchnorm_buffer_hash(model) != reference.initial_batchnorm_sha256
        ):
            raise ValueError("Initial backbone parameters or BatchNorm buffers differ")
        lock.update(
            reference_report_sha256=REFERENCE_REPORT_SHA256,
            fresh_head_parameters_sha256=parameter_hash(model.heads["dr_grade"]),
        )
    else:
        lock.update(baseline_report_sha256=BASELINE_SHA256, prior_224_report_sha256=PRIOR_SHA256)
    write_or_verify(manifest, manifest_bytes, args.resume)
    write_or_verify(role_manifest, role_bytes, args.resume)
    lock_path = private / "execution_lock.json"
    save(lock_path, lock, args.resume)
    fitting = runtime.loc[runtime.split.isin(["train", "validation"])]
    store, fit_cache = prepare_store(fitting, manifest, private / "input_store", size=config.image_size)
    provenance = {
        "data_purpose": "real_research",
        "manifest_sha256": sha256_file(manifest),
        "split_fingerprints": {name: split_fingerprint(part) for name, part in runtime.groupby("split")},
        "execution_lock_sha256": sha256_file(lock_path),
        "input_cache_sha256": fit_cache["file_sha256"],
        "source_commit": args.source_commit,
        "imagenet_weight_sha256": WEIGHT_SHA256,
        "initialization": "torchvision_imagenet_b0_and_seeded_fresh_five_way_head",
    }
    print(
        json.dumps(
            {
                "event": "execution_locked",
                "source_commit": args.source_commit,
                "fitting_images": len(fitting) - len(validation),
                "selection_images": len(validation),
            }
        ),
        flush=True,
    )
    result_path = checkpoints / "training_result.json"
    result = (
        read_json(result_path)
        if args.resume and result_path.exists()
        else fit(
            model,
            store,
            fitting,
            checkpoints,
            config,
            provenance=provenance,
            resume=args.resume and checkpoints.exists(),
        )
    )
    del model
    if not result["full_protocol_completed"]:
        print(
            json.dumps({"event": "resource_budget_incomplete", "epochs_executed": result["epochs_executed"]}),
            flush=True,
        )
        return 2
    for name, expected in result["epoch_checkpoint_sha256"].items():
        if sha256_file(checkpoints / name) != expected:
            raise ValueError("Completed epoch checkpoint changed")
    best = checkpoints / "best.pt"
    model, artifact = load_checkpoint(best)
    metadata = artifact["metadata"]
    if (
        any(metadata.get(key) != value for key, value in provenance.items())
        or artifact["epoch"] + 1 != result["selected_epoch"]
        or metadata["production_eligible"] is not False
        or sha256_file(best) != result["epoch_checkpoint_sha256"][f"epoch_{result['selected_epoch']:03}.pt"]
    ):
        raise ValueError("Selected checkpoint differs from completed training provenance")
    model.to(memory_format=torch.channels_last)
    truth = validation.dr_grade.to_numpy(dtype=int, copy=True)
    positions = np.flatnonzero(fitting.split.eq("validation").to_numpy())
    logits, _ = evaluate(model, store, positions, truth, config.microbatch_size)
    saved = pd.read_csv(
        checkpoints / f"selection_epoch_{result['selected_epoch']:03}.csv", float_precision="round_trip"
    )
    saved_logits = saved[[f"logit_{grade}" for grade in range(5)]].to_numpy(float)
    replay_delta = float(np.max(np.abs(logits.double().numpy() - saved_logits)))
    if (
        not np.array_equal(saved.manifest_row, validation.index)
        or not np.array_equal(saved.true_dr_grade, truth)
        or replay_delta > 1e-6
        or not np.array_equal(logits.numpy().argmax(1), saved_logits.argmax(1))
    ):
        raise ValueError("Selected checkpoint replay differs from saved epoch scores")

    calibration = runtime.loc[runtime.split.eq("calibration")]
    cal_store, cal_cache = prepare_store(
        calibration, manifest, private / "posthoc_store", size=config.image_size, scope="posthoc"
    )
    cal_truth = torch.tensor(calibration.dr_grade.to_numpy(int), dtype=torch.long)
    cal_logits, _ = evaluate(model, cal_store, np.arange(len(calibration)), cal_truth, config.microbatch_size)
    lookup = runtime_roles.set_index("image_id").loc[calibration.image_id]
    t_mask, cp_mask = calibration_masks(lookup)
    if int(t_mask.sum()) != 262 or int(cp_mask.sum()) != 254:
        raise ValueError("Fixed temperature/conformal role counts differ")
    temperature = fit_temperature(cal_logits[t_mask], cal_truth[t_mask])
    raw_p = logits.double().softmax(1).numpy()
    probabilities = (logits.double() / temperature).softmax(1).numpy()
    cp_p = (cal_logits[cp_mask].double() / temperature).softmax(1).numpy()
    binding = {
        "checkpoint_sha256": sha256_file(best),
        "model_training_seed": config.seed,
        "runtime_manifest_sha256": sha256_file(manifest),
        "independent_temperature": temperature,
        "temperature_split_fingerprint": split_fingerprint(calibration.loc[t_mask]),
        "conformal_representatives_fingerprint": split_fingerprint(calibration.loc[cp_mask]),
    }
    representatives = _bools(validation.is_group_representative)
    comparisons, private_sets = conformal_comparisons(
        cp_p, cal_truth[cp_mask].numpy(), probabilities, truth, representatives, binding
    )
    if reference:
        epoch_224 = epoch_probabilities(checkpoints / f"selection_epoch_{FIXED_EPOCH:03}.csv", validation)
        paired, fixed_epoch = compare_resolutions(reference, validation, raw_p, epoch_224)
        comparison_details = {
            "matched_reference": {
                "report_sha256": REFERENCE_REPORT_SHA256,
                "source_commit": reference.report["source_commit"],
                "configuration": reference.report["configuration"],
                "selected_epoch": reference.report["summary"]["selected_epoch"],
                "selection_metrics": reference.report["grading_before_temperature"],
            },
            "fixed_epoch_comparison": fixed_epoch,
        }
    else:
        paired = {}
        for name, reference_p in (("frozen_feature_baseline", baseline_p), ("prior_224_primary", prior_p)):
            paired[name] = paired_fingerprint_bootstrap(
                truth, reference_p, raw_p, validation.split_group_id.to_numpy(), repeats=2000, seed=20261109
            )
            paired[name]["direction"] = "bundled_384_configuration_minus_" + name
        comparison_details = {
            "baseline": {
                "selected_epoch": baseline["selected_epoch"],
                "selection_metrics": _grading(truth, baseline_p),
            },
            "prior_primary": {
                "selected_epoch": prior["selected_epoch"],
                "selection_metrics": _grading(truth, prior_p),
            },
        }
    score_hashes = {}
    for name, part, values in (("validation", validation, logits), ("calibration", calibration, cal_logits)):
        scores = part[["image_id", "split_group_id", "dr_grade", "is_group_representative"]].copy()
        if name == "calibration":
            scores["calibration_role"] = lookup.calibration_role.to_numpy()
        for grade in range(5):
            scores[f"logit_{grade}"] = values[:, grade].double().numpy()
        path = private / "scores" / f"{name}_scores.csv"
        write_or_verify(
            path, scores.to_csv(index=False, lineterminator="\n", float_format="%.17g").encode(), args.resume
        )
        score_hashes[path.name] = sha256_file(path)
    sets_path = private / "scores/conformal_comparisons.json"
    save(sets_path, private_sets, args.resume)
    score_hashes[sets_path.name] = sha256_file(sets_path)
    if any(sha256_file(ROOT / name) != expected for name, expected in snapshot.items()):
        raise ValueError("A preserved source or historical artifact changed")
    if source_bindings(args.source_commit, config_path, protocol) != sources:
        raise ValueError("Execution source changed")
    raw_grading, grading = _grading(truth, raw_p), _grading(truth, probabilities)
    calibration_report = {
        "temperature": temperature,
        "temperature_fit_before": calibration_metrics(cal_logits[t_mask], cal_truth[t_mask]),
        "temperature_fit_after": calibration_metrics(cal_logits[t_mask], cal_truth[t_mask], temperature),
        "selection_before": calibration_metrics(logits, truth),
        "selection_after": calibration_metrics(logits, truth, temperature),
    }
    selected = {
        **result,
        "model_version": metadata["model_version"],
        "checkpoint_sha256": sha256_file(best),
        "last_checkpoint_sha256": sha256_file(checkpoints / "last.pt"),
        "training_seed": config.seed,
        "grading_before_temperature": raw_grading,
        "grading": grading,
        "backbone_parameters_changed": metadata["backbone_parameters_changed"],
        "batchnorm_buffers_changed": metadata["batchnorm_buffers_changed"],
        "production_eligible": False,
    }
    report = {
        "schema_version": 1,
        "spec_version": "MASTER_SPEC_v1.0",
        "experiment_id": experiment,
        "dataset": "APTOS_2019",
        "partition": "validation",
        "scope": scope,
        "model_version": metadata["model_version"],
        "images": len(validation),
        "configuration": asdict(config),
        "summary": {
            "completed_runs": 1,
            "primary_training_seed": config.seed,
            "maximum_epochs": config.epochs,
            "selected_epoch": result["selected_epoch"],
            "epochs_executed": result["epochs_executed"],
            "stop_reason": result["stop_reason"],
            "selected_backbone": f"ImageNet EfficientNet-B0, {config.image_size}px",
            "conformal_comparisons": len(comparisons),
            "production_model_changed": False,
        },
        "population": {
            "partition_images": COUNTS,
            "temperature_images": int(t_mask.sum()),
            "conformal_representative_images": int(cp_mask.sum()),
            "selection_representative_images": int(representatives.sum()),
            "patient_identity_available": False,
            "reserved_test_inference_performed": False,
            "external_inference_performed": False,
        },
        "runs": [selected],
        "grading_before_temperature": raw_grading,
        "grading": grading,
        "calibration": calibration_report,
        **comparison_details,
        "paired_comparisons": paired,
        "comparisons": comparisons,
        "calibration_unit_definition": CALIBRATION_UNIT,
        "source_commit": args.source_commit,
        "execution_source_sha256": sources,
        "frozen_input_bindings": inputs,
        "execution_lock_sha256": sha256_file(lock_path),
        "private_score_sha256": score_hashes,
        "imagenet_pretrained": {
            key: pretrained[key] for key in ("enum", "file_sha256", "backbone_state_sha256")
        },
        "preprocessing": PREPROCESSING,
        "preserved_artifact_count": len(snapshot),
        "selected_replay_max_abs_delta": replay_delta,
        "fit_cache_bytes": fit_cache["bytes"],
        "calibration_cache_bytes": cal_cache["bytes"],
        "elapsed_seconds": time.perf_counter() - start,
        "environment": environment,
        "limitations": [
            "One fixed CPU training seed on reused source-development cohorts. Selection accuracy chose the checkpoint.",
            (
                "Selected checkpoints can use different epochs. The fixed epoch comparison uses equal update counts."
                if matched
                else "The comparison changes initialization, geometry, augmentation, BatchNorm and optimization together."
            ),
            "Torchvision, CPU FP32, microbatch four and the frozen project split differ from the base-paper design.",
            "Fingerprint components are an approximation for related images. Patient identities are unavailable.",
            "Temperature and conformal roles are disjoint. All reported coverage is empirical on reused selection images.",
            "The selected checkpoint is restricted to research with its direct RGB resize.",
        ],
    }
    ensure_aggregate(report)
    temporary = output.with_name(f".{output.name}.{uuid.uuid4().hex}.writing")
    save(temporary, report)
    os.link(temporary, output)
    temporary.unlink()
    print(
        json.dumps(
            {
                "event": "completed",
                "summary": report["summary"],
                "accuracy": raw_grading["accuracy"],
                "macro_f1": raw_grading["macro_f1"],
                "elapsed_seconds": report["elapsed_seconds"],
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    manifest_root = ROOT / "data/manifests/aptos_v1"
    parser.add_argument("--manifest", type=Path, default=manifest_root / "development.csv")
    parser.add_argument("--full-manifest", type=Path, default=manifest_root / "dataset_manifest.csv")
    parser.add_argument("--calibration-roles", type=Path, default=manifest_root / "calibration_roles.csv")
    parser.add_argument("--data-audit", type=Path, default=ROOT / "reports/aptos_v1/data_audit.json")
    parser.add_argument("--aptos-root", type=Path, default=default_aptos_root(ROOT))
    parser.add_argument("--image-size", type=int, choices=(224, 384), default=384)
    parser.add_argument("--source-commit")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true")
    raise SystemExit(run(parser.parse_args()))
