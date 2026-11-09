"""Bind the matched 224px arm to the published 384px development run."""

import ast
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from qsart.models.checkpoint import sha256_file
from scripts.run_aptos_development import _grading, read_json
from scripts.run_aptos_finetune import paired_fingerprint_bootstrap

EXPERIMENT = "aptos_224_matched_v1"
REFERENCE = "aptos_384_v1"
REFERENCE_REPORT_SHA256 = "746e47de7dba071c82e4ea15736daa19021a3a9d804ca42b2351d2554e635daf"
FIXED_EPOCH = 5
BOOTSTRAP_SEED = 20261109
LOGITS = [f"logit_{grade}" for grade in range(5)]
TRAINER = "qsart/training/aptos384.py"
RUNNER = "scripts/run_aptos_384.py"


@dataclass(frozen=True)
class MatchedReference:
    report: dict
    lock: dict
    selected_probabilities: np.ndarray
    epoch_probabilities: np.ndarray
    preserved: dict[str, str]
    initial_backbone_sha256: str
    initial_batchnorm_sha256: str


def _verified(path, expected):
    if sha256_file(path) != expected:
        raise ValueError(f"Frozen 384px reference checksum differs for {path.name}")


def _probabilities(frame):
    values = frame[LOGITS].to_numpy(dtype=np.float64, copy=True)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite saved logits")
    return torch.from_numpy(values).softmax(1).numpy()


def epoch_probabilities(path, selection):
    scores = pd.read_csv(path, float_precision="round_trip")
    if not np.array_equal(scores.manifest_row, selection.index) or not np.array_equal(
        scores.true_dr_grade, selection.dr_grade
    ):
        raise ValueError("Epoch scores differ from the frozen selection order")
    return _probabilities(scores)


def load_reference(root, configuration, input_bindings, selection):
    report_path = root / "reports" / REFERENCE / "evaluation.json"
    _verified(report_path, REFERENCE_REPORT_SHA256)
    report = read_json(report_path)
    private = root / "artifacts" / REFERENCE
    lock_path = private / "execution_lock.json"
    _verified(lock_path, report["execution_lock_sha256"])
    lock = read_json(lock_path)
    if configuration["image_size"] != 224 or {**configuration, "image_size": 384} != lock["configuration"]:
        raise ValueError("Matched configuration may vary only the declared image size")
    if input_bindings != lock["frozen_input_bindings"] or input_bindings != report["frozen_input_bindings"]:
        raise ValueError("Matched study requires the same frozen development cohorts and calibration roles")

    selected_path = private / "scores/validation_scores.csv"
    selected_digest = report["private_score_sha256"][selected_path.name]
    _verified(selected_path, selected_digest)
    scores = pd.read_csv(selected_path, float_precision="round_trip")
    for column in ("image_id", "split_group_id", "dr_grade", "is_group_representative"):
        if not np.array_equal(scores[column], selection[column]):
            raise ValueError("384px scores differ from the frozen selection identities, labels or groups")

    folder = root / "checkpoints" / REFERENCE / "seed2026"
    run = report["runs"][0]
    best = folder / "best.pt"
    _verified(best, run["checkpoint_sha256"])
    epoch = folder / f"epoch_{FIXED_EPOCH:03}.pt"
    epoch_digest = run["epoch_checkpoint_sha256"][epoch.name]
    _verified(epoch, epoch_digest)
    checkpoint = torch.load(epoch, map_location="cpu", weights_only=True)
    epoch_scores = folder / f"selection_epoch_{FIXED_EPOCH:03}.csv"
    epoch_score_digest = checkpoint["selection_scores_sha256"]
    _verified(epoch_scores, epoch_score_digest)
    preserved = {
        report_path: REFERENCE_REPORT_SHA256,
        lock_path: report["execution_lock_sha256"],
        selected_path: selected_digest,
        best: run["checkpoint_sha256"],
        epoch: epoch_digest,
        epoch_scores: epoch_score_digest,
    }
    return MatchedReference(
        report,
        lock,
        _probabilities(scores),
        epoch_probabilities(epoch_scores, selection),
        {path.relative_to(root).as_posix(): digest for path, digest in preserved.items()},
        checkpoint["metadata"]["initial_backbone_parameters_sha256"],
        checkpoint["metadata"]["initial_batchnorm_buffers_sha256"],
    )


def verify_matching_source(root, reference, sources, frozen_source):
    """Require matching scientific code except for the checkpoint name's resolution."""
    for name, expected in reference.lock["source_file_sha256"].items():
        if name == TRAINER or name == RUNNER or name.startswith(("api/", "dashboard/", "docs/", "configs/")):
            continue
        if name.endswith(".ps1"):
            continue
        if sources.get(name) != expected:
            raise ValueError(f"Matched scientific dependency differs: {name}")
    original = frozen_source(TRAINER)
    old_name = b'"model_version": f"aptos-384-v1-seed'
    new_name = b'"model_version": f"aptos-{config.image_size}-v1-seed'
    if original.count(old_name) != 1 or (root / TRAINER).read_bytes() != original.replace(old_name, new_name):
        raise ValueError("Trainer differs beyond the resolution in the checkpoint name")

    def initializer(source):
        node = next(
            item
            for item in ast.parse(source).body
            if isinstance(item, ast.FunctionDef) and item.name == "initialize"
        )
        return ast.dump(node, include_attributes=False)

    if initializer((root / RUNNER).read_bytes()) != initializer(frozen_source(RUNNER)):
        raise ValueError("ImageNet and fresh-head initialization code differs")


def compare_resolutions(reference, selection, selected_224, epoch_224):
    truth = selection.dr_grade.to_numpy(dtype=int, copy=True)
    groups = selection.split_group_id.to_numpy()
    pairs = {}
    for name, low, high in (
        ("selected_checkpoints", selected_224, reference.selected_probabilities),
        ("fixed_epoch_5", epoch_224, reference.epoch_probabilities),
    ):
        pairs[name] = paired_fingerprint_bootstrap(
            truth, low, high, groups, repeats=2000, seed=BOOTSTRAP_SEED
        )
        pairs[name]["direction"] = "matched_384_minus_224"
    return pairs, {
        "epoch": FIXED_EPOCH,
        "grading_224": _grading(truth, epoch_224),
        "grading_384": _grading(truth, reference.epoch_probabilities),
    }
