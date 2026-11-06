"""Synthetic fixtures verify software only; never retinal research performance."""
import json

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from qsart.calibration.temperature import calibration_metrics, fit_temperature, write_calibration
from qsart.inference.engine import InferenceEngine
from qsart.models import build_model
from qsart.models.checkpoint import ArtifactError, load_checkpoint
from qsart.training.data import image_to_tensor, read_manifest
from qsart.training.trainer import TrainingConfig, train


@pytest.fixture(scope="module")
def software_artifacts(tmp_path_factory):
    root = tmp_path_factory.mktemp("software_only_models")
    rows = []
    rng = np.random.default_rng(42)
    for index in range(22):
        image = Image.fromarray(rng.integers(25, 230, (48, 48, 3), dtype=np.uint8))
        path = root / f"fixture_{index}.png"
        image.save(path)
        split = "train" if index < 10 else "validation" if index < 16 else "calibration"
        rows.append({"image_path": str(path), "dr_grade": index % 5, "quality_grade": index % 2,
                     "split": split, "subject_id": f"software_subject_{index}"})
    manifest = root / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    config = TrainingConfig(image_size=32, epochs=1, batch_size=4, cpu_threads=1, augment=False)
    for task in ("dr", "quality"):
        train(manifest, root / task, task=task, config=config, data_purpose="software_test")
    calibration = root / "calibration.json"
    write_calibration(root / "dr/best.pt", root / "quality/best.pt", manifest, calibration,
                       allow_software_test=True, cpu_threads=1)
    return root, manifest, calibration


@pytest.mark.parametrize("backbone", ["resnet18", "mobilenet_v3_small"])
def test_cpu_shapes_and_probabilities(backbone):
    torch.set_num_threads(1)
    model = build_model(backbone).eval()
    with torch.inference_mode():
        for count in (1, 2):
            logits = model(torch.zeros(count, 3, 32, 32))["dr_grade"]
            assert logits.shape == (count, 5)
            assert torch.allclose(logits.softmax(1).sum(1), torch.ones(count))
    with pytest.raises(ValueError):
        model(torch.zeros(1, 32, 32))


def test_invalid_models_and_config():
    with pytest.raises(ValueError):
        build_model("unsupported")
    with pytest.raises(ValueError):
        build_model(heads={"dr_grade": 4})
    with pytest.raises(ValueError):
        TrainingConfig(image_size=16).validate()
    with pytest.raises(ValueError):
        TrainingConfig(mixed_precision=True).validate()


def test_checkpoint_requires_actual_training_and_test_opt_in(software_artifacts, tmp_path):
    root, _, _ = software_artifacts
    with pytest.raises(ArtifactError, match="Software-test"):
        load_checkpoint(root / "dr/best.pt")
    model, checkpoint = load_checkpoint(root / "dr/best.pt", allow_software_test=True)
    assert checkpoint["metadata"]["optimizer_steps"] > 0
    reloaded, _ = load_checkpoint(root / "dr/best.pt", allow_software_test=True)
    with torch.inference_mode():
        image = torch.rand(1, 3, 32, 32)
        assert torch.equal(model(image)["dr_grade"], reloaded(image)["dr_grade"])
    checkpoint["metadata"]["trained"] = False
    path = tmp_path / "untrained.pt"
    torch.save(checkpoint, path)
    with pytest.raises(ArtifactError, match="training marker"):
        load_checkpoint(path, allow_software_test=True)


def test_engine_calibration_and_population_budget(software_artifacts):
    root, _, calibration = software_artifacts
    with pytest.raises(ArtifactError):
        InferenceEngine(root / "dr/best.pt", root / "quality/best.pt", calibration)
    engine = InferenceEngine(root / "dr/best.pt", root / "quality/best.pt", calibration,
                             allow_software_test=True, cpu_threads=1)
    image = Image.new("RGB", (48, 48), "red")
    results = engine.predict_images([image] * 4, second_pass_budget=0.25)
    assert len(results) == 4
    assert sum(row["second_pass_activated"] for row in results) == 1
    for row in results:
        assert len(row["grade_probabilities"]) == 5
        assert sum(row["grade_probabilities"]) == pytest.approx(1.0)
        assert 0 <= row["quality"]["poor_probability"] <= 1
        assert 0 <= row["uncertainty"]["entropy"] <= 1.00001
        assert row["processing_time_ms"] > 0
        assert row["data_purpose"] == "software_test"
    assert engine.predict_images([]) == []
    assert not engine.predict_images([image], second_pass_budget=0.25)[0]["second_pass_activated"]
    with pytest.raises(ValueError):
        engine.predict_images([image], second_pass_budget=1.1)
    with pytest.raises(ValueError):
        engine.predict_images([image], second_pass_policy="all", second_pass_budget=0.25)


def test_wrong_calibration_is_rejected(software_artifacts, tmp_path):
    root, _, calibration = software_artifacts
    content = json.loads(calibration.read_text())
    content["dr"]["checkpoint_sha256"] = "mismatch"
    wrong = tmp_path / "calibration.json"
    wrong.write_text(json.dumps(content))
    with pytest.raises(ArtifactError, match="checksum"):
        InferenceEngine(root / "dr/best.pt", root / "quality/best.pt", wrong,
                         allow_software_test=True, cpu_threads=1)


def test_temperature_reduces_nll_and_metrics_are_finite():
    logits = torch.tensor([[9., 0., 0.], [0., 9., 0.], [0., 0., 9.], [9., 0., 0.]])
    labels = torch.tensor([0, 1, 1, 2])
    temperature = fit_temperature(logits, labels)
    before = calibration_metrics(logits, labels)
    after = calibration_metrics(logits, labels, temperature)
    assert temperature > 0
    assert after["nll"] <= before["nll"]
    assert 0 <= after["ece"] <= 1
    assert 0 <= after["brier"] <= 2
    assert sum(item["count"] for item in after["reliability_bins"]) == 4


def test_shared_preprocessing_consistency():
    image = Image.new("RGB", (48, 64), "orange")
    assert torch.equal(image_to_tensor(image, 32), image_to_tensor(image.copy(), 32))
    assert image_to_tensor(image, 32).shape == (3, 32, 32)


def test_resume_and_manifest_overlap(software_artifacts, tmp_path):
    root, manifest, _ = software_artifacts
    result = train(manifest, root / "dr", task="dr", data_purpose="software_test",
                   config=TrainingConfig(image_size=32, epochs=2, batch_size=4, cpu_threads=1, augment=False),
                   resume=root / "dr/last.pt")
    assert result["epochs_executed"] == 2
    frame = pd.read_csv(manifest)
    frame.loc[10, "subject_id"] = frame.loc[0, "subject_id"]
    duplicate = tmp_path / "overlap.csv"
    frame.to_csv(duplicate, index=False)
    with pytest.raises(ValueError, match="Cross-split overlap"):
        read_manifest(duplicate)


@pytest.mark.parametrize("column,value", [("annotation_status", "quarantined"),
                                          ("eligible_for_training", False), ("readable", False)])
def test_unverified_manifest_records_rejected(tmp_path, column, value):
    manifest = tmp_path / "unverified.csv"
    pd.DataFrame([{"image_path": "image.png", "split": "train", column: value}]).to_csv(manifest, index=False)
    with pytest.raises(ValueError, match=column):
        read_manifest(manifest)
