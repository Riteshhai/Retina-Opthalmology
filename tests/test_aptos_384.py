"""Generated fixtures check the disk reader, weighted updates and epoch recovery."""

from copy import deepcopy
from dataclasses import replace

import numpy as np
import pandas as pd
from PIL import Image
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from qsart.models.checkpoint import sha256_file
from qsart.training import aptos384


class LinearClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Linear(3, 5)

    def forward(self, images):
        return {"dr_grade": self.head(images)}


class SmallClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 4, 3, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.2),
        )
        self.head = nn.Linear(4, 5)

    def forward(self, images):
        return {"dr_grade": self.head(self.backbone(images))}


@pytest.fixture
def development(tmp_path):
    rng = np.random.default_rng(19)
    rows = []
    for index in range(16):
        path = tmp_path / f"fixture_{index}.png"
        Image.fromarray(rng.integers(0, 256, (23, 38, 3), dtype=np.uint8)).save(path)
        rows.append(
            {
                "image_path": str(path),
                "split": "train" if index < 11 else "validation",
                "dr_grade": index % 5,
                "sha256": sha256_file(path),
                "split_group_id": f"fixture_{index}",
                "official_split": "official_train",
                "eligible_for_training": True,
            }
        )
    frame = pd.DataFrame(rows)
    manifest = tmp_path / "development.csv"
    frame.to_csv(manifest, index=False)
    return frame, manifest


def test_weighted_accumulation_uses_full_window_weight_sum():
    torch.manual_seed(12)
    first = LinearClassifier()
    second = deepcopy(first)
    images = torch.randn(5, 3)
    labels = torch.tensor([0, 0, 4, 1, 3])
    weights = torch.tensor([0.5, 2.0, 1.0, 3.0, 6.0])
    reference = F.cross_entropy(first(images)["dr_grade"], labels, weight=weights)
    reference.backward()
    numerator, denominator = aptos384.weighted_microbatch_backward(second, images, labels, weights, 2)
    assert numerator / denominator == pytest.approx(reference.item(), abs=2e-7)
    for expected, actual in zip(first.parameters(), second.parameters()):
        torch.testing.assert_close(actual.grad, expected.grad, rtol=1e-6, atol=2e-7)


def test_disk_reader_matches_direct_resize_and_refuses_corruption(development, tmp_path):
    frame, manifest = development
    store, info = aptos384.prepare_store(frame, manifest, tmp_path / "store", size=32)
    with Image.open(frame.iloc[3].image_path) as image:
        expected = image.convert("RGB").resize((32, 32), Image.Resampling.BILINEAR)
    assert store.image(3).tobytes() == expected.tobytes()
    assert info["bytes"] == len(frame) * 32 * 32 * 3
    assert not info["reused"]
    _, reused = aptos384.prepare_store(frame, manifest, tmp_path / "store", size=32)
    assert reused["reused"] and reused["file_sha256"] == info["file_sha256"]
    with store.path.open("r+b") as stream:
        first = stream.read(1)
        stream.seek(0)
        stream.write(bytes([first[0] ^ 1]))
    with pytest.raises(ValueError, match="cache provenance/checksum"):
        aptos384.prepare_store(frame, manifest, tmp_path / "store", size=32)


@pytest.mark.parametrize("role", ["test", "confirmation", "external", "calibration"])
def test_fitting_store_rejects_excluded_roles_before_opening_images(development, tmp_path, role):
    frame, manifest = development
    frame.loc[0, ["split", "image_path"]] = [role, str(tmp_path / "missing.png")]
    with pytest.raises(ValueError, match="Forbidden partition"):
        aptos384.prepare_store(frame, manifest, tmp_path / "forbidden", size=32)
    assert not (tmp_path / "forbidden").exists()


def test_epoch_recovery_reproduces_trajectory_and_checks_score_witness(development, tmp_path, monkeypatch):
    frame, manifest = development
    store, _ = aptos384.prepare_store(frame, manifest, tmp_path / "store", size=32)
    config = aptos384.Config(
        image_size=32,
        epochs=3,
        microbatch_size=2,
        optimizer_batch_size=4,
        cpu_threads=1,
        minimum_available_bytes=1,
    )
    provenance = {"data_purpose": "software_test", "manifest_sha256": sha256_file(manifest)}
    torch.manual_seed(5)
    initial = SmallClassifier()
    uninterrupted = tmp_path / "complete"
    expected = aptos384.fit(deepcopy(initial), store, frame, uninterrupted, config, provenance=provenance)
    recovered = tmp_path / "recovered"
    save_checkpoint = aptos384._save_checkpoint

    def save_then_interrupt(path, value):
        save_checkpoint(path, value)
        raise InterruptedError("Fixture interruption after an immutable epoch")

    with monkeypatch.context() as patch:
        patch.setattr(aptos384, "_save_checkpoint", save_then_interrupt)
        with pytest.raises(InterruptedError):
            aptos384.fit(deepcopy(initial), store, frame, recovered, config, provenance=provenance)
    first_hash = sha256_file(recovered / "epoch_001.pt")
    with pytest.raises(ValueError, match="configuration/source/epoch chain"):
        aptos384.fit(
            deepcopy(initial),
            store,
            frame,
            recovered,
            replace(config, seed=99),
            provenance=provenance,
            resume=True,
        )
    scores = recovered / "selection_epoch_001.csv"
    original = scores.read_bytes()
    scores.write_bytes(original + b"\n")
    with pytest.raises(ValueError, match="selection-score witness"):
        aptos384.fit(deepcopy(initial), store, frame, recovered, config, provenance=provenance, resume=True)
    scores.write_bytes(original)
    actual = aptos384.fit(
        deepcopy(initial), store, frame, recovered, config, provenance=provenance, resume=True
    )
    assert sha256_file(recovered / "epoch_001.pt") == first_hash
    assert actual["optimizer_steps"] == expected["optimizer_steps"] == 9
    assert actual["selected_epoch"] == expected["selected_epoch"]
    for epoch in range(1, 4):
        path = f"selection_epoch_{epoch:03}.csv"
        assert (recovered / path).read_bytes() == (uninterrupted / path).read_bytes()
    left, right = [torch.load(folder / "last.pt", weights_only=True) for folder in (uninterrupted, recovered)]
    for name in left["model_state"]:
        assert torch.equal(left["model_state"][name], right["model_state"][name])
    assert right["metadata"]["batchnorm_buffers_changed"] is True
    assert right["metadata"]["production_eligible"] is False
