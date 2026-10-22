"""Generated fixtures exercise cache-independent training, not retinal accuracy."""
from dataclasses import replace

import numpy as np
import pandas as pd
from PIL import Image
import pytest
import torch

from qsart.models.checkpoint import load_checkpoint
from qsart.training.transfer import HEAD_RNG_VERSION, TransferConfig, train_transfer
from scripts.run_repeated_seeds import summarize_runs


@pytest.mark.parametrize("task", ["dr", "quality"])
def test_cache_miss_and_hit_have_identical_training_for_fixed_seed(tmp_path, task):
    rows = []
    for index in range(18):
        path = tmp_path / f"fixture_{index}.png"
        pixels = np.random.default_rng(index).integers(0, 256, (40, 40, 3), dtype=np.uint8)
        Image.fromarray(pixels).save(path)
        rows.append({
            "image_path": str(path), "split": "train" if index < 10 else
            "validation" if index < 16 else "calibration", "subject_id": f"fixture_{index}",
            "official_split": "official_train", "dr_grade": index % 5,
            "quality_grade": index % 2, "clarity_class": index % 5,
            "field_definition_class": (index + 1) % 5, "artifact_class": index % 6,
        })
    manifest = tmp_path / "fixture_manifest.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    config = TransferConfig(image_size=32, pretrained=False, num_workers=0, epochs=2,
                            batch_size=8, cpu_threads=1, patience=2, seed=2027)
    heads = {"quality_grade": 2, "clarity_class": 5, "field_definition_class": 5,
             "artifact_class": 6}
    arguments = {"task": task, "config": config, "data_purpose": "software_test"}
    if task == "quality":
        arguments["quality_heads"] = heads
    first = train_transfer(manifest, tmp_path / "miss", tmp_path / "cache", **arguments)
    second = train_transfer(manifest, tmp_path / "hit", tmp_path / "cache", **arguments)
    _, miss = load_checkpoint(tmp_path / "miss/best.pt", allow_software_test=True)
    _, hit = load_checkpoint(tmp_path / "hit/best.pt", allow_software_test=True)
    assert not miss["metadata"]["feature_cache_provenance"]["reused"]
    assert hit["metadata"]["feature_cache_provenance"]["reused"]
    assert miss["metadata"]["feature_cache_key"] == hit["metadata"]["feature_cache_key"]
    assert miss["metadata"]["head_rng_version"] == HEAD_RNG_VERSION
    assert hit["metadata"]["head_initialization_seed"] == config.seed
    assert first["selected_metrics"] == second["selected_metrics"]
    assert first["selected_epoch"] == second["selected_epoch"]
    assert miss["history"] == hit["history"]
    assert all(torch.equal(value, hit["model_state"][name])
               for name, value in miss["model_state"].items())
    assert ((tmp_path / "miss/selection_predictions.csv").read_bytes() ==
            (tmp_path / "hit/selection_predictions.csv").read_bytes())

    # Distinct seeds remain distinct research replicates rather than silently
    # reseeding every run to a fixed global constant.
    arguments["config"] = replace(config, seed=2028)
    train_transfer(manifest, tmp_path / "other_seed", tmp_path / "cache", **arguments)
    _, other = load_checkpoint(tmp_path / "other_seed/best.pt", allow_software_test=True)
    assert other["metadata"]["head_initialization_seed"] == 2028
    assert any(not torch.equal(value, other["model_state"][name])
               for name, value in hit["model_state"].items() if name.startswith("heads."))


def _summary_fixture():
    return [{"task": task, "seed": seed, "metrics": {"fixture_head": {"accuracy": (index + 1) / 4}}}
            for task in ("dr", "quality") for index, seed in enumerate((2026, 2027, 2028))]


def test_seed_summary_includes_every_seed_and_uses_sample_standard_deviation():
    result = summarize_runs(_summary_fixture())
    for task in ("dr", "quality"):
        assert result[task]["fixture_head"]["accuracy"] == {
            "mean": 0.5, "sample_standard_deviation": 0.25, "seeds": 3}


def test_seed_summary_rejects_missing_duplicate_or_nonfinite_outcomes():
    rows = _summary_fixture()
    with pytest.raises(ValueError, match="three predeclared seeds"):
        summarize_runs(rows[:-1])
    rows[-1]["seed"] = 2027
    with pytest.raises(ValueError, match="three predeclared seeds"):
        summarize_runs(rows)
    rows = _summary_fixture()
    rows[0]["metrics"]["fixture_head"]["accuracy"] = float("nan")
    with pytest.raises(ValueError, match="nonfinite"):
        summarize_runs(rows)
