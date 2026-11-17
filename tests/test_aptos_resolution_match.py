"""Generated fixtures exercise the matched-study controls."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from scripts import aptos_resolution_match as match


def test_only_checkpoint_label_change_is_allowed_in_matched_trainer(tmp_path):
    original = b'    "model_version": f"aptos-384-v1-seed{config.seed}",\n    loss.backward()\n'
    trainer = tmp_path / match.TRAINER
    trainer.parent.mkdir(parents=True)
    updated = original.replace(b"384", b"{config.image_size}")
    trainer.write_bytes(updated)
    runner = tmp_path / match.RUNNER
    runner.parent.mkdir(parents=True)
    initializer = b"def initialize(config):\n    return config.seed\n"
    runner.write_bytes(initializer)
    reference = SimpleNamespace(
        lock={"source_file_sha256": {match.TRAINER: "fixture", match.RUNNER: "fixture"}}
    )
    frozen = {match.TRAINER: original, match.RUNNER: initializer}
    match.verify_matching_source(tmp_path, reference, {}, frozen.__getitem__)

    trainer.write_bytes(updated.replace(b"loss.backward()", b"(loss * 2).backward()"))
    with pytest.raises(ValueError, match="Trainer differs"):
        match.verify_matching_source(tmp_path, reference, {}, frozen.__getitem__)


def test_epoch_scores_refuse_reordered_selection(tmp_path):
    selection = pd.DataFrame({"dr_grade": [1, 2]}, index=[10, 11])
    path = tmp_path / "epoch_scores.csv"
    pd.DataFrame({"manifest_row": [11, 10], "true_dr_grade": [2, 1]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="selection order"):
        match.epoch_probabilities(path, selection)


def test_both_matched_contrasts_report_384_minus_224():
    labels = np.tile(np.arange(5), 2)
    selection = pd.DataFrame({"dr_grade": labels, "split_group_id": np.arange(len(labels))})
    correct = np.eye(5)[labels] * 0.8 + 0.04
    wrong = np.roll(correct, 1, axis=1)
    reference = SimpleNamespace(selected_probabilities=wrong, epoch_probabilities=correct)
    paired, fixed = match.compare_resolutions(reference, selection, correct, wrong)
    assert paired["selected_checkpoints"]["accuracy"]["delta"] == -1
    assert paired["fixed_epoch_5"]["accuracy"]["delta"] == 1
    assert all(row["direction"] == "matched_384_minus_224" for row in paired.values())
    assert fixed["epoch"] == 5
    assert fixed["grading_224"]["accuracy"] == 0 and fixed["grading_384"]["accuracy"] == 1
