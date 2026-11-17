"""Generated fixtures test the extension lock; they are not retinal evidence."""
from dataclasses import asdict, replace
import hashlib
import json
from types import SimpleNamespace
import warnings

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from qsart.training.finetune import FinetuneConfig
from scripts import run_aptos_optimization as runner


def test_calibration_indices_own_memory_and_do_not_expose_pandas_views():
    lookup = pd.DataFrame({'calibration_role': ['temperature', 'temperature', 'conformal', 'conformal'],
        'is_group_representative': [True, False, True, False]})
    original = lookup.copy(deep=True)
    temperature, conformal = runner.calibration_masks(lookup)
    with warnings.catch_warnings(record=True) as observed:
        warnings.simplefilter('always')
        t, cp = torch.as_tensor(temperature), torch.as_tensor(conformal)
        assert torch.arange(4)[t].tolist() == [0, 1]
        assert torch.arange(4)[cp].tolist() == [2]
        t[0] = False
    assert temperature.flags.writeable and conformal.flags.writeable
    assert not any('not writable' in str(w.message) for w in observed)
    pd.testing.assert_frame_equal(lookup, original)


def test_extension_is_fixed_primary_seed_and_longer_schedule(tmp_path):
    path = tmp_path/'config.yaml'
    expected = replace(FinetuneConfig(), epochs=15)
    path.write_text(yaml.safe_dump(asdict(expected)))
    assert runner.fixed_config(path) == expected
    for changed in (replace(expected, seed=2027), replace(expected, epochs=4), replace(expected, backbone_lr=.001)):
        path.write_text(yaml.safe_dump(asdict(changed)))
        with pytest.raises(ValueError, match='declared 15-epoch'):
            runner.fixed_config(path)


def test_private_runtime_inputs_require_exact_explicit_resume(tmp_path):
    path = tmp_path/'private/runtime.csv'
    runner.write_or_verify(path, b'generated_fixture\n', False)
    with pytest.raises(ValueError, match='explicit --resume'):
        runner.write_or_verify(path, b'generated_fixture\n', False)
    runner.write_or_verify(path, b'generated_fixture\n', True)
    with pytest.raises(ValueError, match='differ'):
        runner.write_or_verify(path, b'changed_fixture\n', True)
    assert path.read_bytes() == b'generated_fixture\n'


@pytest.mark.parametrize('private', [{'runs': [{'image_id': 'generated'}]}, {'nested': {'patient_id': 'generated'}},
    {'source': 'C:/generated/fixture.png'}, {'source': '/generated/fixture.png'}])
def test_public_aggregate_rejects_identifiers_and_absolute_paths(private):
    with pytest.raises(ValueError, match='private identities'):
        runner.ensure_aggregate(private)
    runner.ensure_aggregate({'images': 5, 'checkpoint': 'checkpoints/generated/best.pt', 'accuracy': .2})


def test_execution_rejects_dirty_or_wrong_commit_before_inputs(monkeypatch, tmp_path):
    def output(command, **kwargs):
        return 'fixture_commit\n' if command[1] == 'rev-parse' else ' M scripts/generated.py\n'

    monkeypatch.setattr(runner.subprocess, 'check_output', output)
    for commit in ('fixture_commit', 'wrong_commit', None):
        with pytest.raises(ValueError, match='exact declared clean'):
            runner.execution_sources(tmp_path, SimpleNamespace(source_commit=commit))


def test_previous_primary_comparator_is_bound_and_paired_not_best_seed(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    folder = tmp_path/'checkpoints/aptos_finetune_v1/seed2026'
    folder.mkdir(parents=True)
    scores = pd.DataFrame({'manifest_row': [3, 8], 'true_dr_grade': [0, 4]})
    p = np.array([[.6, .1, .1, .1, .1], [.1, .1, .1, .1, .6]])
    for i in range(5):
        scores[f'dr_grade_p{i}'] = p[:, i]
    path = folder/'selection_predictions.csv'
    scores.to_csv(path, index=False)
    report = {'summary': {'primary_training_seed': 2026}, 'runs': [
        {'training_seed': 2027, 'accuracy': 1.},
        {'training_seed': 2026, 'artifact_sha256': {'selection_predictions.csv': hashlib.sha256(path.read_bytes()).hexdigest()}}]}
    prior = tmp_path/'previous.json'
    prior.write_text(json.dumps(report))
    monkeypatch.setattr(runner, 'PRIOR_SHA256', hashlib.sha256(prior.read_bytes()).hexdigest())
    validation = pd.DataFrame({'dr_grade': [0, 4]}, index=[3, 8])
    row, observed = runner.prior_primary(prior, validation)
    assert row['training_seed'] == 2026
    np.testing.assert_allclose(observed, p)
    with pytest.raises(ValueError, match='not paired'):
        runner.prior_primary(prior, validation.iloc[::-1])
    path.write_text('changed generated fixture')
    with pytest.raises(ValueError, match='comparator scores changed'):
        runner.prior_primary(prior, validation)
