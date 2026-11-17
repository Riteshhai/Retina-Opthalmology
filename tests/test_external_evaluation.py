"""Generated external-schema fixtures; no retinal performance claims."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import pytest
import torch
from torch import nn

from qsart.data.external_evaluation import (POPULATIONS, ambiguity_labels, annotation_envelope, external_pixels,
    external_uniforms, frozen_comparisons, grading_views, infer_logits, uncertainty_views, validate_external)
from qsart.data.manifests import sha256_file
from qsart.uncertainty.conformal import fit_conformal
from qsart.uncertainty.randomized_ordinal import fit_randomized

spec = importlib.util.spec_from_file_location('evaluate_idrid', Path(__file__).parents[1]/'scripts/evaluate_idrid.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


@pytest.fixture
def frame(tmp_path):
    rows = []
    for index, grade in enumerate((0, 2, 1, 1, 3, 4)):
        path = tmp_path/f'generated_external_{index}.png'
        Image.fromarray(np.random.default_rng(index).integers(15, 240, (40, 50, 3), dtype=np.uint8)).save(path)
        rows.append({'image_id': f'generated_{index}', 'image_path': str(path), 'sha256': sha256_file(path),
            'split_group_id': ['a', 'a', 'b', 'b', 'c', 'd'][index], 'dr_grade': grade,
            'split': 'external_holdout', 'eligible_for_training': False, 'component_label_conflict': index < 2,
            'in_full_source': True, 'in_cross_source_filtered': index < 5,
            'in_cross_source_filtered_representatives': index in (0, 2, 4)})
    return pd.DataFrame(rows)


@pytest.fixture
def artifacts():
    rng = np.random.default_rng(92)
    p = rng.dirichlet(np.ones(5), 254)
    truth = rng.integers(0, 5, 254)
    rows = []
    for alpha in (.1, .05):
        for method, seed in [('lac', None), ('ordinal_greedy', None)]+[
                ('ordinal_greedy_randomized', s) for s in range(2026, 2031)]:
            metadata = {'checkpoint_sha256': 'a'*64, 'independent_temperature': 1.5, 'randomization_seed': seed}
            artifact = (fit_conformal(p, truth, alpha=alpha, method=method, metadata=metadata, empty_policy='allow')
                        if seed is None else fit_randomized(p, truth, rng.random(254), alpha=alpha, metadata=metadata))
            rows.append({'method': method, 'alpha': alpha, 'randomization_seed': seed, 'artifact': artifact})
    return rows


@pytest.mark.parametrize('mutation', ['train', 'eligible', 'string_mask', 'split_component', 'two_reps', 'bad_hash'])
def test_external_cache_scope_fails_closed(frame, mutation):
    if mutation == 'train': frame.loc[0, 'split'] = 'train'
    elif mutation == 'eligible': frame.loc[0, 'eligible_for_training'] = True
    elif mutation == 'string_mask': frame['in_full_source'] = 'True'
    elif mutation == 'split_component': frame.loc[1, 'in_cross_source_filtered'] = False
    elif mutation == 'two_reps': frame.loc[1, 'in_cross_source_filtered_representatives'] = True
    else: frame.loc[0, 'sha256'] = 'not_a_digest'
    with pytest.raises(ValueError):
        validate_external(frame)


def test_external_cache_reuse_matches_and_rejects_changed_source_or_tensor(frame, tmp_path):
    path = tmp_path/'external_inputs.pt'
    first, proof = external_pixels(frame, path, workers=1)
    second, again = external_pixels(frame, path, workers=2)
    assert torch.equal(first, second) and proof == again
    changed = torch.load(path, weights_only=True)
    changed['pixels'][0, 0, 0, 0] ^= 1
    torch.save(changed, path)
    with pytest.raises(ValueError, match='cache content mismatch'):
        external_pixels(frame, path)
    Path(frame.iloc[0].image_path).write_bytes(b'changed')
    with pytest.raises(ValueError, match='image bytes differ'):
        external_pixels(frame, tmp_path/'new.pt')


def test_inference_freezes_batchnorm_and_gradients(frame, tmp_path):
    pixels, _ = external_pixels(frame, tmp_path/'external.pt')

    class GeneratedModel(nn.Module):
        def __init__(self):
            super().__init__(); self.bn = nn.BatchNorm2d(3); self.head = nn.Linear(3, 5)

        def forward(self, values):
            return {'dr_grade': self.head(self.bn(values).mean((2, 3)))}

    model = GeneratedModel().train()
    before = {name: value.clone() for name, value in model.state_dict().items()}
    output = infer_logits(model, pixels, batch_size=2)
    assert output.shape == (6, 5) and torch.isfinite(output).all() and not output.requires_grad
    assert not model.training and all(parameter.grad is None for parameter in model.parameters())
    assert all(torch.equal(before[name], value) for name, value in model.state_dict().items())


def test_annotation_envelope_uses_original_component_labels_before_filtering(frame):
    validate_external(frame)
    allowed = ambiguity_labels(frame)
    assert allowed[0].tolist() == [True, False, True, False, False]
    assert np.array_equal(allowed[0], allowed[1])
    sets = [[0], [0, 2], [1], [], [3], [0]]
    outcome = annotation_envelope(sets, allowed)
    assert outcome['conditional_lower_count'] == 3 and outcome['conditional_upper_count'] == 4
    assert outcome['ambiguous_observations'] == 2
    probabilities = np.full((6, 5), .05); probabilities[np.arange(6), [0, 0, 1, 1, 3, 0]] = .8
    reports = grading_views(frame, probabilities)
    main = reports[POPULATIONS[2]]
    assert main['n'] == 3 and main['source_annotation_sensitivity']['ambiguous_observations'] == 1
    assert main['unambiguous_component_stratum']['images'] == 2
    for name in POPULATIONS:
        value = reports[name]; limits = value['source_annotation_sensitivity']
        assert limits['conditional_lower'] <= value['accuracy'] <= limits['conditional_upper']
    with pytest.raises(ValueError, match='Valid predicted'):
        annotation_envelope([[True]]*6, allowed)


def test_frozen_thresholds_preserve_all_methods_without_refitting(frame, artifacts):
    original = deepcopy(artifacts)
    rows = frozen_comparisons(artifacts, checkpoint_sha256='a'*64, temperature=1.5)
    p = np.random.default_rng(9).dirichlet(np.ones(5), 6)
    public, private = uncertainty_views(frame, p, rows, external_uniforms(6))
    assert len(public) == len(private) == 14 and artifacts == original
    for row in public:
        for name in POPULATIONS:
            raw, fallback = row['raw'][name], row['argmax_fallback'][name]
            assert raw['n_images'] == int(frame['in_'+name].sum())
            assert fallback['marginal_coverage'] >= raw['marginal_coverage']
            assert fallback['empty_set_count'] == 0
            bounds = raw['source_annotation_sensitivity']
            assert bounds['conditional_lower'] <= raw['marginal_coverage'] <= bounds['conditional_upper']
    with pytest.raises(ValueError, match='fourteen'):
        frozen_comparisons(rows[:-1], checkpoint_sha256='a'*64, temperature=1.5)
    with pytest.raises(ValueError):
        frozen_comparisons(rows, checkpoint_sha256='a'*64, temperature=2.)


def test_external_uniform_stream_is_distinct_and_repeatable():
    uniforms = external_uniforms(516)
    for seed, values in uniforms.items():
        children = np.random.SeedSequence(seed).spawn(3)
        np.testing.assert_array_equal(values, np.random.default_rng(children[2]).random(516))
        assert not np.array_equal(values, np.random.default_rng(children[1]).random(516))
        np.testing.assert_array_equal(values, external_uniforms(516)[seed])


def test_output_reuse_requires_identical_json_and_fixed_analysis(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, 'PRIVATE', tmp_path/'private')
    path = tmp_path/'complete.json'
    runner.save_once(path, {'value': 1})
    before = path.read_bytes()
    runner.save_once(path, {'value': 1})
    with pytest.raises(ValueError, match='Existing output differs'):
        runner.save_once(path, {'value': 2})
    assert path.read_bytes() == before
    config = json.loads((Path(__file__).parents[1]/'configs/idrid_external_v1.json').read_text())
    runner.validate_config(config)
    config['primary_model'] = 'finetune_seed2027'
    with pytest.raises(ValueError, match='fixed declaration'):
        runner.validate_config(config)


def test_completed_score_export_replays_logits_and_probabilities_exactly(tmp_path):
    values = torch.linspace(-31.7, 37.6, 30, dtype=torch.float32).reshape(6, 5)
    path = tmp_path/'scores.csv'
    runner.export_scores(path, values)
    replay = runner.import_scores(path, 6)
    assert torch.equal(replay, values.double())
    assert torch.equal((replay/1.7353179819929263).softmax(1), (values.double()/1.7353179819929263).softmax(1))
    with pytest.raises(ValueError, match='Preserve existing'):
        runner.export_scores(path, values)
    invalid = pd.read_csv(path).iloc[::-1]
    invalid.to_csv(path, index=False)
    with pytest.raises(ValueError, match='original order'):
        runner.import_scores(path, 6)
