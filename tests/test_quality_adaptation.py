"""Generated software fixtures exercise isolation and recovery, not retinal accuracy."""
from copy import deepcopy
from dataclasses import replace
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import pytest
from sklearn.metrics import average_precision_score, f1_score, log_loss, roc_auc_score
import torch
from torch import nn

from qsart.models.checkpoint import ArtifactError, FORMAT_VERSION, load_checkpoint, sha256_file
from qsart.models.networks import build_model
from qsart.training import quality_adaptation as qa
from qsart.training.data import MEAN, STD
from qsart.training.finetune import batchnorm_buffer_hash, parameter_hash, prepare_input_cache


class TinyQuality(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.BatchNorm2d(4),
            nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(.2))
        self.heads = nn.ModuleDict({name: nn.Linear(4, size) for name, size in qa.HEADS.items()})

    def forward(self, images):
        features = self.backbone(images)
        return {name: head(features) for name, head in self.heads.items()}


@pytest.fixture
def fixture(tmp_path):
    torch.set_num_threads(1)
    records = []
    for index in range(19):
        image = tmp_path/f'generated_{index}.png'
        Image.fromarray(np.random.default_rng(index).integers(10, 255, (41, 53, 3), dtype=np.uint8)).save(image)
        records.append({'image_path': str(image), 'sha256': sha256_file(image),
            'split': 'train' if index < 12 else 'validation' if index < 18 else 'calibration',
            'split_group_id': f'fixture_{index//2}', 'subject_id': f'fixture_{index//2}',
            'official_split': 'official_train', 'annotation_status': 'verified', 'dr_grade': index%5,
            **{name: index%size for name, size in qa.HEADS.items()}})
    frame = pd.DataFrame(records)
    manifest = tmp_path/'generated_manifest.csv'
    frame.to_csv(manifest, index=False)
    selected = frame.loc[frame.split != 'calibration']
    cache, _ = prepare_input_cache(selected, manifest, tmp_path/'cache', image_size=32, workers=1)
    base = {'data_purpose': 'software_test', 'backbone': 'software_tiny', 'heads': qa.HEADS,
            'poor_class_indices': [0], 'manifest_sha256': sha256_file(manifest),
            'split_fingerprints': {'train': 'software_train', 'validation': 'software_selection'}}
    torch.manual_seed(1234)
    initial = TinyQuality()
    return manifest, frame, selected, cache, base, initial


def assert_nested_equal(first, second):
    if isinstance(first, torch.Tensor):
        assert torch.equal(first, second)
    elif isinstance(first, dict):
        assert first.keys() == second.keys()
        for key in first:
            assert_nested_equal(first[key], second[key])
    elif isinstance(first, (tuple, list)):
        assert len(first) == len(second)
        for a, b in zip(first, second, strict=True):
            assert_nested_equal(a, b)
    else:
        assert first == second


def test_matched_arms_have_equal_exposure_but_different_backbone_updates(fixture, tmp_path):
    _, _, frame, cache, base, initial = fixture
    config = qa.QualityConfig(image_size=32, epochs=3, batch_size=4, cpu_threads=1)
    initial_backbone = parameter_hash(initial.backbone)
    initial_bn = batchnorm_buffer_hash(initial)
    states = []
    for arm in qa.ARMS:
        model = deepcopy(initial)
        result = qa.fit_quality(model, cache, frame, tmp_path/arm, config, arm=arm, base_metadata=base)
        assert result['total_optimizer_steps'] == 9 and result['epochs_executed'] == 3
        assert (parameter_hash(model.backbone) == initial_backbone) == (arm == 'frozen')
        assert batchnorm_buffer_hash(model) == initial_bn
        assert parameter_hash(model.heads) != parameter_hash(initial.heads)
        assert all(p.requires_grad == (arm == 'adapted') for p in model.backbone.parameters())
        states.append(torch.load(tmp_path/arm/'state.pt', weights_only=True))
        with pytest.raises(ArtifactError, match='Software-test'):
            load_checkpoint(tmp_path/arm/'best.pt')
        with pytest.raises(ValueError, match='already exists'):
            qa.fit_quality(deepcopy(initial), cache, frame, tmp_path/arm, config, arm=arm, base_metadata=base)
        assert qa.fit_quality(deepcopy(initial), cache, frame, tmp_path/arm, config, arm=arm,
                              base_metadata=base, resume=True) == result
    for a, b in zip(states[0]['history'], states[1]['history'], strict=True):
        assert a['order_sha256'] == b['order_sha256'] and a['flip_sha256'] == b['flip_sha256']
        assert a['learning_rates']['heads'] == b['learning_rates']['heads']
    assert len(pd.read_csv(tmp_path/'adapted/selection_epoch_predictions.csv')) == 24


def test_atomic_resume_reproduces_every_completed_state_and_rejects_cross_arm(fixture, tmp_path, monkeypatch):
    _, _, frame, cache, base, initial = fixture
    config = qa.QualityConfig(image_size=32, epochs=3, batch_size=4, cpu_threads=1)
    qa.fit_quality(deepcopy(initial), cache, frame, tmp_path/'whole', config, arm='adapted', base_metadata=base)
    atomic = qa._atomic_torch

    def stop_after_committed_epoch(value, path):
        atomic(value, path)
        if value['epoch'] == 1:
            raise InterruptedError('simulated interruption after atomic epoch commit')

    monkeypatch.setattr(qa, '_atomic_torch', stop_after_committed_epoch)
    with pytest.raises(InterruptedError):
        qa.fit_quality(deepcopy(initial), cache, frame, tmp_path/'resume', config, arm='adapted', base_metadata=base)
    monkeypatch.setattr(qa, '_atomic_torch', atomic)
    for bad_arm, bad_config, bad_base in [('frozen', config, base),
        ('adapted', replace(config, epochs=4), base), ('adapted', config, {**base, 'source_sha256': 'changed'})]:
        with pytest.raises(ValueError, match='mismatch'):
            qa.fit_quality(deepcopy(initial), cache, frame, tmp_path/'resume', bad_config,
                           arm=bad_arm, base_metadata=bad_base, resume=True)
    qa.fit_quality(deepcopy(initial), cache, frame, tmp_path/'resume', config,
                   arm='adapted', base_metadata=base, resume=True)
    a = torch.load(tmp_path/'whole/state.pt', weights_only=True)
    b = torch.load(tmp_path/'resume/state.pt', weights_only=True)
    for key in a:
        if key != 'elapsed_training_seconds':
            assert_nested_equal(a[key], b[key])
    assert (tmp_path/'whole/selection_predictions.csv').read_bytes() == (tmp_path/'resume/selection_predictions.csv').read_bytes()


def test_final_export_failure_can_resume_without_repeating_updates(fixture, tmp_path, monkeypatch):
    _, _, frame, cache, base, initial = fixture
    config = qa.QualityConfig(image_size=32, epochs=1, batch_size=4, cpu_threads=1)
    export = qa._export
    monkeypatch.setattr(qa, '_export', lambda *args: (_ for _ in ()).throw(InterruptedError('export interrupted')))
    with pytest.raises(InterruptedError):
        qa.fit_quality(deepcopy(initial), cache, frame, tmp_path/'run', config, arm='adapted', base_metadata=base)
    state_sha = sha256_file(tmp_path/'run/state.pt')
    monkeypatch.setattr(qa, '_export', export)
    result = qa.fit_quality(deepcopy(initial), cache, frame, tmp_path/'run', config,
                            arm='adapted', base_metadata=base, resume=True)
    assert result['total_optimizer_steps'] == 3 and sha256_file(tmp_path/'run/state.pt') == state_sha
    (tmp_path/'run/selection_predictions.csv').write_text('tampered', encoding='utf-8')
    with pytest.raises(ValueError, match='artifact changed'):
        qa.verify_seal(tmp_path/'run')


@pytest.mark.parametrize('role', ['test', 'holdout', 'confirmation', 'external_holdout'])
def test_reserved_roles_rejected_before_weights_or_pixels(fixture, tmp_path, role):
    manifest, frame, *_ = fixture
    frame.loc[0, 'split'] = role
    # Keep all known views in that new role, so the role guard is exercised.
    frame.loc[1, 'split'] = role
    frame.image_path = [f'missing_generated_image_{i}.png' for i in range(len(frame))]
    frame.to_csv(manifest, index=False)
    with pytest.raises(ValueError, match='development train/validation'):
        qa.train_quality(manifest, tmp_path/'missing.pt', tmp_path/'out', tmp_path/'cache', data_purpose='software_test')


@pytest.mark.parametrize('value', [None, -1, 1.5, 5])
def test_incomplete_or_invalid_defect_labels_fail_before_fitting(fixture, value):
    _, _, frame, *_ = fixture
    frame = frame.copy()
    frame['clarity_class'] = frame.clarity_class.astype(float)
    frame.loc[0, 'clarity_class'] = value
    with pytest.raises(ValueError, match='Complete integer labels'):
        qa.validate_frame(frame, selected_only=True, data_purpose='software_test')


def test_source_mapping_and_group_guards(fixture):
    _, _, frame, *_ = fixture
    frame = frame.copy()
    frame['dataset_source'] = 'DeepDRiD_regular_fundus'
    for name in ('eligible_for_training', 'readable', 'exists'):
        frame[name] = True
    for target, codes, column in [('clarity_class', [1,4,6,8,10], 'clarity'),
        ('field_definition_class', [1,4,6,8,10], 'field_definition'),
        ('artifact_class', [0,1,4,6,8,10], 'artifact')]:
        frame[column] = frame[target].map(dict(enumerate(codes)))
    qa.validate_frame(frame, selected_only=True)
    frame.loc[0, 'clarity'] = 10
    with pytest.raises(ValueError, match='disagree with source'):
        qa.validate_frame(frame, selected_only=True)
    frame.loc[0, 'clarity'] = 1
    frame.loc[12, 'split_group_id'] = frame.loc[0, 'split_group_id']
    with pytest.raises(ValueError, match='crosses development'):
        qa.validate_frame(frame, selected_only=True)


def test_balanced_selector_and_poor_positive_direction_are_explicit():
    metrics = {name: {'macro_f1': .4, 'nll': 1.} for name in qa.HEADS}
    other = deepcopy(metrics)
    metrics['quality_grade']['macro_f1'] = .9
    for name in qa.HEADS:
        other[name]['macro_f1'] = .6
    assert qa.selection_score(other) > qa.selection_score(metrics)
    truth = np.array([0, 0, 0, 1, 1, 1])
    logits = np.array([[4.,1.], [2.,1.], [0.,2.], [1.,3.], [0.,2.], [2.,1.]])
    probability = torch.softmax(torch.tensor(logits), 1).numpy()
    result = qa.metrics_from_logits(logits, truth, 2)
    assert result['poor_class_index'] == 0
    assert result['poor_auroc'] == roc_auc_score(truth==0, probability[:,0])
    assert result['poor_auprc'] == average_precision_score(truth==0, probability[:,0])
    assert result['poor_sensitivity'] == 2/3 and result['poor_specificity'] == 2/3
    assert result['macro_f1'] == f1_score(truth, probability.argmax(1), average='macro')
    assert np.isclose(result['nll'], log_loss(truth, probability))


def test_component_bootstrap_retains_all_views_and_common_seed_stream():
    # Each subject contributes two identical views; subject resampling has exactly
    # the same statistic as the independently computed one-row-per-subject case.
    groups = np.repeat(['a', 'b', 'c', 'd'], 2)
    truth = {name: np.repeat(np.array([0, 1, 0, 1]), 2) for name in qa.HEADS}
    first = {name: np.repeat(np.array([0, 0, 0, 1]), 2) for name in qa.HEADS}
    second = {name: truth[name].copy() for name in qa.HEADS}
    a = qa.paired_component_changes(truth, first, second, groups, draws=100, seed=123)
    b = qa.paired_component_changes({k:v[::2] for k,v in truth.items()}, {k:v[::2] for k,v in first.items()},
        {k:v[::2] for k,v in second.items()}, groups[::2], draws=100, seed=123)
    assert a == b and a['metrics']['quality_grade_accuracy']['difference'] == .25
    assert a['bootstrap_seed'] == 123 and 'seed' not in a
    assert {'seed': 2026, **a}['seed'] == 2026
    null = qa.paired_component_changes(truth, first, first, groups, draws=100, seed=123)
    assert null['component_draw_sha256'] == a['component_draw_sha256']
    assert all(value == 0 for row in null['metrics'].values() for value in row.values())


def test_real_warm_adapter_cache_rng_and_production_rejection(fixture, tmp_path, monkeypatch):
    manifest, _, *_ = fixture
    from qsart.training import finetune
    torch.manual_seed(123)
    model = build_model('mobilenet_v3_small', qa.HEADS, pretrained=False)
    metadata = {'trained': True, 'optimizer_steps': 1, 'data_purpose': 'software_test', 'task': 'quality',
        'heads': qa.HEADS, 'poor_class_indices': [0], 'backbone': 'mobilenet_v3_small',
        'image_size': 32, 'normalization_mean': MEAN, 'normalization_std': STD,
        'preprocessing': 'fundus_crop_rgb_v1', 'manifest_sha256': sha256_file(manifest),
        'split_fingerprints': {}, 'model_version': 'generated_software_warm_state'}
    initial = tmp_path/'initial.pt'
    torch.save({'format_version': FORMAT_VERSION, 'metadata': metadata, 'model_state': model.state_dict()}, initial)
    preprocessing = finetune.preprocess_image

    def consume_cache_rng(*args, **kwargs):
        torch.rand(7)
        return preprocessing(*args, **kwargs)

    monkeypatch.setattr(finetune, 'preprocess_image', consume_cache_rng)
    config = qa.QualityConfig(image_size=32, epochs=1, batch_size=6, cpu_threads=1, cache_workers=1)
    for name in ('miss', 'hit'):
        qa.train_quality(manifest, initial, tmp_path/name, tmp_path/'adapter_cache', config=config,
                         arm='adapted', data_purpose='software_test')
    a = torch.load(tmp_path/'miss/state.pt', weights_only=True)
    b = torch.load(tmp_path/'hit/state.pt', weights_only=True)
    assert_nested_equal(a['history'], b['history'])
    assert_nested_equal(a['model_state'], b['model_state'])
    _, trained = load_checkpoint(tmp_path/'hit/best.pt', allow_software_test=True)
    assert trained['metadata']['model_version'].startswith('quality-adaptation-v1-adapted-')
    with pytest.raises(ArtifactError, match='Software-test'):
        load_checkpoint(tmp_path/'hit/best.pt')


def test_public_report_guard_and_locked_config():
    path = Path(__file__).resolve().parents[1]/'scripts/run_quality_adaptation.py'
    spec = importlib.util.spec_from_file_location('quality_runner', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ensure_aggregate({'metrics': {'accuracy': .5}, 'hash': 'abc'})
    for payload in ({'subject_id': 'private'}, {'nested': ['C:\\private\\image.png']}, {'metric': float('nan')}):
        with pytest.raises(ValueError):
            module.ensure_aggregate(payload)
    config = qa.QualityConfig()
    assert config.epochs == 15 and list(qa.HEADS.values()) == [2,5,5,6]
    for bad in (replace(config, head_lr=float('inf')), replace(config, seed=True), replace(config, channels_last=1)):
        with pytest.raises(ValueError):
            bad.validate()
    assert json.loads(json.dumps({'poor_class_indices': [0]}))['poor_class_indices'] == [0]


def test_paired_seed_metadata_correction_preserves_every_measured_value():
    from scripts.correct_quality_seed_metadata import corrected_report
    original = {'schema_version': 'qsart.quality_adaptation.v1',
        'source_commit': '2d4243c977a0669776a5dc356d6d6e3166732a12',
        'runs': [{'seed': s, 'arm': a, 'selected_metrics': {'generated_metric': .5}}
                 for s in qa.SEEDS for a in qa.ARMS],
        'paired_changes': [{'seed': 20261107, 'metrics': {'difference': float(i)}} for i in range(3)]}
    before = deepcopy(original)
    result = corrected_report(original)
    assert original == before and result['runs'] == original['runs']
    assert [row['seed'] for row in result['paired_changes']] == list(qa.SEEDS)
    for old, new in zip(original['paired_changes'], result['paired_changes'], strict=True):
        assert new['bootstrap_seed'] == old['seed'] and new['metrics'] == old['metrics']
    original['paired_changes'][0]['seed'] = 999
    with pytest.raises(ValueError, match='known metadata-only defect'):
        corrected_report(original)
