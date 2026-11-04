"""Explicit generated fixtures verify fine-tuning isolation, never retinal accuracy."""
from dataclasses import replace
import hashlib

import numpy as np
import pandas as pd
from PIL import Image
import pytest
import torch
from torch import nn

from qsart.models import build_model
from qsart.models.checkpoint import ArtifactError, FORMAT_VERSION, load_checkpoint
from qsart.training.data import MEAN, STD, image_to_tensor
from qsart.training.finetune import (FinetuneConfig, batchnorm_buffer_hash, fit_finetune,
    normalize_cached, parameter_hash, prepare_input_cache, train_finetune, train_with_fixed_batchnorm)


@pytest.fixture
def source(tmp_path):
    records = []
    for index in range(16):
        path = tmp_path/f'generated_{index}.png'
        pixels = np.random.default_rng(index).integers(10, 255, (43, 59, 3), dtype=np.uint8)
        pixels[:, :7] = 0
        Image.fromarray(pixels).save(path)
        records.append({'image_path': str(path), 'split': 'train' if index < 10 else
                        'validation' if index < 15 else 'calibration', 'dr_grade': index % 5,
                        'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                        'split_group_id': f'generated_group_{index}', 'official_split': 'official_train',
                        'annotation_status': 'verified'})
    manifest = tmp_path/'generated_manifest.csv'
    frame = pd.DataFrame(records)
    frame.to_csv(manifest, index=False)
    return manifest, frame


def test_canonical_cache_matches_established_preprocessing_exactly_and_verifies_hits(source, tmp_path):
    manifest, frame = source
    selected = frame.loc[frame.split != 'calibration']
    first, first_info = prepare_input_cache(selected, manifest, tmp_path/'cache', image_size=32)
    second, second_info = prepare_input_cache(selected, manifest, tmp_path/'cache', image_size=32)
    assert first_info['reused'] is False and second_info['reused'] is True
    assert torch.equal(first['images'], second['images'])
    for index in (0, 4, 14):
        with Image.open(selected.iloc[index].image_path) as image:
            expected = image_to_tensor(image, 32)
        assert torch.equal(normalize_cached(first['images'][index:index+1])[0], expected)
    assert 'calibration' not in first['metadata']['partitions']


@pytest.mark.parametrize('role', ['calibration', 'test', 'holdout', 'confirmation'])
def test_canonical_cache_rejects_nonfitting_roles_before_opening_images(source, tmp_path, role):
    manifest, frame = source
    selected = frame.loc[frame.split != 'calibration'].copy()
    selected.loc[0, 'split'] = role
    selected.loc[0, 'image_path'] = '/never_open_this_software_fixture.png'
    with pytest.raises(ValueError, match='development rows only'):
        prepare_input_cache(selected, manifest, tmp_path/'cache')


def test_canonical_cache_detects_changed_original_and_tampered_cached_pixels(source, tmp_path):
    manifest, frame = source
    selected = frame.loc[frame.split != 'calibration']
    cached, info = prepare_input_cache(selected, manifest, tmp_path/'cache', image_size=32)
    cached['images'][0, 0, 0, 0] ^= 1
    torch.save(cached, tmp_path/'cache'/f"canonical_uint8_{info['cache_key']}.pt")
    with pytest.raises(ValueError, match='pixels/shape/hash'):
        prepare_input_cache(selected, manifest, tmp_path/'cache', image_size=32)
    Image.new('RGB', (43, 59), 'black').save(selected.iloc[0].image_path)
    with pytest.raises(ValueError, match='Input image changed'):
        prepare_input_cache(selected, manifest, tmp_path/'cache', image_size=32)


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.BatchNorm2d(4),
            nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(.2))
        self.heads = nn.ModuleDict({'dr_grade': nn.Linear(4, 5)})

    def forward(self, inputs):
        return {'dr_grade': self.heads['dr_grade'](self.backbone(inputs))}


def fixture_metadata(manifest):
    return {'data_purpose': 'software_test', 'backbone': 'efficientnet_b0',
        'manifest_sha256': hashlib.sha256(manifest.read_bytes()).hexdigest(),
        'split_fingerprints': {'train': 'generated_train', 'validation': 'generated_selection'}}


def test_finetune_changes_backbone_and_bn_affine_but_never_running_buffers(source, tmp_path):
    manifest, frame = source
    selected = frame.loc[frame.split != 'calibration']
    cache, _ = prepare_input_cache(selected, manifest, tmp_path/'cache', image_size=32)
    torch.manual_seed(123)
    model = TinyModel()
    model.backbone.requires_grad_(False)
    before_parameters = parameter_hash(model.backbone)
    before_buffers = batchnorm_buffer_hash(model)
    before_affine = model.backbone[1].weight.detach().clone()
    config = FinetuneConfig(image_size=32, epochs=2, patience=2, batch_size=5, cpu_threads=1)
    result = fit_finetune(model, cache, selected, tmp_path/'run', config,
                          base_metadata=fixture_metadata(manifest))
    assert result['total_optimizer_steps'] == 4 and result['selected_epoch'] >= 1
    assert parameter_hash(model.backbone) != before_parameters
    assert batchnorm_buffer_hash(model) == before_buffers
    assert not torch.equal(model.backbone[1].weight, before_affine)
    assert all(value.requires_grad for value in model.parameters())
    epochs = pd.read_csv(tmp_path/'run/selection_epoch_predictions.csv')
    assert len(epochs) == 10 and set(epochs.epoch) == {1, 2}
    checkpoint = torch.load(tmp_path/'run/best.pt', weights_only=True)
    assert checkpoint['metadata']['backbone_parameters_changed'] is True
    assert checkpoint['metadata']['batchnorm_buffers_unchanged'] is True
    with pytest.raises(ValueError, match='already exists'):
        fit_finetune(model, cache, selected, tmp_path/'run', config,
                     base_metadata=fixture_metadata(manifest))


def test_training_mode_keeps_stochastic_layers_active_and_bn_eval():
    model = TinyModel().eval()
    train_with_fixed_batchnorm(model)
    assert model.training and model.backbone.training
    assert model.backbone[-1].training is True
    assert model.backbone[1].training is False
    assert model.backbone[1].weight.requires_grad


def test_versioned_model_names_and_resume_cannot_cross_studies(source, tmp_path):
    """A genuinely optimized generated checkpoint keeps its declared study identity."""
    manifest, _ = source
    torch.set_num_threads(1)
    torch.manual_seed(123)
    initial = build_model('efficientnet_b0', pretrained=False)
    meta = {**fixture_metadata(manifest), 'trained': True, 'optimizer_steps': 1, 'task': 'dr',
        'heads': {'dr_grade': 5}, 'image_size': 32, 'normalization_mean': MEAN, 'normalization_std': STD,
        'preprocessing': 'fundus_crop_rgb_v1', 'model_version': 'generated-software-fixture'}
    path = tmp_path/'generated_initial.pt'
    torch.save({'format_version': FORMAT_VERSION, 'metadata': meta, 'model_state': initial.state_dict()}, path)
    config = FinetuneConfig(image_size=32, epochs=1, patience=1, batch_size=5, cpu_threads=1)
    train_finetune(manifest, path, tmp_path/'run', tmp_path/'cache', config=config,
                   data_purpose='software_test', model_version_prefix='aptos-optimization-v2')
    _, checkpoint = load_checkpoint(tmp_path/'run/best.pt', allow_software_test=True)
    assert checkpoint['metadata']['model_version'].startswith('aptos-optimization-v2-efficientnet_b0-')
    with pytest.raises(ValueError, match='model_version_prefix'):
        train_finetune(manifest, path, tmp_path/'run', tmp_path/'cache', config=config,
                       data_purpose='software_test', resume_last=True)
    with pytest.raises(ValueError, match='namespace'):
        train_finetune(manifest, path, tmp_path/'new', tmp_path/'cache', config=config,
                       data_purpose='software_test', model_version_prefix='unrecorded-study')


def test_warm_start_roundtrip_and_cache_miss_hit_training_states_are_identical(source, tmp_path, monkeypatch):
    from qsart.training import finetune
    manifest, _ = source
    torch.set_num_threads(1)
    torch.manual_seed(123)
    initial = build_model('efficientnet_b0', pretrained=False)
    meta = {**fixture_metadata(manifest), 'trained': True, 'optimizer_steps': 1, 'task': 'dr',
        'heads': {'dr_grade': 5}, 'image_size': 32, 'normalization_mean': MEAN, 'normalization_std': STD,
        'preprocessing': 'fundus_crop_rgb_v1', 'model_version': 'generated-software-fixture'}
    initial_path = tmp_path/'generated_initial.pt'
    torch.save({'format_version': FORMAT_VERSION, 'metadata': meta, 'model_state': initial.state_dict()}, initial_path)
    normal_preprocess = finetune.preprocess_image

    def random_advancing_preprocess(*args, **kwargs):
        torch.rand(17)  # Simulate incidental RNG consumption only on cache misses.
        return normal_preprocess(*args, **kwargs)

    monkeypatch.setattr(finetune, 'preprocess_image', random_advancing_preprocess)
    config = FinetuneConfig(image_size=32, epochs=1, patience=1, batch_size=5, cpu_threads=1, cache_workers=1)
    for name in ('miss', 'hit'):
        train_finetune(manifest, initial_path, tmp_path/name, tmp_path/'cache', config=config,
                       data_purpose='software_test')
    with pytest.raises(ArtifactError, match='Software-test'):
        load_checkpoint(tmp_path/'miss/best.pt')
    _, first = load_checkpoint(tmp_path/'miss/best.pt', allow_software_test=True)
    _, second = load_checkpoint(tmp_path/'hit/best.pt', allow_software_test=True)
    assert first['history'] == second['history']
    assert first['metadata']['canonical_cache_provenance']['reused'] is False
    assert second['metadata']['canonical_cache_provenance']['reused'] is True
    assert all(torch.equal(value, second['model_state'][name]) for name, value in first['model_state'].items())
    third_config = replace(config, seed=2027)
    train_finetune(manifest, initial_path, tmp_path/'different_seed', tmp_path/'cache', config=third_config,
                   data_purpose='software_test')
    _, third = load_checkpoint(tmp_path/'different_seed/best.pt', allow_software_test=True)
    assert any(not torch.equal(value, third['model_state'][name]) for name, value in first['model_state'].items())


@pytest.mark.parametrize('role', ['test', 'holdout', 'confirmation'])
def test_reserved_manifest_rejected_before_loading_initial_weights(source, tmp_path, role):
    manifest, frame = source
    frame.loc[0, 'split'] = role
    frame.to_csv(manifest, index=False)
    with pytest.raises(ValueError, match='development rows only'):
        train_finetune(manifest, tmp_path/'missing.pt', tmp_path/'output', tmp_path/'cache')


def test_optimizer_and_normalization_inputs_fail_closed():
    with pytest.raises(ValueError, match='Finite'):
        replace(FinetuneConfig(), backbone_lr=float('nan')).validate()
    with pytest.raises(ValueError, match='Positive'):
        replace(FinetuneConfig(), head_lr=0).validate()
    with pytest.raises(ValueError, match='uint8'):
        normalize_cached(torch.zeros(1, 3, 32, 32))
    with pytest.raises(ValueError, match='workers'):
        replace(FinetuneConfig(), cache_workers=3).validate()


def test_channels_last_preserves_fixed_input_geometry_and_eval_outputs():
    torch.set_num_threads(1)
    torch.manual_seed(23)
    first, second = TinyModel().eval(), TinyModel().eval()
    second.load_state_dict(first.state_dict())
    second.to(memory_format=torch.channels_last)
    pixels = torch.randint(0, 255, (3, 32, 32, 3), dtype=torch.uint8)
    normalized = normalize_cached(pixels)
    nchw = normalized.contiguous()
    nhwc_memory = normalized.contiguous(memory_format=torch.channels_last)
    assert torch.equal(nchw, nhwc_memory)
    with torch.inference_mode():
        assert torch.allclose(first(nchw)['dr_grade'], second(nhwc_memory)['dr_grade'], atol=1e-6, rtol=1e-6)


def test_partial_resume_restores_optimizer_scheduler_and_both_rng_streams(source, tmp_path, monkeypatch):
    from qsart.training import finetune
    torch.set_num_threads(1)
    manifest, frame = source
    selected = frame.loc[frame.split != 'calibration']
    cache, _ = prepare_input_cache(selected, manifest, tmp_path/'cache', image_size=32)
    torch.manual_seed(23)
    initial = TinyModel().state_dict()
    config = FinetuneConfig(image_size=32, epochs=4, patience=4, batch_size=5, cpu_threads=1)
    baseline = TinyModel(); baseline.load_state_dict(initial)
    fit_finetune(baseline, cache, selected, tmp_path/'complete', config, base_metadata=fixture_metadata(manifest))
    interrupted = TinyModel(); interrupted.load_state_dict(initial)
    real_evaluate, calls = finetune._evaluate, []

    def failing_evaluate(*args, **kwargs):
        calls.append(True)
        if len(calls) == 3:
            raise RuntimeError('generated interruption after completed epoch two')
        return real_evaluate(*args, **kwargs)

    monkeypatch.setattr(finetune, '_evaluate', failing_evaluate)
    with pytest.raises(RuntimeError, match='generated interruption'):
        fit_finetune(interrupted, cache, selected, tmp_path/'partial', config, base_metadata=fixture_metadata(manifest))
    monkeypatch.setattr(finetune, '_evaluate', real_evaluate)
    resumed = TinyModel(); resumed.load_state_dict(initial)
    torch.rand(51)  # Work done after interruption must not alter resumed draws.
    fit_finetune(resumed, cache, selected, tmp_path/'partial', config,
                 base_metadata=fixture_metadata(manifest), resume_last=True)
    complete = torch.load(tmp_path/'complete/last.pt', weights_only=True)
    partial = torch.load(tmp_path/'partial/last.pt', weights_only=True)
    assert complete['history'] == partial['history']
    assert complete['scheduler_state'] == partial['scheduler_state']
    assert torch.equal(complete['torch_rng_state'], partial['torch_rng_state'])
    assert torch.equal(complete['loader_rng_state'], partial['loader_rng_state'])
    assert all(torch.equal(value, partial['model_state'][name]) for name, value in complete['model_state'].items())
    changed = TinyModel(); changed.load_state_dict(initial)
    with pytest.raises(ValueError, match='configuration differs'):
        fit_finetune(changed, cache, selected, tmp_path/'partial', replace(config, backbone_lr=.002),
                     base_metadata=fixture_metadata(manifest), resume_last=True)


def test_resume_rejects_improved_last_checkpoint_with_stale_selected_sibling(source, tmp_path):
    """Generated metadata recreates interruption between last and best replacement."""
    torch.set_num_threads(1)
    manifest, frame = source
    selected = frame.loc[frame.split != 'calibration']
    cache, _ = prepare_input_cache(selected, manifest, tmp_path/'cache', image_size=32)
    torch.manual_seed(31)
    initial = TinyModel().state_dict()
    model = TinyModel(); model.load_state_dict(initial)
    config = FinetuneConfig(image_size=32, epochs=2, patience=2, batch_size=5, cpu_threads=1)
    fit_finetune(model, cache, selected, tmp_path/'run', config, base_metadata=fixture_metadata(manifest))
    last = torch.load(tmp_path/'run/last.pt', weights_only=True)
    last['history'][-1]['validation_macro_f1'] = 1.0
    last['history'][-1]['validation_nll'] = 0.0
    last['best_selection_score'] = [1.0, 0.0]
    torch.save(last, tmp_path/'run/last.pt')
    restarted = TinyModel(); restarted.load_state_dict(initial)
    with pytest.raises(ValueError, match='rank differs'):
        fit_finetune(restarted, cache, selected, tmp_path/'run', config,
                     base_metadata=fixture_metadata(manifest), resume_last=True)
