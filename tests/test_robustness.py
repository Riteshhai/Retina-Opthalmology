"""Generated software fixtures for canonical-input stress and paired statistics."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
from PIL import Image
import pytest
import torch

from qsart.evaluation_robustness import (CONDITIONS, apply_degradation, canonical_rgb,
                                        development_selection, paired_patient_bootstrap,
                                        score_canonical)
from qsart.models.checkpoint import sha256_file


def image_fixture():
    pixels = np.zeros((224, 224, 3), dtype=np.uint8)
    pixels[40:184, 40:184] = [31, 127, 201]
    pixels[110:114, 20:204] = 255
    return Image.fromarray(pixels)


def test_canonical_crop_once_and_conditions_preserve_geometry_and_source():
    source = Image.new('RGB', (400, 300))
    source.paste((100, 50, 25), (100, 50, 300, 250))
    canonical = canonical_rgb(source)
    assert canonical.mode == 'RGB' and canonical.size == (224, 224)
    assert np.array_equal(np.asarray(canonical)[0, 0], [100, 50, 25])
    before = np.asarray(canonical).copy()
    for condition in CONDITIONS:
        changed = apply_degradation(canonical, condition)
        assert changed.mode == 'RGB' and changed.size == (224, 224) and changed is not canonical
        np.testing.assert_array_equal(np.asarray(changed), np.asarray(apply_degradation(canonical, condition)))
    np.testing.assert_array_equal(np.asarray(canonical), before)


def test_exposure_clipping_and_rounding_and_blur_pixel_scale():
    source = Image.new('RGB', (224, 224), (1, 3, 200))
    assert apply_degradation(source, 'exposure_0_5').getpixel((0, 0)) == (0, 2, 100)
    assert apply_degradation(source, 'exposure_1_5').getpixel((0, 0)) == (2, 4, 255)
    impulse = Image.new('RGB', (224, 224)); impulse.putpixel((112, 112), (255, 255, 255))
    clean_peak = np.asarray(impulse)[112, 112, 0]
    blurred_1 = np.asarray(apply_degradation(impulse, 'blur_1'))
    blurred_2 = np.asarray(apply_degradation(impulse, 'blur_2'))
    assert 0 < blurred_2[112, 112, 0] < blurred_1[112, 112, 0] < clean_peak
    assert blurred_2[112, 115, 0] > blurred_1[112, 115, 0]


def test_jpeg_roundtrip_is_a_real_condition_not_identity():
    image = image_fixture()
    assert not np.array_equal(np.asarray(apply_degradation(image, 'jpeg_20')), np.asarray(image))
    assert not np.array_equal(np.asarray(apply_degradation(image, 'jpeg_20')),
                              np.asarray(apply_degradation(image, 'jpeg_50')))


@pytest.mark.parametrize('image', [None, Image.new('L', (224, 224)), Image.new('RGB', (225, 224))])
def test_reject_noncanonical_degradation_inputs(image):
    with pytest.raises(ValueError, match='Canonical'):
        apply_degradation(image, 'clean')


def test_reject_unknown_transform_and_nonprotocol_size():
    with pytest.raises(ValueError, match='Unknown'):
        apply_degradation(image_fixture(), 'blur_999')
    with pytest.raises(ValueError, match='224px'):
        canonical_rgb(image_fixture(), 256)
    with pytest.raises(ValueError, match='PIL'):
        canonical_rgb(None)


class RecordingModel(torch.nn.Module):
    def __init__(self, heads):
        super().__init__(); self.heads = heads; self.received = []

    def forward(self, tensor):
        self.received.append(tensor.clone())
        return {head: torch.arange(classes, dtype=torch.float32).unsqueeze(0)
                for head, classes in self.heads.items()}


def software_engine():
    metadata = {'image_size': 224, 'normalization_mean': [0., 0., 0.],
                'normalization_std': [1., 1., 1.], 'data_purpose': 'software_test',
                'poor_class_indices': [0]}
    engine = SimpleNamespace(dr_metadata=metadata.copy(), quality_metadata=metadata.copy(),
        dr_model=RecordingModel({'dr_grade': 5}).eval(),
        quality_model=RecordingModel({'quality_grade': 2, 'artifact': 3}).eval(), model_version='software_test')
    engine._probabilities = lambda logits, task: {head: (values / 2.).softmax(1)[0]
                                                  for head, values in logits.items()}
    return engine


def test_scoring_retains_black_border_without_reprocessing_and_preserves_raw_scores(monkeypatch):
    engine = software_engine(); image = image_fixture()
    def forbidden_preprocessing(*args, **kwargs):
        raise AssertionError('Canonical image must not be preprocessed again')
    monkeypatch.setattr('qsart.evaluation_robustness.preprocess_image', forbidden_preprocessing)
    rows = score_canonical(engine, [image])
    assert len(rows) == 1
    tensor = engine.dr_model.received[0]
    assert tensor.shape == (1, 3, 224, 224)
    assert tensor[0, :, 0, 0].sum() == 0
    np.testing.assert_allclose(tensor[0, :, 50, 50].numpy(), np.array([31, 127, 201])/255.)
    row = rows[0]
    assert row['second_pass_activated'] is False
    assert row['input_scope'] == 'canonical_224px_model_input_stress'
    assert row['predicted_grade'] == 4 and row['quality']['predicted_class'] == 1
    assert len(row['quality']['defects']['artifact']) == 3
    assert row['uncertainty']['entropy'] > 0
    assert row['grade_probabilities'] != row['uncalibrated']['grade_probabilities']
    np.testing.assert_allclose(row['grade_probabilities'], (torch.arange(5)/2).softmax(0).numpy())


@pytest.mark.parametrize('change', ['size', 'normalization', 'training', 'input'])
def test_scoring_rejects_invalid_model_or_input_configuration(change):
    engine = software_engine(); images = [image_fixture()]
    if change == 'size': engine.quality_metadata['image_size'] = 256
    if change == 'normalization': engine.dr_metadata['normalization_std'] = [1., 0., 1.]
    if change == 'training': engine.dr_model.train()
    if change == 'input': images = [Image.new('RGB', (225, 224))]
    with pytest.raises(ValueError): score_canonical(engine, images)


def test_paired_bootstrap_is_zero_for_identical_conditions_and_constant_for_constant_delta():
    before = [0., 1., 0., 1.]; groups = ['a', 'a', 'b', 'b']
    same = paired_patient_bootstrap(before, before, groups)
    assert same['mean_delta'] == same['lower'] == same['upper'] == 0.
    changed = paired_patient_bootstrap(before, np.array(before)+.25, groups)
    assert changed['mean_delta'] == changed['lower'] == changed['upper'] == .25
    assert changed['n_patients'] == 2 and changed['n_images'] == 4
    assert changed['repeats'] == 1000 and changed['seed'] == 20261022


def test_cluster_bootstrap_keeps_all_views_and_paired_conditions_together():
    # Two two-view patients have deltas 0 and 1. Resampling two whole patients
    # produces only 0, .5 or 1; it cannot invent within-patient independence.
    result = paired_patient_bootstrap([.2]*4, [.2, .2, 1.2, 1.2], ['a', 'a', 'b', 'b'])
    assert result['mean_delta'] == .5 and result['lower'] == 0. and result['upper'] == 1.
    assert result == paired_patient_bootstrap([.2]*4, [.2, .2, 1.2, 1.2], ['a', 'a', 'b', 'b'])
    singleton = paired_patient_bootstrap([0., 1.], [1., 1.], ['a', 'a'])
    assert singleton['mean_delta'] == .5 and singleton['lower'] is None and singleton['upper'] is None


@pytest.mark.parametrize('before,after,groups,kwargs', [
    ([], [], [], {}), ([0], [1, 2], ['a'], {}), ([0], [float('nan')], ['a'], {}),
    ([0], [1], [None], {}), ([0], [1], ['a'], {'repeats': 0}),
    ([0], [1], ['a'], {'seed': -1}),
])
def test_paired_bootstrap_rejects_invalid_pairs(before, after, groups, kwargs):
    with pytest.raises(ValueError): paired_patient_bootstrap(before, after, groups, **kwargs)


def selection_fixture(tmp_path):
    image = tmp_path/'software.png'; image_fixture().save(image)
    return pd.DataFrame([{'split': 'validation', 'subject_id': 'software-patient',
        'image_path': str(image), 'sha256': sha256_file(image), 'dr_grade': 2, 'quality_grade': 1}])


def test_development_selection_verifies_source_bytes_and_refuses_holdout(tmp_path):
    frame = selection_fixture(tmp_path)
    selected = development_selection(frame)
    assert len(selected) == 1 and selected.iloc[0].dr_grade == 2
    frame.loc[0, 'split'] = 'confirmation'
    with pytest.raises(ValueError, match='development partitions'): development_selection(frame)
    frame.loc[0, 'split'] = 'validation'; frame.loc[0, 'sha256'] = '0'*64
    with pytest.raises(ValueError, match='SHA256'): development_selection(frame)


def test_development_selection_refuses_overlap_in_patient_groups(tmp_path):
    frame = selection_fixture(tmp_path)
    overlap = frame.copy(); overlap.loc[0, 'split'] = 'train'
    with pytest.raises(ValueError, match='Cross-split'): development_selection(pd.concat([frame, overlap]))
