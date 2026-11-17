"""Generated image fixtures check APTOS plumbing, never retinal performance."""
import json

import numpy as np
import pandas as pd
from PIL import Image
import pytest

from qsart.data.aptos import (PATIENT_LIMITATION, assign_aptos_splits, build_aptos_manifest,
                              parse_aptos_labels, write_aptos_outputs)
from qsart.training.data import read_manifest


@pytest.fixture
def aptos_root(tmp_path):
    root = tmp_path / 'aptos_generated_software_fixture'
    images = root / 'train_images'
    images.mkdir(parents=True)
    rows = []
    for index in range(25):
        image_id = f'fixture{index:05}'
        pixels = np.random.default_rng(index).integers(0, 255, (40, 40, 3), dtype=np.uint8)
        Image.fromarray(pixels).save(images / f'{image_id}.png')
        rows.append({'id_code': image_id, 'diagnosis': index % 5})
    pd.DataFrame(rows).to_csv(root / 'train.csv', index=False)
    return root


def test_aptos_real_schema_missing_metadata_and_frozen_outputs(aptos_root, tmp_path):
    frame, duplicates, report = build_aptos_manifest(aptos_root)
    assert len(frame) == 25 and frame.dr_grade.between(0, 4).all()
    assert not {'subject_id', 'patient_id', 'eye_id', 'quality_grade'} & set(frame)
    assert frame.original_id_code.equals(frame.image_id)
    assert set(frame.split) == {'train', 'validation', 'calibration', 'test'}
    assert report['patient_disjointness_established'] is False
    assert report['patient_grouping_limitation'] == PATIENT_LIMITATION
    assert report['supervision']['quality'] == 'unavailable'
    assert duplicates.empty
    output = tmp_path / 'private_outputs'
    write_aptos_outputs(frame, duplicates, report, output)
    assert len(read_manifest(output / 'dataset_manifest.csv')) == 25
    summary = json.loads((output / 'dataset_summary.json').read_text())
    assert summary['manifest_sha256'] and 'image_path' not in json.dumps(summary)
    assert 'fixture00000' not in json.dumps(summary)
    assert not (output / 'quality_distribution.png').exists()
    assert (output / 'class_distribution.png').is_file()
    with pytest.raises(ValueError, match='already exist'):
        write_aptos_outputs(frame, duplicates, report, output)


def test_aptos_connected_duplicates_stay_together_and_conflicts_block():
    rows = [
        {'image_id': 'fixture_a', 'dr_grade': 0, 'sha256': 'a' * 64, 'phash': '0000000000000000'},
        {'image_id': 'fixture_b', 'dr_grade': 0, 'sha256': 'b' * 64, 'phash': '0000000000000001'},
        {'image_id': 'fixture_c', 'dr_grade': 0, 'sha256': 'c' * 64, 'phash': '0000000000000003'},
        {'image_id': 'fixture_d', 'dr_grade': 1, 'sha256': 'd' * 64, 'phash': '3fffffffffffffff'},
    ]
    frame = pd.DataFrame(rows)
    result, candidates, report = assign_aptos_splits(frame, near_duplicate_distance=1)
    assert result.iloc[:3].split_group_id.nunique() == 1
    assert result.iloc[:3].split.nunique() == 1
    assert len(candidates) == 2 and report['duplicate_components'] == 1
    shuffled, _, _ = assign_aptos_splits(frame.sample(frac=1, random_state=8), near_duplicate_distance=1)
    pd.testing.assert_frame_equal(result.set_index('image_id').sort_index(),
                                  shuffled.set_index('image_id').sort_index())
    frame.loc[2, 'dr_grade'] = 2
    with pytest.raises(ValueError, match='Conflicting DR labels'):
        assign_aptos_splits(frame, near_duplicate_distance=1)


def test_aptos_exact_duplicate_label_conflict_even_distant_fingerprints():
    frame = pd.DataFrame([
        {'image_id': 'fixture_a', 'dr_grade': 0, 'sha256': 'a' * 64, 'phash': '0000000000000000'},
        {'image_id': 'fixture_b', 'dr_grade': 1, 'sha256': 'a' * 64, 'phash': '3fffffffffffffff'},
    ])
    with pytest.raises(ValueError, match='Conflicting DR labels'):
        assign_aptos_splits(frame)


@pytest.mark.parametrize('diagnosis', [-1, 5, 1.5, '', 'unknown', 'True'])
def test_aptos_invalid_labels_rejected(aptos_root, diagnosis):
    labels = pd.read_csv(aptos_root / 'train.csv').astype({'diagnosis': 'object'})
    labels.loc[0, 'diagnosis'] = diagnosis
    labels.to_csv(aptos_root / 'train.csv', index=False)
    with pytest.raises(ValueError, match='diagnosis'):
        parse_aptos_labels(aptos_root)


def test_aptos_unsafe_ids_duplicate_ids_extra_metadata_and_missing_images(aptos_root):
    original = pd.read_csv(aptos_root / 'train.csv')
    invalid = original.copy(); invalid.loc[0, 'id_code'] = '../outside'
    invalid.to_csv(aptos_root / 'train.csv', index=False)
    with pytest.raises(ValueError, match='safe image stems'):
        parse_aptos_labels(aptos_root)
    invalid = original.copy(); invalid.loc[1, 'id_code'] = invalid.loc[0, 'id_code'].upper()
    invalid.to_csv(aptos_root / 'train.csv', index=False)
    with pytest.raises(ValueError, match='Duplicate'):
        parse_aptos_labels(aptos_root)
    original.assign(patient_id='not_authoritative').to_csv(aptos_root / 'train.csv', index=False)
    with pytest.raises(ValueError, match='only'):
        parse_aptos_labels(aptos_root)
    original.to_csv(aptos_root / 'train.csv', index=False)
    (aptos_root / 'train_images' / f'{original.iloc[0].id_code}.png').unlink()
    with pytest.raises(ValueError, match='missing, corrupt'):
        build_aptos_manifest(aptos_root)


def test_aptos_invalid_fraction_and_fingerprint_rejected():
    frame = pd.DataFrame([{'image_id': 'fixture_a', 'dr_grade': 0, 'sha256': 'a' * 64,
                           'phash': '0000000000000000'}])
    with pytest.raises(ValueError, match='sum to one'):
        assign_aptos_splits(frame, fractions={'train': .6, 'validation': .2, 'calibration': .2, 'test': .2})
    with pytest.raises(ValueError, match='distance'):
        assign_aptos_splits(frame, near_duplicate_distance=64)
    frame.loc[0, 'phash'] = None
    with pytest.raises(ValueError, match='fingerprints'):
        assign_aptos_splits(frame)
