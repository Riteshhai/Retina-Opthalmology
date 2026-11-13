"""Generated IDRiD-schema software fixtures, never external retinal evidence."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import pytest

from qsart.data.idrid import (PARTITIONS, audit_idrid, cross_dataset_candidates,
                             parse_idrid_labels, validate_fingerprints)
from qsart.data.manifests import sha256_file
from qsart.training.data import read_manifest

spec = importlib.util.spec_from_file_location('prepare_idrid', Path(__file__).parents[1]/'scripts/prepare_idrid.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


@pytest.fixture
def package(tmp_path):
    root = tmp_path/'idrid_generated_software_fixture'
    (root/'2. Groundtruths').mkdir(parents=True)
    (root/'LICENSE.txt').write_text('Porwal, Pachade and Kokare; Creative Commons Attribution 4.0 International License.')
    (root/'CC-BY-4.0.txt').write_text('Generated software license fixture; not a dataset source.')
    for part, (directory, filename, _) in enumerate(PARTITIONS.values()):
        folder = root/'1. Original Images'/directory
        folder.mkdir(parents=True)
        rows = []
        for index in range(5):
            stem = f'IDRiD_{index+1:03}'
            pixels = np.random.default_rng(part*100+index).integers(0, 255, (40, 40, 3), dtype=np.uint8)
            Image.fromarray(pixels).save(folder/f'{stem}.jpg')
            rows.append({'Image name': stem, 'Retinopathy grade': index,
                         'Risk of macular edema ': index % 3, 'Unnamed: 3': ''})
        pd.DataFrame(rows).to_csv(root/'2. Groundtruths'/filename, index=False)
    return root


def source_csv(root):
    return root/'2. Groundtruths'/PARTITIONS['official_train'][1]


def references():
    return pd.DataFrame([{'dataset_source': 'generated_reference', 'image_id': 'fixture_reference',
                          'sha256': 'a'*64, 'phash': 'ffffffffffffffff'}])


def test_idrid_preserves_dr_dme_original_splits_and_blocks_training(package, tmp_path):
    frame, internal, cross, report = audit_idrid(package, references(), require_release_counts=False)
    assert len(frame) == 10 and frame.image_id.nunique() == 10
    assert frame.original_image_id.nunique() == 5  # Source stems repeat across original partitions.
    assert set(frame.official_split) == set(PARTITIONS) and frame.split.eq('external_holdout').all()
    assert not frame.eligible_for_training.any() and frame.evaluation_status.eq('reserved_not_inferred').all()
    assert frame.dr_grade.tolist() == list(range(5))*2
    assert frame.dme_risk.tolist() == [0, 1, 2, 0, 1]*2
    assert not {'patient_id', 'subject_id', 'eye_id', 'quality_grade'} & set(frame)
    assert internal.empty and cross.empty and report['observations_excluded'] == 0
    assert not report['model_inference_performed'] and not report['external_model_evaluation_completed']
    text = json.dumps(report)
    assert 'IDRiD_001' not in text and str(package) not in text and 'image_path' not in text
    path = tmp_path/'external.csv'
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match='eligible_for_training'):
        read_manifest(path)
    with pytest.raises(ValueError, match='413 training'):
        parse_idrid_labels(package, require_release_counts=True)


@pytest.mark.parametrize(('column', 'value'), [
    ('Retinopathy grade', '-1'), ('Retinopathy grade', '5'), ('Retinopathy grade', '1.5'),
    ('Retinopathy grade', ''), ('Retinopathy grade', '1.0'),
    ('Risk of macular edema ', '3'), ('Risk of macular edema ', 'NaN'),
])
def test_idrid_invalid_source_labels_fail_closed(package, column, value):
    path = source_csv(package)
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    frame.loc[0, column] = value
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match='integer|DR must'):
        parse_idrid_labels(package)


@pytest.mark.parametrize('mutation', ['extra_column', 'populated_unnamed', 'unsafe_stem', 'duplicate_stem'])
def test_idrid_rejects_ambiguous_schema_and_identities(package, mutation):
    path = source_csv(package)
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    if mutation == 'extra_column': frame['invented_patient_id'] = 'not_authoritative'
    elif mutation == 'populated_unnamed': frame.loc[0, 'Unnamed: 3'] = 'unexpected data'
    elif mutation == 'unsafe_stem': frame.loc[0, 'Image name'] = '../outside'
    else: frame.loc[1, 'Image name'] = frame.loc[0, 'Image name']
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match='three grading|safe unmodified|Duplicate'):
        parse_idrid_labels(package)


@pytest.mark.parametrize('mutation', ['missing', 'extra', 'corrupt', 'wrong_format', 'missing_license'])
def test_idrid_inventory_decode_and_license_fail_closed(package, mutation):
    image = package/'1. Original Images'/PARTITIONS['official_train'][0]/'IDRiD_001.jpg'
    if mutation == 'missing': image.unlink()
    elif mutation == 'extra': image.with_name('IDRiD_999.jpg').write_bytes(image.read_bytes())
    elif mutation == 'corrupt': image.write_bytes(b'not an image')
    elif mutation == 'wrong_format': Image.new('RGB', (40, 40), 'red').save(image, format='PNG')
    else: (package/'LICENSE.txt').unlink()
    with pytest.raises(ValueError, match='inventory mismatch|unreadable|JPEG|license files'):
        audit_idrid(package, references(), require_release_counts=False)


def test_fixed_overlap_exact_and_perceptual_matches_and_reference_conflicts():
    external = pd.DataFrame([{'dataset_source': 'IDRiD', 'image_id': 'external_fixture',
                              'sha256': 'a'*64, 'phash': '0000000000000000'}])
    ref = pd.DataFrame([
        {'dataset_source': 'APTOS_2019', 'image_id': 'exact_fixture', 'sha256': 'a'*64, 'phash': 'ffffffffffffffff'},
        {'dataset_source': 'APTOS_2019', 'image_id': 'near_fixture', 'sha256': 'b'*64, 'phash': '0000000000000003'},
        {'dataset_source': 'APTOS_2019', 'image_id': 'outside_fixture', 'sha256': 'c'*64, 'phash': '0000000000000007'},
    ])
    matches = cross_dataset_candidates(external, pd.concat([ref, ref.iloc[[0]]]))
    assert matches.reference_image_id.tolist() == ['exact_fixture', 'near_fixture']
    assert matches.byte_exact.tolist() == [True, False] and matches.phash_distance.tolist() == [64, 2]
    with pytest.raises(ValueError, match='Conflicting fingerprints'):
        validate_fingerprints(pd.concat([ref, ref.iloc[[0]].assign(sha256='d'*64)]))
    with pytest.raises(ValueError, match='Complete valid'):
        validate_fingerprints(ref.assign(phash='bad'))
    with pytest.raises(ValueError, match='Integer pHash'):
        cross_dataset_candidates(external, ref, distance=True)


def test_idrid_internal_conflicts_are_flagged_without_dropping_or_relabeling(package):
    folder = package/'1. Original Images'/PARTITIONS['official_train'][0]
    (folder/'IDRiD_002.jpg').write_bytes((folder/'IDRiD_001.jpg').read_bytes())
    frame, internal, _, report = audit_idrid(package, references(), require_release_counts=False)
    assert len(frame) == 10 and report['observations_excluded'] == 0
    assert len(internal) == 1 and internal.label_conflict.all()
    assert report['internal_label_conflict_components'] == 1
    assert frame.iloc[:2].dr_grade.tolist() == [0, 1] and frame.iloc[:2].split_group_id.nunique() == 1


def test_private_output_guard_and_environment_precedence(tmp_path, monkeypatch):
    root = tmp_path/'project'; root.mkdir()
    private, public = root/'data/manifests/idrid_v1', root/'reports/idrid_data_v1/data_audit.json'
    runner.validate_output_paths(private, public, root)
    with pytest.raises(ValueError, match='ignored manifests'):
        runner.validate_output_paths(root/'public_manifest', public, root)
    private.mkdir(parents=True)
    with pytest.raises(ValueError, match='Preserve'):
        runner.validate_output_paths(private, public, root)
    monkeypatch.delenv('QSART_IDRID_ROOT', raising=False)
    assert runner.default_idrid_root(root) == (root/'../idrid/B. Disease Grading').resolve()
    (root/'.env').write_text('QSART_IDRID_ROOT=../relocated/B. Disease Grading\n')
    assert runner.default_idrid_root(root) == (root/'../relocated/B. Disease Grading').resolve()
    monkeypatch.setenv('QSART_IDRID_ROOT', '../environment_source')
    assert runner.default_idrid_root(root) == (root/'../environment_source').resolve()


def test_reference_bindings_detect_changed_bytes_and_dataset(tmp_path):
    config = {'near_duplicate_distance': 2, 'references': []}
    for index in range(4):
        path = tmp_path/f'reference{index}.csv'
        frame = references().assign(image_id=f'fixture{index}')
        frame.to_csv(path, index=False)
        config['references'].append({'path': path.name, 'dataset': 'generated_reference',
                                     'rows': 1, 'sha256': sha256_file(path)})
    frame, bindings = runner.load_references(config, tmp_path)
    assert len(frame) == 4 and len(bindings) == 4
    config['references'][0]['dataset'] = 'wrong_source'
    with pytest.raises(ValueError, match='dataset identity'):
        runner.load_references(config, tmp_path)
    config['references'][0]['sha256'] = 'b'*64
    with pytest.raises(ValueError, match='manifest hash'):
        runner.load_references(config, tmp_path)
