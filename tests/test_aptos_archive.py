"""Generated archives/fingerprints exercise import integrity, never retinal evidence."""
from importlib.util import module_from_spec, spec_from_file_location
from argparse import Namespace
from io import BytesIO
from pathlib import Path
import stat
from zipfile import BadZipFile, ZipFile, ZipInfo

import numpy as np
import pandas as pd
from PIL import Image
import pytest

from qsart.data.aptos import assign_aptos_splits, fingerprint_components

spec = spec_from_file_location('prepare_aptos_archive', Path(__file__).resolve().parents[1]/
                              'scripts/prepare_aptos_archive.py')
archive_module = module_from_spec(spec)
spec.loader.exec_module(archive_module)


def make_archive(path, *, extra=None, labels=b'id_code,diagnosis\nfixture_a,0\n'):
    pixels = BytesIO()
    Image.new('RGB', (32, 32), 'green').save(pixels, format='PNG')
    with ZipFile(path, 'w') as zipped:
        zipped.writestr('train.csv', labels)
        zipped.writestr('train_images/fixture_a.png', pixels.getvalue())
        zipped.writestr('test_images/untouched_test.png', b'not decoded or extracted')
        zipped.writestr('test.csv', 'id_code\nuntouched_test\n')
        if extra:
            zipped.writestr(*extra)
    return pixels.getvalue()


def test_safe_extract_uses_only_labeled_training_and_refuses_overwrite(tmp_path):
    archive = tmp_path/'fixture.zip'
    original = make_archive(archive)
    output = tmp_path/'raw'
    receipt = tmp_path/'private/receipt.json'
    result = archive_module.extract_labeled_training(archive, output, receipt=receipt)
    assert result['training_images'] == 1
    assert result['crc_verified_for_all_extracted_members'] is True
    assert result['competition_test_extracted'] is False
    assert (output/'train_images/fixture_a.png').read_bytes() == original
    assert not (output/'test.csv').exists() and not (output/'test_images').exists()
    assert len(result['extracted_sha256']) == 2
    assert receipt.is_file()
    with pytest.raises(ValueError, match='already exist'):
        archive_module.extract_labeled_training(archive, output, receipt=receipt)


@pytest.mark.parametrize('member', ['../escape.txt', '/absolute.txt', 'C:/outside.txt',
                                  'train_images/../escape.png', 'train_images\\escape.png',
                                  './train.csv', 'train_images//escape.png'])
def test_unsafe_member_rejected_before_creating_output(tmp_path, member):
    archive = tmp_path/'bad.zip'
    make_archive(archive, extra=(member, b'bad'))
    output = tmp_path/'raw'
    # ZipInfo normalizes backslashes on Windows before serialization; either a
    # raw unsafe name or the resulting unreferenced training member is rejected.
    with pytest.raises(ValueError, match='Unsafe|exactly match'):
        archive_module.extract_labeled_training(archive, output, receipt=tmp_path/'receipt.json')
    assert not output.exists()


def test_casefold_duplicates_symlinks_and_extra_training_members_rejected(tmp_path):
    archive = tmp_path/'duplicate.zip'
    make_archive(archive, extra=('TRAIN.CSV', b'duplicate'))
    with pytest.raises(ValueError, match='Duplicate case'):
        archive_module.archive_plan(archive)
    link = ZipInfo('symbolic_link')
    link.create_system = 3
    link.external_attr = (stat.S_IFLNK | 0o777) << 16
    make_archive(archive, extra=(link, 'outside'))
    with pytest.raises(ValueError, match='symlinks'):
        archive_module.archive_plan(archive)
    make_archive(archive, extra=('train_images/not_in_csv.png', b'extra'))
    with pytest.raises(ValueError, match='exactly match'):
        archive_module.archive_plan(archive)


def test_missing_labeled_member_and_invalid_labels_rejected(tmp_path):
    archive = tmp_path/'missing.zip'
    make_archive(archive, labels=b'id_code,diagnosis\nfixture_b,0\n')
    with pytest.raises(ValueError, match='exactly match'):
        archive_module.archive_plan(archive)
    make_archive(archive, labels=b'id_code,diagnosis\nfixture_a,1.5\n')
    with pytest.raises(ValueError, match='integer grades'):
        archive_module.archive_plan(archive)


def test_private_outputs_cannot_be_placed_in_public_reports_tree(tmp_path):
    project = Path(__file__).resolve().parents[1]
    for field in ('raw_root', 'output', 'private'):
        values = {'raw_root': project/'data/raw/fixture', 'output': project/'data/manifests/fixture',
                  'private': project/'artifacts/fixture', 'audit': project/'reports/fixture.json'}
        values[field] = project/'reports/private_data_would_leak'
        with pytest.raises(ValueError, match='ignored project'):
            archive_module.prepare(Namespace(**values))


def test_bad_member_crc_never_produces_success_receipt(tmp_path):
    archive = tmp_path/'bad_crc.zip'
    original = make_archive(archive)
    corrupted = bytearray(archive.read_bytes())
    position = corrupted.find(original)
    assert position >= 0  # This fixture is ZIP_STORED, so the PNG is present verbatim.
    corrupted[position + len(original)//2] ^= 1
    archive.write_bytes(corrupted)
    receipt = tmp_path/'receipt.json'
    with pytest.raises(BadZipFile, match='CRC'):
        archive_module.extract_labeled_training(archive, tmp_path/'raw', receipt=receipt)
    assert not receipt.exists()
    # Failed partial extraction is preserved, never silently accepted or erased.
    assert (tmp_path/'raw/train.csv').exists()


def test_encrypted_archive_flag_is_rejected_before_extraction(tmp_path):
    archive = tmp_path/'encrypted_flag.zip'
    make_archive(archive)
    payload = bytearray(archive.read_bytes())
    central = payload.find(b'PK\x01\x02')
    assert central >= 0
    payload[central + 8] |= 1
    archive.write_bytes(payload)
    with pytest.raises(ValueError, match='Encrypted'):
        archive_module.extract_labeled_training(archive, tmp_path/'raw', receipt=tmp_path/'receipt.json')
    assert not (tmp_path/'raw').exists()


def fingerprints():
    rows = []
    rng = np.random.default_rng(731)
    for index in range(30):
        rows.append({'image_id': f'fixture{index:03}', 'dr_grade': index % 5,
                     'sha256': f'{index:064x}', 'phash': f'{int(rng.integers(1, 2**62)):016x}'})
    # A-B-C forms a component; A-C are not directly within the one-bit threshold.
    rows.extend([
        {'image_id': 'conflict_a', 'dr_grade': 0, 'sha256': 'a'*64, 'phash': '0000000000000000'},
        {'image_id': 'conflict_b', 'dr_grade': 0, 'sha256': 'b'*64, 'phash': '0000000000000001'},
        {'image_id': 'conflict_c', 'dr_grade': 3, 'sha256': 'c'*64, 'phash': '0000000000000003'},
    ])
    return pd.DataFrame(rows)


def test_explicit_quarantine_excludes_entire_transitive_component_without_relabeling():
    original = fingerprints()
    with pytest.raises(ValueError, match='Conflicting DR labels'):
        assign_aptos_splits(original, near_duplicate_distance=1)
    result, duplicates, report = assign_aptos_splits(original, near_duplicate_distance=1,
                                                    conflict_policy='quarantine')
    assert not result.image_id.str.startswith('conflict').any()
    assert report['images'] == 30 and report['original_images'] == 33
    assert report['quarantined_images'] == 3 and report['conflicting_components'] == 1
    assert report['quarantined_dr_grade_counts'] == {'0': 2, '3': 1}
    assert report['quarantined_component_size_counts'] == {'3': 1}
    assert len(duplicates) == 2
    assert result.groupby('split_group_id').split.nunique().eq(1).all()
    expected = original.set_index('image_id').dr_grade.loc[result.image_id]
    assert result.dr_grade.tolist() == expected.tolist()
    reordered, _, reordered_report = assign_aptos_splits(original.sample(frac=1, random_state=9),
        near_duplicate_distance=1, conflict_policy='quarantine')
    assert result.set_index('image_id').split.to_dict() == reordered.set_index('image_id').split.to_dict()
    assert reordered_report['quarantined_dr_grade_counts'] == report['quarantined_dr_grade_counts']


def test_quarantine_not_silent_and_fingerprint_component_audit_is_label_preserving():
    frame = fingerprints()
    audited, _ = fingerprint_components(frame, near_duplicate_distance=1)
    assert audited.component_label_conflict.sum() == 3
    assert audited.dr_grade.tolist() == frame.dr_grade.tolist()
    assert audited.loc[audited.component_label_conflict, 'split_group_id'].nunique() == 1
    with pytest.raises(ValueError, match='Conflict policy'):
        assign_aptos_splits(frame, conflict_policy='vote')
    with pytest.raises(ValueError, match='No label-consistent'):
        assign_aptos_splits(frame.tail(3), near_duplicate_distance=1, conflict_policy='quarantine')


def test_calibration_groups_and_representatives_are_stable_and_complete():
    frame = fingerprints().head(30)
    frame, _, _ = assign_aptos_splits(frame)
    frame, roles = archive_module.assign_calibration_roles(frame)
    assert frame.groupby('split_group_id').is_group_representative.sum().eq(1).all()
    assert roles.split.eq('calibration').all()
    assert roles.groupby('split_group_id').calibration_role.nunique().eq(1).all()
    assert set(roles.calibration_role) == {'temperature', 'conformal'}
    assert roles.loc[roles.calibration_role == 'temperature', 'split_group_id'].nunique() == len(
        roles.split_group_id.unique()) // 2
    shuffled, shuffled_roles = archive_module.assign_calibration_roles(frame.sample(frac=1, random_state=3))
    assert frame.set_index('image_id').is_group_representative.to_dict() == shuffled.set_index(
        'image_id').is_group_representative.to_dict()
    assert roles.set_index('image_id').calibration_role.to_dict() == shuffled_roles.set_index(
        'image_id').calibration_role.to_dict()
