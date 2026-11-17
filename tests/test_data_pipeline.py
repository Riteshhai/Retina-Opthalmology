"""Software fixtures exercise integrity checks, not retinal-model performance."""
import numpy as np
import pandas as pd
from PIL import Image
import pytest

from qsart.data.manifests import (assign_subject_splits, build_deepdrid_manifest,
                                  inspect_image, parse_deepdrid, resolve_image_path,
                                  validate_manifest)
from qsart.preprocessing import load_fundus_image, preprocess_image


def annotation(path, patients, folder):
    rows = []
    for patient in patients:
        for side in ['l', 'r']:
            for view in [1, 2]:
                image = f'{patient}_{side}{view}'
                rows.append({'patient_id': patient, 'image_id': image,
                             'image_path': f'\\{folder}\\{patient}\\{image}.jpg',
                             'Overall quality': view - 1,
                             'left_eye_DR_Level': patient % 5 if side == 'l' else np.nan,
                             'right_eye_DR_Level': (patient + 1) % 5 if side == 'r' else np.nan,
                             'patient_DR_Level': 4, 'Clarity': 8, 'Field definition': 6, 'Artifact': 0})
    pd.DataFrame(rows).to_csv(path, index=False)
    return rows


def fixture_manifest(tmp_path, images=True):
    train, holdout = tmp_path / 'train.csv', tmp_path / 'holdout.csv'
    rows = annotation(train, range(1, 16), 'regular-fundus-training')
    rows += annotation(holdout, [100, 101], 'regular-fundus-validation')
    if images:
        rng = np.random.default_rng(3)
        for row in rows:
            path, _ = resolve_image_path(tmp_path, row['image_path'])
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(rng.integers(0, 256, (32, 40, 3), dtype=np.uint8)).save(path)
    return build_deepdrid_manifest(train, holdout, tmp_path)


def test_real_path_readability_and_eye_label(tmp_path):
    frame = fixture_manifest(tmp_path)
    report = validate_manifest(frame)
    assert report['ready_for_training']
    assert report['images'] == 68
    assert frame[frame.image_id.eq('1_l1')].dr_grade.item() == 1
    assert frame[frame.image_id.eq('1_r1')].dr_grade.item() == 2
    assert frame[frame.official_split.eq('official_validation')].split.eq('holdout').all()
    assert frame.groupby('subject_id').split.nunique().max() == 1
    assert {'train', 'validation', 'calibration', 'holdout'} == set(frame.split)
    assert frame.width.eq(40).all()


def test_missing_images_not_trainable(tmp_path):
    frame = fixture_manifest(tmp_path, images=False)
    assert validate_manifest(frame, strict=False)['unreadable_images'] == len(frame)
    with pytest.raises(ValueError, match='missing, corrupt'):
        validate_manifest(frame)


def test_corrupt_file_rejected(tmp_path):
    image = tmp_path / 'corrupt.jpg'
    image.write_bytes(b'not an image')
    assert inspect_image(image)['image_status'].startswith('corrupt:')
    with pytest.raises(OSError):
        load_fundus_image(image)


def test_annotation_conflicts_quarantined_without_corrected_labels(tmp_path):
    path = tmp_path / 'annotations.csv'
    annotation(path, [56], 'regular-fundus-training')
    raw = pd.read_csv(path)
    raw.loc[1, 'image_id'] = '56_r3'
    raw.loc[1, 'image_path'] = '\\regular-fundus-training\\56\\56_r3.jpg'
    raw.to_csv(path, index=False)
    frame = parse_deepdrid(path, 'official_train')
    conflict = frame[frame.image_id.eq('56_r3')].iloc[0]
    assert conflict.annotation_status == 'quarantined_eye_or_grade_conflict'
    assert pd.isna(conflict.dr_grade) and pd.isna(conflict.eye)
    assert conflict.original_left_eye_DR_Level == 1


@pytest.mark.parametrize('field,value,message', [('Overall quality', 2, 'Overall quality'),
                                               ('Clarity', 9, 'Clarity')])
def test_unknown_quality_code_rejected(tmp_path, field, value, message):
    path = tmp_path / 'annotations.csv'
    annotation(path, [1], 'regular-fundus-training')
    raw = pd.read_csv(path)
    raw.loc[0, field] = value
    raw.to_csv(path, index=False)
    with pytest.raises(ValueError, match=message):
        parse_deepdrid(path, 'official_train')


def test_missing_column_rejected(tmp_path):
    path = tmp_path / 'annotations.csv'
    annotation(path, [1], 'regular-fundus-training')
    pd.read_csv(path).drop(columns=['Artifact']).to_csv(path, index=False)
    with pytest.raises(ValueError, match='missing official columns'):
        parse_deepdrid(path, 'official_train')


def test_source_subject_overlap_rejected(tmp_path):
    train, holdout = tmp_path / 'train.csv', tmp_path / 'holdout.csv'
    annotation(train, [1], 'regular-fundus-training')
    annotation(holdout, [1], 'regular-fundus-validation')
    with pytest.raises(ValueError, match='overlap'):
        build_deepdrid_manifest(train, holdout, tmp_path)


def test_duplicate_and_leakage_checks(tmp_path):
    frame = fixture_manifest(tmp_path)
    frame.loc[1, 'sha256'] = frame.loc[0, 'sha256']
    with pytest.raises(ValueError, match='exact duplicate'):
        validate_manifest(frame)
    frame.loc[1, 'split'] = 'holdout'
    with pytest.raises(ValueError, match='Subject leakage'):
        validate_manifest(frame)


def test_perceptual_cross_split_candidates_block(tmp_path):
    frame = fixture_manifest(tmp_path)
    first = frame.index[frame.split.eq('train')][0]
    second = frame.index[frame.split.eq('holdout')][0]
    frame.loc[second, 'phash'] = frame.loc[first, 'phash']
    with pytest.raises(ValueError, match='perceptual duplicate'):
        validate_manifest(frame)


def test_reproducible_splits_independent_of_row_order():
    frame = pd.DataFrame({'subject_id': [f'p{i}' for i in range(30)], 'dr_grade': [i % 5 for i in range(30)]})
    original = assign_subject_splits(frame)
    shuffled = frame.sample(frac=1, random_state=19)
    pd.testing.assert_series_equal(original, assign_subject_splits(shuffled).sort_index())
    with pytest.raises(ValueError, match='fractions'):
        assign_subject_splits(frame, calibration_fraction=.9, validation_fraction=.9)


def test_unsafe_paths_rejected(tmp_path):
    with pytest.raises(ValueError, match='Unsafe'):
        resolve_image_path(tmp_path, '\\regular-fundus-training\\..\\secret.jpg')


def test_shared_preprocess_crop_pad_and_bytes(tmp_path):
    raw = np.zeros((100, 200, 3), dtype=np.uint8)
    raw[20:80, 40:160] = [200, 80, 20]
    image = Image.fromarray(raw)
    result = preprocess_image(image, image_size=64)
    assert result.size == (64, 64) and result.mode == 'RGB'
    path = tmp_path / 'test.png'
    image.save(path)
    assert np.array_equal(np.asarray(result), np.asarray(load_fundus_image(path.read_bytes(), 64)))
    with pytest.raises(ValueError, match='image_size'):
        preprocess_image(image, 0)
