from hashlib import sha256
from io import BytesIO

from PIL import Image
import pytest

from qsart.paired_view_inputs import AUTHOR_COMMIT, _ordered_pairs, _score_pairs


def fixture_rows(tmp_path, patient, split):
    rows = []
    for view in ('l1', 'l2', 'r1', 'r2'):
        image_id = f'{patient}_{view}'
        image_path = tmp_path / f'{image_id}.jpg'
        encoded = BytesIO()
        with Image.new('RGB', (32, 32), (220 * int(view.endswith('2')), 25, 40)) as image:
            image.save(encoded, format='JPEG')
        payload = encoded.getvalue() + image_id.encode()
        image_path.write_bytes(payload)
        rows.append({'subject_id': f'fixture:{patient}', 'subject_id_original': patient, 'image_id': image_id,
                     'original_image_id': image_id, 'original_patient_id': patient, 'image_path': image_path,
                     'dataset_source': 'DeepDRiD_regular_fundus', 'official_split': 'official_train',
                     'cohort_role': 'development_v1', 'annotation_status': 'verified', 'source_commit': AUTHOR_COMMIT,
                     'split': split, 'eye': 'left' if view.startswith('l') else 'right', 'dr_grade': 3,
                     'sha256': sha256(payload).hexdigest(), 'source_bytes': len(payload), 'width': 32, 'height': 32})
    return rows


def test_only_same_eye_suffix_order_inside_patient(tmp_path):
    train = fixture_rows(tmp_path, 2, 'train')
    validation = fixture_rows(tmp_path, 1, 'validation')
    partitions = {'train': 1, 'validation': 1}
    pairs = _ordered_pairs(list(reversed(train + validation)), partitions)
    assert [(eye, first['image_id'], second['image_id']) for _, eye, first, second in pairs['train']] == [
        ('left', '2_l1', '2_l2'), ('right', '2_r1', '2_r2'),
    ]
    train[2]['split'] = 'validation'
    with pytest.raises(ValueError, match='one partition'):
        _ordered_pairs(train + validation, partitions)


def test_reject_conflicting_labels_and_duplicate_photo(tmp_path):
    rows = fixture_rows(tmp_path, 4, 'train')
    rows[1]['dr_grade'] = 1
    with pytest.raises(ValueError, match='Released grades differ'):
        _ordered_pairs(rows, {'train': 1})
    rows[1]['dr_grade'] = 3
    rows[1]['sha256'] = rows[0]['sha256']
    with pytest.raises(ValueError, match='duplicate'):
        _ordered_pairs(rows, {'train': 1})


def test_decodes_bound_pairs_with_first_view_features_only(tmp_path):
    rows = fixture_rows(tmp_path, 3, 'train')
    pairs = _ordered_pairs(rows, {'train': 1})['train']

    class FixtureEngine:
        def predict_images(self, images, *, second_pass_budget, second_pass_policy):
            assert second_pass_budget == 0 and second_pass_policy == 'none'
            return [{'grade_probabilities': ([.1, .2, .3, .3, .1] if image.getpixel((0, 0))[0] > 100
                                              else [.7, .1, .1, .1, 0]),
                     'quality': {'poor_probability': .9 if image.getpixel((0, 0))[0] > 100 else .25},
                     'model_version': 'fixture', 'data_purpose': 'real_research', 'second_pass_activated': False}
                    for image in images]

    scores = _score_pairs(FixtureEngine(), 'fixture', pairs)
    assert len(scores) == 2
    for score in scores:
        assert score['first_quality'] == .75 and score['first'][0] == .7 and score['second'][0] == .1
        assert 'second_quality' not in score
    rows[1]['image_path'].write_bytes(b'altered')
    with pytest.raises(ValueError, match='differs from the frozen inventory'):
        _score_pairs(FixtureEngine(), 'fixture', pairs)
