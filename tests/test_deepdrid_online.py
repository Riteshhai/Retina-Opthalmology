"""Generated fixtures for the official DeepDRiD online readiness checks."""

from dataclasses import dataclass

import pytest

from scripts.prepare_deepdrid_online import join_online, read_workbook, screen_overlap


@dataclass(frozen=True)
class PhotoFixture:
    patient: str
    eye: str
    view: int
    digest: int
    fingerprint: int

    def record(self):
        image_id = f'{self.patient}_{self.eye}{self.view}'
        return {
            'image_id': image_id, 'patient_id': self.patient, 'eye': self.eye, 'view': self.view,
            'path': f'Images/{self.patient}/{image_id}.jpg',
            'sha256': f'{self.digest:064x}', 'phash': f'{self.fingerprint:016x}', 'width': 32, 'height': 24,
        }


def write_workbook(path, rows):
    workbook = pytest.importorskip('openpyxl').Workbook()
    for row in rows:
        workbook.active.append(row)
    workbook.save(path)
    workbook.close()


@pytest.mark.parametrize('invalid_row', [('1_l1', 0), ('1_l2', 5), ('1_l2', '2'), ('1_l2', '=1+1')])
def test_grade_workbook_rejects_duplicate_ids_and_noninteger_source_grades(tmp_path, invalid_row):
    path = tmp_path / 'grades.xlsx'
    columns = ('image_id', 'DR_Levels')
    write_workbook(path, [columns, ('1_l1', 0), invalid_row])

    with pytest.raises(ValueError):
        read_workbook(path, columns, (range(5),))


@pytest.fixture
def online_labels():
    online = [PhotoFixture('1', eye, view, index, index).record()
              for index, (eye, view) in enumerate((('l', 1), ('l', 2), ('r', 1), ('r', 2)), 1)]
    grading = {'1_l1': (1,), '1_l2': (1,), '1_r1': (4,), '1_r2': (4,)}
    quality = {'1_l1': (0, 0, 6, 8), '1_l2': (1, 1, 10, 10),
               '1_r1': (1, 6, 8, 8), '1_r2': (0, 4, 1, 1)}
    return online, grading, quality


def test_join_keeps_eye_grade_and_image_quality_in_photo_order(online_labels):
    online, grading, quality = online_labels
    online.reverse()
    join_online(online, grading, quality, expected_patients=1)

    assert [(row['image_id'], row['dr_grade']) for row in online] == [
        ('1_r2', 4), ('1_r1', 4), ('1_l2', 1), ('1_l1', 1),
    ]
    fields = ('overall_quality', 'artifact', 'clarity', 'field_definition')
    assert [tuple(row[field] for field in fields) for row in online] == [
        (0, 4, 1, 1), (1, 6, 8, 8), (1, 1, 10, 10), (0, 0, 6, 8),
    ]


@pytest.mark.parametrize('invalid', ['missing_grading', 'extra_quality', 'disagreeing_views', 'duplicate_slot'])
def test_join_rejects_inexact_cohorts_and_ambiguous_eye_labels(online_labels, invalid):
    online, grading, quality = online_labels
    if invalid == 'missing_grading':
        del grading['1_l2']
    elif invalid == 'extra_quality':
        quality['2_l1'] = (1, 1, 10, 10)
    elif invalid == 'disagreeing_views':
        grading['1_l2'] = (3,)
    else:
        online[1]['view'] = 1

    with pytest.raises(ValueError):
        join_online(online, grading, quality, expected_patients=1)


def test_overlap_excludes_whole_patients_and_preserves_within_patient_near_views():
    online = [fixture.record() for fixture in (
        PhotoFixture('1', 'l', 1, 1, 0x0000000000000000),
        PhotoFixture('1', 'l', 2, 2, 0x0000000000000003),
        PhotoFixture('1', 'r', 1, 3, 0x0000000000000005),
        PhotoFixture('2', 'l', 1, 4, 0x0F0F0F0F0F0F0F0F),
        PhotoFixture('2', 'l', 2, 5, 0xF0F0F0F0F0F0F0F0),
        PhotoFixture('3', 'l', 1, 6, 0x0F0F0F0F0F0F0F0C),
        PhotoFixture('4', 'l', 1, 7, 0x3333333333333333),
        PhotoFixture('4', 'r', 1, 7, 0x3333333333333333),
        PhotoFixture('5', 'l', 1, 8, 0xCCCCCCCCCCCCCCCC),
        PhotoFixture('6', 'l', 1, 9, 0xAAAAAAAAAAAAAAAA),
        PhotoFixture('6', 'l', 2, 10, 0xCCCC3333CCCC3333),
        PhotoFixture('7', 'l', 1, 11, 0x5555555555555555),
    )]
    reference = [
        {'patient_id': '5', 'path': 'training/5_l1.jpg', 'split': 'training',
         'sha256': f'{100:064x}', 'phash': '2222222222222222'},
        {'patient_id': '80', 'path': 'validation/80_l1.jpg', 'split': 'validation',
         'sha256': f'{9:064x}', 'phash': '8888888888888888'},
        {'patient_id': '81', 'path': 'validation/81_l1.jpg', 'split': 'validation',
         'sha256': f'{101:064x}', 'phash': '5555555555555556'},
    ]

    internal, reference_pairs, excluded, _ = screen_overlap(online, reference)

    assert excluded == ['2', '3', '4', '5', '6', '7']
    assert [row['eligible'] for row in online] == [True, True, True] + [False] * 9
    assert {frozenset((pair['first'], pair['second'])) for pair in internal} == {
        frozenset(('Images/1/1_l1.jpg', 'Images/1/1_l2.jpg')),
        frozenset(('Images/1/1_l1.jpg', 'Images/1/1_r1.jpg')),
        frozenset(('Images/1/1_l2.jpg', 'Images/1/1_r1.jpg')),
        frozenset(('Images/2/2_l1.jpg', 'Images/3/3_l1.jpg')),
        frozenset(('Images/4/4_l1.jpg', 'Images/4/4_r1.jpg')),
    }
    assert {frozenset((pair['first'], pair['second'])) for pair in reference_pairs} == {
        frozenset(('Images/6/6_l1.jpg', 'validation/80_l1.jpg')),
        frozenset(('Images/7/7_l1.jpg', 'validation/81_l1.jpg')),
    }
