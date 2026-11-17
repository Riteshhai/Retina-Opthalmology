"""Synthetic software fixtures for the locked Set-C aggregation rules."""

import pytest

from qsart.set_c_metrics import paired_comparison, summarize_actions, summarize_predictions
from qsart.triage_v2 import Action


AUTO = Action.AUTO_NONREFERABLE
RECAPTURE = Action.RECAPTURE
REVIEW = Action.SPECIALIST_REVIEW


def test_rollup_separates_an_automatic_view_from_full_patient_clearance():
    grades = [1, 1, 4, 4, 0, 0, 3, 3, 0, 0, 3, 3, 1, 1, 2, 2]
    groups = [patient for patient in ('a', 'b', 'c', 'd') for _ in range(4)]
    actions = [AUTO, AUTO, RECAPTURE, REVIEW, AUTO, AUTO, AUTO, AUTO,
               AUTO, AUTO, RECAPTURE, RECAPTURE, AUTO, AUTO, AUTO, REVIEW]
    metrics = summarize_actions(grades, groups, actions)
    assert metrics['image_actions'] == {AUTO.value: 11, RECAPTURE.value: 3, REVIEW.value: 2}
    assert metrics['patient_actions'] == {AUTO.value: 1, RECAPTURE.value: 1, REVIEW.value: 2}
    assert metrics['missed_referable_images'] == 3
    assert metrics['missed_severe_images'] == 2
    assert metrics['severe_recapture_patients'] == 1
    assert metrics['patients_with_any_automatic_image'] == 4
    assert metrics['patients_with_partial_automatic_images'] == 3
    assert metrics['referable_patients_with_any_automatic_image'] == 4
    assert metrics['patients_with_automatic_referable_image'] == 2
    assert metrics['fully_automatic_referable_patients'] == metrics['fully_automatic_severe_patients'] == 1
    assert metrics['full_patient_clearance_miss_fraction'] == 1
    assert metrics['patient_weighted_missed_referral_per_patient'] == 1.5
    assert metrics['patient_proxy_per_patient'] == pytest.approx(3.175)


def test_zero_full_clearance_has_no_conditional_miss_fraction():
    metrics = summarize_actions([0, 0, 4, 4], ['a'] * 4, [AUTO, AUTO, REVIEW, REVIEW])
    assert metrics['full_patient_clearance_miss_fraction'] is None
    assert metrics['fully_automatic_patient_fraction'] == 0


def test_paired_intervals_require_both_image_and_patient_workload_matches():
    grades = [0, 0, 1, 1, 0, 0, 3, 3]
    groups = ['a'] * 4 + ['b'] * 4
    reference = [REVIEW, REVIEW, AUTO, AUTO] + [AUTO] * 4
    image_only = [REVIEW, AUTO, AUTO, AUTO] * 2
    result = paired_comparison(grades, groups, image_only, reference)
    assert result == {'matched_image_workload': True, 'matched_patient_workload': False,
                      'identical_image_actions': False, 'intervals': None}
    patient_only = [REVIEW] * 4 + [AUTO] * 4
    result = paired_comparison(grades, groups, patient_only, reference)
    assert result == {'matched_image_workload': False, 'matched_patient_workload': True,
                      'identical_image_actions': False, 'intervals': None}


def test_paired_intervals_use_candidate_minus_reference_and_fixed_patient_actions():
    grades = [0, 0, 1, 1, 0, 0, 3, 3]
    groups = ['a'] * 4 + ['b'] * 4
    reference = [AUTO] * 4 + [REVIEW] * 4
    candidate = [REVIEW] * 4 + [AUTO] * 4
    result = paired_comparison(grades, groups, candidate, reference)
    assert result['matched_image_workload'] and result['matched_patient_workload']
    assert not result['identical_image_actions']
    intervals = result['intervals']
    assert intervals['patient_weighted_missed_referral_per_patient'] == {
        'difference_candidate_minus_reference': 3.0, 'ci95': [0.0, 6.0],
    }
    assert intervals['patient_proxy_per_patient'] == {
        'difference_candidate_minus_reference': 3.0, 'ci95': [.25, 5.75],
    }
    assert intervals['fully_automatic_patient_fraction'] == {
        'difference_candidate_minus_reference': 0.0, 'ci95': [-1.0, 1.0],
    }
    identical = paired_comparison(grades, groups, candidate, candidate)
    assert identical['identical_image_actions']
    assert all(value == {'difference_candidate_minus_reference': 0.0, 'ci95': [0.0, 0.0]}
               for value in identical['intervals'].values())


def prediction_fixture():
    grades = [0, 0, 1, 1, 2, 2, 3, 3]
    return [
        {'id': f'fixture-image-{index}', 'patient': f'fixture-patient-{index // 4}',
         'eye': 'lr'[(index % 4) // 2], 'view': index % 2 + 1, 'grade': grade,
         'quality_label': index % 2, 'sha256': f'{index:064x}',
         'p': [.8 if grade == value else .05 for value in range(5)],
         'raw': [.6 if grade == value else .1 for value in range(5)],
         'good_quality': [.2, .5, .3, .8][index % 4], 'entropy': .4,
         'ordinal': [0, 1, 2, 3, 4], 'lac': [grade] if index < 7 else []}
        for index, grade in enumerate(grades)
    ]


def test_prediction_metrics_keep_grade_quality_and_all_view_coverage_separate():
    result = summarize_predictions(prediction_fixture())
    assert result['image_grading']['accuracy'] == 1.0
    assert result['patient_grading']['n'] == 2
    assert result['patient_grading']['worst_grade_accuracy'] == 1.0
    assert result['patient_grading']['referable']['confusion_matrix'] == [[1, 0], [0, 1]]
    assert result['calibration']['production']['nll'] < result['calibration']['raw']['nll']
    assert sum(row['count'] for row in result['calibration']['production']['reliability_bins']) == 8
    assert result['quality']['good_probability_threshold'] == .50
    assert result['quality']['confusion_matrix'] == [[4, 0], [0, 4]]
    assert result['conformal']['ordinal']['covered_images'] == 8
    assert result['conformal']['ordinal']['covered_patients'] == 2
    assert result['conformal']['lac']['covered_images'] == 7
    assert result['conformal']['lac']['covered_patients'] == 1
    assert result['conformal']['lac']['empty_set_count'] == 1


def test_rejects_incomplete_groups_invalid_probabilities_and_disagreeing_eye_labels():
    with pytest.raises(ValueError, match='exactly four views'):
        summarize_actions([0, 0], ['a'] * 2, [AUTO, REVIEW])
    images = prediction_fixture()
    images[0]['p'] = [0.25] * 5
    with pytest.raises(ValueError, match='must sum to one'):
        summarize_predictions(images)
    images = prediction_fixture()
    images[0]['grade'] = 1
    with pytest.raises(ValueError, match='agree within each eye'):
        summarize_predictions(images)
