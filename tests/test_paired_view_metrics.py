"""Synthetic software fixtures for the paired-view evaluation endpoints."""

import numpy as np
import pytest

from qsart.paired_view_metrics import evaluate_policy, paired_interval, summarize_policy


def _scores(winner, confidence=.6):
    return [confidence if grade == winner else (1. - confidence) / 4. for grade in range(5)]


def _eyes():
    return [
        {'patient': 'fixture-a', 'eye': 'left', 'grade': 4, 'first': _scores(1), 'second': _scores(4, .96)},
        {'patient': 'fixture-b', 'eye': 'left', 'grade': 3, 'first': _scores(2), 'second': _scores(3, .96)},
        {'patient': 'fixture-a', 'eye': 'right', 'grade': 0, 'first': _scores(0), 'second': _scores(0, .96)},
        {'patient': 'fixture-b', 'eye': 'right', 'grade': 2, 'first': _scores(2), 'second': _scores(2, .96)},
    ]


def test_fusion_reads_only_acquired_views_and_keeps_severe_miss_at_referral_cutoff():
    rows = _eyes()
    for row in rows[1:]:
        del row['second']
    base = evaluate_policy(rows, np.zeros(len(rows), dtype=bool))
    observed = evaluate_policy(rows, [True, False, False, False])
    assert base['predicted_grade'].tolist() == [1, 2, 0, 2]
    assert observed['predicted_grade'].tolist() == [4, 2, 0, 2]
    assert base['referable_miss'].tolist() == base['severe_miss'].tolist() == [True, False, False, False]
    assert observed['referable_miss'].sum() == observed['severe_miss'].sum() == 0
    assert base['ledger'].tolist() == [10., .25, 0., .25]
    assert observed['ledger'].tolist() == [.45, .25, 0., .25]


def test_summary_keeps_eye_and_patient_counts_and_ledger_separate():
    observed = summarize_policy(_eyes(), [True, False, False, False])
    assert (observed['eyes'], observed['patients'], observed['extra_views']) == (4, 2, 1)
    assert (observed['referable_eyes'], observed['severe_eyes']) == (3, 2)
    assert (observed['referable_patients'], observed['severe_patients']) == (2, 2)
    assert observed['referable_misses'] == observed['severe_misses'] == 0
    assert observed['patient_referable_misses'] == observed['patient_severe_misses'] == 0
    assert observed['patient_weighted_miss_per_patient'] == 0.
    assert observed['grade_accuracy'] == .75
    assert observed['review_proxy_count'] == 3
    assert observed['ledger_per_eye'] == pytest.approx(.2375)
    delta = observed['delta_vs_first']
    assert delta['grade_accuracy'] == .25
    assert delta['referable_misses'] == delta['severe_misses'] == -1
    assert delta['patient_referable_misses'] == delta['patient_severe_misses'] == -1
    assert delta['patient_weighted_miss_per_patient'] == -5.
    assert delta['review_proxy_count'] == 1
    assert delta['ledger_per_eye'] == pytest.approx(-2.3875)

    rows = _eyes()
    rows[2]['first'] = _scores(2)
    first = summarize_policy(rows, [False] * len(rows))
    assert first['referable_misses'] == first['severe_misses'] == 1
    assert first['patient_referable_misses'] == first['patient_severe_misses'] == 0


def test_paired_patient_bootstrap_uses_a_common_exact_acquisition_workload():
    rows = _eyes()
    candidate = [True, False, False, False]
    reference = [False, False, True, False]
    result = paired_interval(rows, candidate, reference)
    assert result['ledger_per_eye']['difference_candidate_minus_reference'] == pytest.approx(-2.4375)
    assert result['ledger_per_eye']['ci95'] == pytest.approx([-4.875, 0.])
    assert result['referable_misses_per_eye'] == result['severe_misses_per_eye'] == {
        'difference_candidate_minus_reference': -.25, 'ci95': [-.5, 0.],
    }
    assert result['grade_accuracy'] == {'difference_candidate_minus_reference': .25, 'ci95': [0., .5]}
    assert paired_interval(rows[::-1], candidate[::-1], reference[::-1]) == result
    with pytest.raises(ValueError, match='matching realized extra-view counts'):
        paired_interval(rows, candidate, [False] * len(rows))


def test_rejects_invalid_grades_masks_probabilities_and_patient_layouts():
    with pytest.raises(ValueError, match='boolean acquisition decision'):
        evaluate_policy(_eyes(), [1, 0, 0, 0])
    rows = _eyes()
    rows[0]['grade'] = True
    with pytest.raises(ValueError, match='integer eye grades'):
        evaluate_policy(rows, [False] * len(rows))
    rows[0]['grade'] = 0
    rows[0]['second'] = [.4] * 5
    with pytest.raises(ValueError, match='probabilities summing to one'):
        evaluate_policy(rows, [True, False, False, False])
    with pytest.raises(ValueError, match='distinct left and right eye'):
        summarize_policy(rows[:-1], [False] * (len(rows) - 1))
    rows[2]['eye'] = 'l'
    with pytest.raises(ValueError, match='distinct left and right eye'):
        summarize_policy(rows, [False] * len(rows))
