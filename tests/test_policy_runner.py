"""Fixtures check cohort accounting; generated scores are not retinal evidence."""
import numpy as np
import pytest

from scripts.run_policy_comparison import outcome_vectors, pooled_metrics, public_fit


def test_pooled_capacity_is_sum_of_fold_floors_not_floor_of_pooled_count():
    rows = [{'grade_probabilities': [1., 0., 0., 0., 0.]}]*9
    actions = ['SPECIALIST_REVIEW']*3+['AUTO_NONREFERABLE']*6
    result = pooled_metrics(np.zeros(9, dtype=int), rows, actions, .5, .4, [3, 3, 3])
    assert result['review_capacity'] == 3  # floor(.5*9) would incorrectly grant4.
    assert result['recapture_capacity'] == 3
    assert not result['review_budget_violation']
    with pytest.raises(ValueError, match='summed fold capacities'):
        pooled_metrics(np.zeros(9, dtype=int), rows, ['SPECIALIST_REVIEW']*4+['AUTO_NONREFERABLE']*5,
                       .5, .4, [3, 3, 3])


def test_per_image_proxy_keeps_fixed_prices_and_severe_recapture_burden():
    result = outcome_vectors(np.array([4, 3, 0]),
                             ['AUTO_NONREFERABLE', 'RECAPTURE', 'SPECIALIST_REVIEW'])
    np.testing.assert_allclose(result['proxy_cost'], [10., 6.2, .25])
    np.testing.assert_array_equal(result['referable_auto_miss_indicator'], [1., 0., 0.])


def test_public_fit_omits_individual_candidate_actions():
    fields = ('proxy_total_cost_per_image', 'missed_severe_auto', 'missed_referable_auto',
              'severe_deferred_to_recapture', 'review_count', 'auto_nonreferable_count',
              'recapture_count', 'proxy_review_cost', 'proxy_recapture_cost')
    result = public_fit({'status': 'tuned', 'actions': ['AUTO_NONREFERABLE'],
                         'candidates': [{'actions': ['AUTO_NONREFERABLE'], 'metrics': dict.fromkeys(fields, 0)}]})
    assert 'actions' not in result
    assert 'actions' not in result['candidates'][0]
