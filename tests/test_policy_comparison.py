"""Synthetic score fixtures verify development-only comparator bookkeeping."""
from copy import deepcopy

import numpy as np
import pytest

from qsart.evaluation_policy import (
    BUDGET_PAIRS, DECISION_REVIEW_PRICES, EVALUATION_LEDGER, METHODS, MODES,
    evaluate_policy, paired_patient_delta, patient_folds, prepare_mode_rows, tune_policy,
)
from qsart.triage_v2 import Action, PolicyConfig, allocate_rows, screening_metrics


LOW = [.96, .02, .01, .005, .005]
OPTIONAL = [.70, .22, .04, .025, .015]
HIGH = [.01, .01, .03, .05, .90]


def row(p=LOW, quality=.9, entropy=.1, ordinal=(0, 1), lac=(0,)):
    return {
        'grade_probabilities': list(p), 'quality': {'poor_probability': 1 - quality},
        'uncertainty': {'entropy': entropy}, 'conformal_set': list(ordinal),
        'conformal': {'status': 'available', 'method': 'ordinal_greedy',
                      'prediction_set': list(ordinal), 'baseline_lac_set': list(lac)},
    }


def test_declared_constants_and_patient_folds_are_balanced_grouped_and_order_invariant():
    assert MODES == ('ordinal', 'lac', 'no_cp')
    assert len(BUDGET_PAIRS) == 4 and len(DECISION_REVIEW_PRICES) == 6 and len(METHODS) == 8
    groups = np.repeat([f'fixture-{i}' for i in range(42)], 4)
    folds = patient_folds(groups)
    assert folds.dtype.kind == 'i'
    assert np.bincount(folds).tolist() == [56, 56, 56]
    for group in set(groups):
        assert len(set(folds[groups == group])) == 1
    order = np.random.default_rng(7).permutation(len(groups))
    assert np.array_equal(patient_folds(groups[order]), folds[order])
    for fold in range(3):
        assert set(groups[folds == fold]).isdisjoint(groups[folds != fold])


@pytest.mark.parametrize('groups', [[], [None, 'a'], [' ', 'b'], [np.nan, 'a'],
                                   [True, False], ['1', 1], [['a'], ['b']]])
def test_invalid_patient_groups_are_rejected(groups):
    with pytest.raises(ValueError):
        patient_folds(groups)


@pytest.mark.parametrize('n_folds', [True, 1, 2.5, 4])
def test_invalid_fold_counts_are_rejected(n_folds):
    with pytest.raises(ValueError):
        patient_folds(['a', 'b', 'c'], n_folds)


def test_integer_patient_groups_are_supported():
    assert len(patient_folds([1, 1, 2, 2, 3, 3])) == 6


def test_conformal_modes_are_explicit_copy_inputs_and_remove_both_no_cp_fields():
    original = [row(ordinal=(0, 1, 2), lac=(0,))]
    untouched = deepcopy(original)
    ordinal, lac, no_cp = [prepare_mode_rows(original, mode) for mode in MODES]
    assert ordinal[0]['conformal_set'] == [0, 1, 2]
    assert lac[0]['conformal_set'] == lac[0]['conformal']['prediction_set'] == [0]
    assert lac[0]['conformal']['method'] == 'lac'
    assert 'conformal_set' not in no_cp[0] and 'conformal' not in no_cp[0]
    lac[0]['grade_probabilities'][0] = 123
    assert original == untouched
    assert prepare_mode_rows([{'grade_probabilities': list(LOW)}], 'no_cp')


@pytest.mark.parametrize('mode', ['ordinal', 'lac'])
def test_missing_conformal_data_cannot_silently_fall_back(mode):
    missing = row()
    del missing['conformal']
    with pytest.raises(ValueError):
        prepare_mode_rows([missing], mode)
    unavailable = row()
    unavailable['conformal']['status'] = 'unavailable'
    with pytest.raises(ValueError):
        prepare_mode_rows([unavailable], mode)
    missing = row()
    del missing['conformal']['prediction_set' if mode == 'ordinal' else 'baseline_lac_set']
    with pytest.raises(ValueError):
        prepare_mode_rows([missing], mode)


def test_disagreeing_or_invalid_conformal_sets_and_modes_are_rejected():
    disagree = row()
    disagree['conformal_set'] = [4]
    with pytest.raises(ValueError, match='disagree'):
        prepare_mode_rows([disagree], 'ordinal')
    for invalid in ([True], [0, 0], [5], None):
        malformed = row()
        malformed['conformal']['baseline_lac_set'] = invalid
        with pytest.raises(ValueError):
            prepare_mode_rows([malformed], 'lac')
    with pytest.raises(ValueError):
        prepare_mode_rows([row()], 'unknown')


@pytest.mark.parametrize('price', DECISION_REVIEW_PRICES)
def test_review_outcome_ledger_is_fixed_across_all_decision_prices(price):
    result = evaluate_policy([row(HIGH)], [4], 'dual_budget', 1., 1., price)
    assert result['status'] == 'evaluated'
    assert result['actions'] == [Action.SPECIALIST_REVIEW.value]
    assert result['metrics']['proxy_review_cost'] == .25
    assert result['metrics']['proxy_total_cost_per_image'] == .25
    assert result['evaluation_ledger'] == EVALUATION_LEDGER


def test_fixed_ledger_includes_recapture_and_severity_costs():
    rows = [row(quality=.1), row()]
    result = evaluate_policy(rows, [3, 2], 'plain', 0., .5, 2.)
    assert result['actions'] == [Action.RECAPTURE.value, Action.AUTO_NONREFERABLE.value]
    metrics = result['metrics']
    assert metrics['proxy_recapture_cost'] == .2
    assert metrics['proxy_missed_referable_cost'] == 3
    assert metrics['proxy_severe_recapture_cost'] == 6
    assert metrics['proxy_total_cost_per_image'] == 4.6


def test_evaluation_label_changes_cannot_change_actions_but_do_change_outcomes():
    rows = [row(OPTIONAL), row(LOW), row(HIGH)]
    first = evaluate_policy(rows, [0, 0, 0], 'confidence_only', 1., .4, .5)
    second = evaluate_policy(rows, [4, 4, 4], 'confidence_only', 1., .4, .5)
    assert first['actions'] == second['actions']
    assert first['metrics']['proxy_total_cost_per_image'] != second['metrics']['proxy_total_cost_per_image']
    assert rows == [row(OPTIONAL), row(LOW), row(HIGH)]


def test_independent_tuning_depends_on_tuning_labels_and_retains_every_candidate():
    low = tune_policy([row(OPTIONAL)], [0], 'confidence_only', 1., .4)
    high = tune_policy([row(OPTIONAL)], [4], 'confidence_only', 1., .4)
    assert low['status'] == high['status'] == 'tuned'
    assert low['chosen_decision_review_price'] == .5
    assert high['chosen_decision_review_price'] == .05
    assert len(low['candidates']) == len(high['candidates']) == 6
    assert all('actions' not in candidate for candidate in low['candidates'])
    # A separately evaluated grade vector cannot mutate a selected tuning price.
    chosen = low['chosen_decision_review_price']
    evaluate_policy([row(OPTIONAL)], [4], 'confidence_only', 1., .4, chosen)
    assert low['chosen_decision_review_price'] == chosen


def test_tuning_uses_fixed_outcome_cost_not_a_candidates_allocation_price():
    tuned = tune_policy([row(HIGH)], [4], 'dual_budget', 1., .4)
    assert tuned['chosen_decision_review_price'] == DECISION_REVIEW_PRICES[0]
    assert {candidate['metrics']['proxy_total_cost_per_image'] for candidate in tuned['candidates']} == {.25}


def test_guard_overload_never_invents_actions_costs_or_a_chosen_threshold():
    rows = [row(ordinal=(0, 1, 2, 3, 4))] * 3
    evaluated = evaluate_policy(rows, [0, 0, 0], 'dual_budget', .9, 1., .05)
    assert evaluated['status'] == 'overloaded'
    assert evaluated['diagnostics']['review_capacity'] == 2
    assert evaluated['diagnostics']['mandatory_review_shortfall'] == 1
    assert evaluated['diagnostics']['unsafe_clearance_permitted'] is False
    assert 'actions' not in evaluated and 'metrics' not in evaluated
    tuned = tune_policy(rows, [0, 0, 0], 'dual_budget', .9, 1.)
    assert tuned['status'] == 'untunable'
    assert 'chosen_decision_review_price' not in tuned and 'metrics' not in tuned
    assert len(tuned['candidates']) == 6
    assert all(candidate['status'] == 'overloaded' for candidate in tuned['candidates'])


@pytest.mark.parametrize('method', METHODS)
def test_wrapper_preserves_production_guards_exact_capacities_and_action_assignment(method):
    rows = [row(HIGH), row(quality=.1), row(OPTIONAL), row(), row()]
    result = evaluate_policy(rows, [4, 0, 0, 0, 0], method, .6, .2, .5)
    expected = allocate_rows(rows, PolicyConfig(review_budget=.6, recapture_budget=.2, review_cost=.5), method)
    assert result['actions'] == [action.value for action in expected]
    assert result['actions'][0] == Action.SPECIALIST_REVIEW.value
    assert result['actions'][1] != Action.AUTO_NONREFERABLE.value
    assert result['metrics'] == screening_metrics([4, 0, 0, 0, 0],
                                                 [r['grade_probabilities'] for r in rows], expected,
                                                 PolicyConfig(review_budget=.6, recapture_budget=.2))
    assert result['metrics']['review_capacity'] == 3
    assert result['metrics']['recapture_capacity'] == 1
    assert not result['metrics']['review_budget_violation']
    assert not result['metrics']['recapture_budget_violation']


def test_plain_and_single_budget_restrictions_are_retained_at_extreme_prices():
    rows = [row(OPTIONAL), row(quality=.1)]
    plain = evaluate_policy(rows, [0, 0], 'plain', 1., 1., .001)
    assert plain['actions'][0] == Action.AUTO_NONREFERABLE.value
    single = evaluate_policy(rows, [0, 0], 'single_budget', 1., 1., 2.)
    assert Action.RECAPTURE.value not in single['actions']
    assert single['actions'][1] == Action.SPECIALIST_REVIEW.value
    overloaded = evaluate_policy(rows, [0, 0], 'single_budget', 0., 1., 2.)
    assert overloaded['status'] == 'overloaded'


def test_empty_explicit_conformal_set_retains_review_guard():
    rows = prepare_mode_rows([row(ordinal=())], 'ordinal')
    assert evaluate_policy(rows, [0], 'plain', 1., 0.)['actions'] == [Action.SPECIALIST_REVIEW.value]


@pytest.mark.parametrize('labels', [[True], [.0], [-1], [5], [], [[0]]])
def test_invalid_evaluation_grades_are_rejected(labels):
    with pytest.raises(ValueError):
        evaluate_policy([row()], labels, 'plain', 1., .4)


@pytest.mark.parametrize('price', [True, -.1, np.nan, np.inf])
def test_invalid_decision_prices_are_rejected(price):
    with pytest.raises(ValueError):
        evaluate_policy([row()], [0], 'plain', 1., .4, price)


def test_patient_bootstrap_preserves_cluster_pairing_and_sign():
    result = paired_patient_delta([0, 0, 0, 0], [1, 1, 3, 3], ['a', 'a', 'b', 'b'], repeats=1000)
    assert result['difference_after_minus_before'] == 2
    assert result['ci95'] == [1., 3.]
    assert result['patients'] == 2 and result['images'] == 4
    assert result == paired_patient_delta([0, 0, 0, 0], [1, 1, 3, 3], ['a', 'a', 'b', 'b'], repeats=1000)
    variable_sizes = paired_patient_delta([0, 0, 0], [1, 1, 4], ['a', 'a', 'b'], repeats=100)
    assert variable_sizes['difference_after_minus_before'] == 2


@pytest.mark.parametrize(('before', 'after', 'groups', 'repeats'), [
    ([0], [0, 1], ['a'], 10), ([0], [np.nan], ['a'], 10), ([0], [0], ['a'], 0),
    ([0], [0], ['a'], True),
])
def test_patient_bootstrap_invalid_inputs(before, after, groups, repeats):
    with pytest.raises(ValueError):
        paired_patient_delta(before, after, groups, repeats=repeats)
