"""Explicitly synthetic score fixtures test prospective v2 policy semantics."""
import numpy as np
import pytest

from qsart.triage_v2 import (Action, BudgetOverloaded, METHODS, PolicyConfig,
                             allocate, allocate_rows, capacity, no_deferral_predictions,
                             pointwise_action, screening_metrics)


LOW = [.96, .02, .01, .005, .005]
GRADE_TWO_WITH_LOW_SUM = [.29, .28, .30, .07, .06]


@pytest.mark.parametrize(('p', 'q', 'u', 'prediction_set', 'reason'), [
    (GRADE_TWO_WITH_LOW_SUM, 1, .1, None, 'PREDICTED_REFERABLE_GRADE'),
    ([.30, .19, .2, .16, .15], 1, .1, None, 'HIGH_REFERABLE_PROBABILITY'),
    ([.4, .2, .1, .15, .15], 1, .1, None, 'HIGH_SEVERE_PROBABILITY'),
    (LOW, 1, .75, None, 'HIGH_PREDICTIVE_UNCERTAINTY'),
    (LOW, 1, .1, [0, 1, 2], 'CONFORMAL_SET_INCLUDES_REFERABLE'),
    (LOW, 1, .1, [], 'EMPTY_CONFORMAL_SET'),
])
def test_each_guard_prevents_clearance(p, q, u, prediction_set, reason):
    action, reasons = pointwise_action(p, q, u, conformal_set=prediction_set)
    assert action == Action.SPECIALIST_REVIEW
    assert reason in reasons


def test_quality_routing_severe_override_and_threshold_boundaries():
    assert pointwise_action(LOW, .499, .9)[0] == Action.RECAPTURE
    assert pointwise_action(LOW, .5, .749, conformal_set=[0, 1])[0] == Action.AUTO_NONREFERABLE
    p = [.01, .01, .01, .01, .96]
    assert pointwise_action(p, .01, 1)[0] == Action.SPECIALIST_REVIEW
    c = PolicyConfig(referable_grade=1, severe_grade=4, review_risk=.2, severe_risk=.1)
    assert pointwise_action([.7, .25, .03, .01, .01], 1, 0, c)[0] == Action.SPECIALIST_REVIEW


def mixed_scores():
    p = np.array([LOW] * 10)
    p[0] = [.01, .01, .05, .03, .90]
    p[1] = GRADE_TWO_WITH_LOW_SUM
    q = np.ones(10)
    q[0] = .1
    q[2:4] = .1
    return p, q, np.full(10, .1)


@pytest.mark.parametrize('method', METHODS)
def test_all_deployed_methods_share_guards_and_independent_caps(method):
    p, q, u = mixed_scores()
    review_slots = 4 if method == 'single_budget' else 3
    c = PolicyConfig(review_budget=review_slots / 10, recapture_budget=.2)
    actions = allocate(p, q, u, c, method)
    assert actions[:2] == [Action.SPECIALIST_REVIEW] * 2
    assert all(actions[i] != Action.AUTO_NONREFERABLE for i in range(4))
    assert actions.count(Action.SPECIALIST_REVIEW) <= review_slots
    assert actions.count(Action.RECAPTURE) <= 2
    assert actions == allocate(p, q, u, c, method)


def test_single_budget_really_uses_only_review_and_reports_its_extra_requirement():
    p, q, u = mixed_scores()
    actions = allocate(p, q, u, PolicyConfig(.4, 1), 'single_budget')
    assert actions[:4] == [Action.SPECIALIST_REVIEW] * 4
    assert Action.RECAPTURE not in actions
    with pytest.raises(BudgetOverloaded) as exc:
        allocate(p, q, u, PolicyConfig(.3, 1), 'single_budget')
    assert exc.value.diagnostics['needed_review'] == 4
    assert exc.value.diagnostics['mandatory_review_shortfall'] == 1


def test_zero_capacity_is_valid_only_for_eligible_clearances():
    c = PolicyConfig(0, 0)
    assert allocate([LOW] * 3, [1] * 3, [0] * 3, c) == [Action.AUTO_NONREFERABLE] * 3
    with pytest.raises(BudgetOverloaded) as exc:
        allocate([LOW], [0], [0], c)
    d = exc.value.diagnostics
    assert d['needed_review'] == 0
    assert d['needed_recapture_or_review'] == 1
    assert d['needed_total'] == d['total_shortfall'] == 1
    assert d['total_capacity'] == 0
    assert d['unsafe_clearance_permitted'] is False


def test_fractional_capacities_floor_independently_and_poor_can_use_review():
    assert capacity(.58, 100) == 58
    assert capacity(.099999999999, 10) == 0
    c = PolicyConfig(.2, .2)
    assert allocate([LOW] * 3, [1] * 3, [0] * 3, c) == [Action.AUTO_NONREFERABLE] * 3
    with pytest.raises(BudgetOverloaded):
        allocate([LOW] * 3, [0, 1, 1], [0] * 3, c)
    actions = allocate([LOW] * 3, [0, 1, 1], [0] * 3, PolicyConfig(.34, 0))
    assert actions[0] == Action.SPECIALIST_REVIEW


def test_review_only_shortfall_cannot_be_hidden_by_recapture_capacity():
    with pytest.raises(BudgetOverloaded) as exc:
        allocate([[0, 0, 0, 0, 1]] * 5, [0] * 5, [0] * 5, PolicyConfig(.2, 1))
    assert exc.value.diagnostics['needed_review'] == 5
    assert exc.value.diagnostics['mandatory_review_shortfall'] == 4
    assert exc.value.diagnostics['total_shortfall'] == 0


def test_engine_row_adapter_and_labels_do_not_enter_allocation():
    rows = [{'grade_probabilities': LOW, 'quality': {'poor_probability': .1},
             'uncertainty': {'entropy': .1}, 'conformal_set': [0, 1], 'true_grade': 4},
            {'grade_probabilities': LOW, 'quality': {'poor_probability': .1},
             'uncertainty': {'entropy': .1}, 'conformal_set': [0, 2], 'true_grade': 0}]
    c = PolicyConfig(.5, 0)
    assert allocate_rows(rows, c, 'plain') == [Action.AUTO_NONREFERABLE, Action.SPECIALIST_REVIEW]
    rows[0]['true_grade'], rows[1]['true_grade'] = 0, 4
    assert allocate_rows(rows, c, 'plain') == [Action.AUTO_NONREFERABLE, Action.SPECIALIST_REVIEW]


def test_no_deferral_baseline_returns_binary_predictions_not_clearance_actions():
    result = no_deferral_predictions([LOW, [0, 0, 0, 0, 1]])
    assert result['predicted_referable'] == [False, True]
    assert result['has_triage_actions'] is False
    assert 'actions' not in result


def test_every_true_referable_clearance_counts_as_a_miss_and_denominators_are_explicit():
    m = screening_metrics([2, 4, 0, 3], [LOW] * 4,
                          [Action.AUTO_NONREFERABLE, Action.RECAPTURE,
                           Action.AUTO_NONREFERABLE, Action.SPECIALIST_REVIEW], PolicyConfig(.25, .25))
    assert m['missed_referable_auto'] == 1
    assert m['missed_referable_per_1000_all'] == 250
    assert m['missed_referable_fraction_of_referable'] == 1 / 3
    assert m['false_omission_fraction_of_auto'] == .5
    assert m['severe_deferred_to_recapture'] == 1
    assert m['severe_recapture_fraction_of_severe'] == .5
    assert m['review_count'] == m['recapture_count'] == 1
    assert m['proxy_total_cost_per_image'] > 0
    assert not m['review_budget_violation']
    assert screening_metrics([0], [LOW], [Action.SPECIALIST_REVIEW])[
        'false_omission_fraction_of_auto'] is None


@pytest.mark.parametrize('kwargs', [
    {'review_budget': float('nan')}, {'recapture_budget': -1}, {'severe_risk': float('inf')},
    {'referable_grade': 1.5}, {'severe_grade': 1}, {'review_cost': -1},
    {'severity_costs': (0, 0, 1, 2, float('nan'))},
])
def test_invalid_config(kwargs):
    with pytest.raises(ValueError):
        PolicyConfig(**kwargs)


@pytest.mark.parametrize(('p', 'q', 'u', 'sets'), [
    ([[1, 1, 1, 1, 1]], [1], [0], None),
    ([LOW], [float('nan')], [0], None),
    ([LOW], [1], [1.1], None),
    ([LOW], [1], [0], [[5]]),
    ([LOW], [1], [0], [[0, 0]]),
    ([LOW], [1], [0], [[True]]),
    ([LOW], [1], [0], [[.0]]),
    ([LOW], [1], [0], []),
])
def test_invalid_scores_and_conformal_sets(p, q, u, sets):
    with pytest.raises(ValueError):
        allocate(p, q, u, conformal_sets=sets)


def test_unknown_method_and_noninteger_true_grades_rejected():
    with pytest.raises(ValueError):
        allocate([LOW], [1], [0], method='unknown')
    with pytest.raises(ValueError):
        screening_metrics([1.2], [LOW], [Action.AUTO_NONREFERABLE])
