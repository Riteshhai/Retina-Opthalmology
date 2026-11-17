import numpy as np
import pytest
from qsart.triage import (Action, BudgetInfeasible, PolicyConfig, allocate,
                          pointwise_action, screening_metrics)


def scores(n=20):
    p = np.tile([.2, .1, .3, .2, .2], (n, 1))
    return p, np.ones(n), np.full(n, .8)


def test_exact_population_budgets_and_determinism():
    p, q, u = scores()
    c = PolicyConfig(review_budget=.25, recapture_budget=.15)
    a = allocate(p, q, u, c)
    assert a == allocate(p, q, u, c)
    assert a.count(Action.SPECIALIST_REVIEW) <= 5
    assert a.count(Action.RECAPTURE) <= 3
    assert len(a) == 20


def test_zero_maximum_and_fractional_budgets():
    p, q, u = scores(3)
    assert allocate(p, q, u, PolicyConfig(0, 0)) == [Action.AUTO_SCREEN]*3
    assert len(allocate(p, q, u, PolicyConfig(1, 1))) == 3
    assert allocate(p, q, u, PolicyConfig(.1, .1)) == [Action.AUTO_SCREEN]*3


def test_low_quality_infeasible_instead_of_silent_clearance():
    p, q, u = scores()
    q[:] = 0
    with pytest.raises(BudgetInfeasible):
        allocate(p, q, u, PolicyConfig(.1, .2))
    a = allocate(p, q, u, PolicyConfig(.5, .5))
    assert Action.AUTO_SCREEN not in a


def test_auto_positive_is_not_a_missed_referable_case():
    p = np.array([[.01, .01, .9, .04, .04], [.9, .02, .03, .03, .02]])
    m = screening_metrics([2, 4], p, [Action.AUTO_SCREEN]*2)
    assert m['missed_referable_auto'] == 1
    assert m['auto_sensitivity'] == .5


def test_high_risk_low_quality_pointwise_routes_to_review():
    action, _ = pointwise_action([0, 0, .1, .2, .7], 0, 1)
    assert action == Action.SPECIALIST_REVIEW


def test_invalid_config_and_scores():
    with pytest.raises(ValueError):
        PolicyConfig(review_budget=float('nan'))
    with pytest.raises(ValueError):
        allocate([[1, 1, 1, 1, 1]], [1], [0])


def test_combined_budget_ablation_retains_total_capacity():
    p, q, u = scores(20)
    c = PolicyConfig(.3, .2)
    a = allocate(p, q, u, c, 'combined_budget')
    assert sum(x != Action.AUTO_SCREEN for x in a) <= 10
