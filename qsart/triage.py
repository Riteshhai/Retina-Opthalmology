"""Research allocation proxy; deferred actions do not imply successful treatment."""
from dataclasses import dataclass
from enum import Enum
import math

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix
from scipy.sparse import vstack


class Action(str, Enum):
    AUTO_SCREEN = 'AUTO_SCREEN'
    RECAPTURE = 'RECAPTURE'
    SPECIALIST_REVIEW = 'SPECIALIST_REVIEW'


class BudgetInfeasible(ValueError):
    """Insufficient capacity to avoid auto-screening low-quality images."""


@dataclass(frozen=True)
class PolicyConfig:
    review_budget: float = 0.30
    recapture_budget: float = 0.20
    minimum_quality: float = 0.50
    review_risk: float = 0.50
    uncertainty_threshold: float = 0.75
    referable_grade: int = 2
    # Exploratory utility weights, not established clinical costs.
    severity_costs: tuple[float, ...] = (1., 1., 3., 6., 10.)

    def __post_init__(self):
        for name in ('review_budget', 'recapture_budget', 'minimum_quality',
                     'review_risk', 'uncertainty_threshold'):
            x = getattr(self, name)
            if not math.isfinite(x) or not 0 <= x <= 1:
                raise ValueError(f'{name} must be finite in [0,1]')
        if self.referable_grade not in (1, 2, 3, 4):
            raise ValueError('referable_grade must be 1..4')
        if len(self.severity_costs) != 5 or any(
            not math.isfinite(x) or x < 0 for x in self.severity_costs
        ):
            raise ValueError('Five nonnegative finite severity costs required')


def validate_scores(probabilities, quality, uncertainty):
    p = np.asarray(probabilities, dtype=float)
    q = np.asarray(quality, dtype=float)
    u = np.asarray(uncertainty, dtype=float)
    if p.ndim != 2 or p.shape[1] != 5 or len(p) == 0:
        raise ValueError('Nonempty N x 5 probabilities required')
    if q.shape != (len(p),) or u.shape != (len(p),):
        raise ValueError('Quality and uncertainty must have N entries')
    for x in (p, q, u):
        if not np.isfinite(x).all() or (x < 0).any() or (x > 1).any():
            raise ValueError('Scores must be finite in [0,1]')
    if not np.allclose(p.sum(axis=1), 1., atol=1e-5):
        raise ValueError('Grade probabilities must sum to 1')
    return p, q, u


def pointwise_action(probabilities, quality, uncertainty, config=PolicyConfig()):
    """Single image routing has no population workload guarantee."""
    p, q, u = validate_scores([probabilities], [quality], [uncertainty])
    risk = float(p[0, config.referable_grade:].sum())
    if risk >= config.review_risk:
        return Action.SPECIALIST_REVIEW, ['POSSIBLE_REFERABLE_DR']
    if q[0] < config.minimum_quality:
        return Action.RECAPTURE, ['INSUFFICIENT_IMAGE_QUALITY']
    if u[0] >= config.uncertainty_threshold:
        return Action.SPECIALIST_REVIEW, ['HIGH_PREDICTIVE_UNCERTAINTY']
    return Action.AUTO_SCREEN, ['PROVISIONAL_RESULT']


METHODS = ('plain', 'confidence_only', 'quality_only', 'severity_only',
           'single_budget', 'naive', 'dual_budget')


def allocate(probabilities, quality, uncertainty, config=PolicyConfig(),
             method='dual_budget'):
    """Solve a three-action transportation LP with integer cohort capacities.

    Only model outputs enter allocation. The assignment polytope is integral.
    Costs are a documented research proxy, not actual downstream care outcomes.
    Quality restrictions can make a configuration infeasible; never relax silently.
    """
    p, q, u = validate_scores(probabilities, quality, uncertainty)
    if method not in (*METHODS, 'combined_budget'):
        raise ValueError(f'Unknown policy: {method}')
    n = len(p)
    if method == 'plain':
        return [Action.AUTO_SCREEN] * n
    predicted = p.argmax(axis=1)
    weights = np.asarray(config.severity_costs)
    # Expected grade error under model distribution for the provisional grade.
    cost = ((np.arange(5)[None, :] != predicted[:, None]) * weights * p).sum(1)
    if method == 'confidence_only':
        cost = u
    elif method == 'quality_only':
        cost = 1-q
    elif method == 'severity_only':
        cost = p @ weights
    else:
        cost = cost + (1-q) + .5*u
    # No assumption that recapture cures poor quality. This is a delay/deferral proxy.
    recapture_cost = .25 + p @ weights + .5*q
    if method in ('quality_only', 'naive'):
        recapture_cost = q
    review_cost = np.full(n, .25)
    costs = np.column_stack((cost, recapture_cost, review_cost)).ravel()
    bounds = [(0., 1.) for _ in range(3*n)]
    for i in range(n):
        if q[i] < config.minimum_quality:
            bounds[3*i] = (0., 0.)
        if method in ('confidence_only', 'severity_only', 'single_budget'):
            bounds[3*i+1] = (0., 0.)
    review_slots = math.floor(config.review_budget*n + 1e-10)
    recapture_slots = math.floor(config.recapture_budget*n + 1e-10)
    row = np.repeat(np.arange(n), 3)
    eq = coo_matrix((np.ones(3*n), (row, np.arange(3*n))), shape=(n, 3*n)).tocsr()
    ub = coo_matrix((np.ones(2*n),
                    (np.repeat([0, 1], n),
                     np.concatenate((3*np.arange(n)+1, 3*np.arange(n)+2)))),
                   shape=(2, 3*n)).tocsr()
    capacities = [recapture_slots, review_slots]
    if method == 'combined_budget':
        total = review_slots + recapture_slots
        # Deliberate ablation: treat capacity as fungible while matching total slots.
        ub = vstack((ub, ub.sum(axis=0)))
        capacities = [total, total, total]
    fit = linprog(costs, A_ub=ub, b_ub=capacities,
                  A_eq=eq, b_eq=np.ones(n), bounds=bounds, method='highs')
    if not fit.success:
        raise BudgetInfeasible('Budgets cannot accommodate required quality deferrals')
    assignment = fit.x.reshape(n, 3)
    if not np.allclose(assignment, assignment.round(), atol=1e-6):
        raise RuntimeError('Allocation did not return an integral solution')
    choices = assignment.argmax(1)
    violation = ((choices != 0).sum() > review_slots+recapture_slots if method == 'combined_budget'
                 else (choices == 1).sum() > recapture_slots or (choices == 2).sum() > review_slots)
    if violation:
        raise RuntimeError('Allocation exceeded a cohort budget')
    return [list(Action)[i] for i in choices]


def screening_metrics(grades, probabilities, actions, config=PolicyConfig()):
    p = np.asarray(probabilities, dtype=float)
    g = np.asarray(grades)
    validate_scores(p, np.ones(len(g)), np.zeros(len(g)))
    a = np.asarray([Action(x).value for x in actions])
    if a.shape != g.shape or not np.isin(g, np.arange(5)).all():
        raise ValueError('Actions and integer grades must have N entries')
    auto = a == Action.AUTO_SCREEN.value
    pred = p.argmax(1)
    positive = g >= config.referable_grade
    positive_prediction = pred >= config.referable_grade
    review = a == Action.SPECIALIST_REVIEW.value
    recapture = a == Action.RECAPTURE.value
    fn = positive & ~positive_prediction & auto
    def ratio(num, den):
        return float(num/den) if den else None
    return {
        'n': len(g), 'auto_count': int(auto.sum()),
        'review_rate': float(review.mean()), 'recapture_rate': float(recapture.mean()),
        'coverage': float(auto.mean()), 'missed_referable_auto': int(fn.sum()),
        'missed_severe_auto': int((fn & (g >= 3)).sum()),
        'referable_deferred_to_recapture': int((positive & recapture).sum()),
        'severe_deferred_to_recapture': int(((g >= 3) & recapture).sum()),
        'auto_sensitivity': ratio((positive & positive_prediction & auto).sum(),
                                  (positive & auto).sum()),
        'auto_specificity': ratio((~positive & ~positive_prediction & auto).sum(),
                                  (~positive & auto).sum()),
        'weighted_auto_error_per_image': float((
            np.asarray(config.severity_costs)[g.astype(int)]*(pred != g)*auto).mean()),
        'selective_grade_error': ratio(((pred != g) & auto).sum(), auto.sum()),
        'review_budget_violation': bool(review.mean() > config.review_budget+1e-9),
        'recapture_budget_violation': bool(recapture.mean() > config.recapture_budget+1e-9),
        'interpretation': 'Sensitivity/specificity use AUTO_SCREEN only; deferral is not diagnosis.'
    }
