"""Guarded non-referable clearance and explicit, independent capacity limits.

This prospective v2 policy does not reinterpret historical AUTO_SCREEN results.
Costs are research proxies. Review and recapture are recommendations, not outcomes.
"""
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
import math
from numbers import Integral

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

from qsart.triage import validate_scores


class Action(str, Enum):
    AUTO_NONREFERABLE = 'AUTO_NONREFERABLE'
    RECAPTURE = 'RECAPTURE'
    SPECIALIST_REVIEW = 'SPECIALIST_REVIEW'


class BudgetOverloaded(ValueError):
    """An unsafe clearance would be necessary to meet the configured capacities."""

    def __init__(self, diagnostics):
        self.diagnostics = dict(diagnostics)
        super().__init__('Required deferrals exceed configured independent capacities')


@dataclass(frozen=True)
class PolicyConfig:
    review_budget: float = .30
    recapture_budget: float = .20
    minimum_quality: float = .50
    review_risk: float = .50
    uncertainty_threshold: float = .75
    referable_grade: int = 2
    severe_grade: int = 3
    severe_risk: float = .25
    severity_costs: tuple[float, ...] = (0., 0., 3., 6., 10.)
    review_cost: float = .25
    recapture_cost: float = .20
    uncertainty_weight: float = .50
    quality_weight: float = 1.

    def __post_init__(self):
        for name in ('review_budget', 'recapture_budget', 'minimum_quality',
                     'review_risk', 'uncertainty_threshold', 'severe_risk'):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f'{name} must be finite in [0,1]')
        for name in ('referable_grade', 'severe_grade'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value not in (1, 2, 3, 4):
                raise ValueError(f'{name} must be an integer in 1..4')
        if self.severe_grade < self.referable_grade:
            raise ValueError('severe_grade must be at least referable_grade')
        if len(self.severity_costs) != 5 or any(
                not math.isfinite(value) or value < 0 for value in self.severity_costs):
            raise ValueError('Five nonnegative finite severity costs required')
        for name in ('review_cost', 'recapture_cost', 'uncertainty_weight', 'quality_weight'):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f'{name} must be finite and nonnegative')


METHODS = ('plain', 'confidence_only', 'entropy_only', 'severity_only',
           'quality_only', 'single_budget', 'naive', 'dual_budget')


def capacity(budget: float, n: int) -> int:
    """Floor each declared decimal workload independently, without epsilon slots."""
    return int(Decimal(str(budget)) * n)


def _conformal_sets(sets, n):
    if sets is None:
        return [None] * n
    if len(sets) != n:
        raise ValueError('Conformal sets must have N entries')
    validated = []
    for values in sets:
        if values is None:
            validated.append(None)
            continue
        if isinstance(values, (str, bytes)):
            raise ValueError('Conformal labels must be integers in 0..4')
        try:
            labels = list(values)
        except TypeError:
            raise ValueError('Conformal labels must be integers in 0..4') from None
        if any(isinstance(value, bool) or not isinstance(value, Integral)
               or value not in range(5) for value in labels):
            raise ValueError('Conformal labels must be integers in 0..4')
        if len(set(labels)) != len(labels):
            raise ValueError('Conformal sets cannot contain duplicate labels')
        validated.append(labels)
    return validated


def _guards(p, q, u, config, conformal_sets=None):
    sets = _conformal_sets(conformal_sets, len(p))
    predicted = p.argmax(1)
    risk = p[:, config.referable_grade:].sum(1)
    severe_risk = p[:, config.severe_grade:].sum(1)
    poor = q < config.minimum_quality
    mandatory_review = np.zeros(len(p), dtype=bool)
    reasons = []
    for i in range(len(p)):
        why = []
        if predicted[i] >= config.referable_grade:
            why.append('PREDICTED_REFERABLE_GRADE')
        if risk[i] >= config.review_risk:
            why.append('HIGH_REFERABLE_PROBABILITY')
        if severe_risk[i] >= config.severe_risk:
            why.append('HIGH_SEVERE_PROBABILITY')
        if sets[i] is not None:
            if not sets[i]:
                why.append('EMPTY_CONFORMAL_SET')
            elif any(value >= config.referable_grade for value in sets[i]):
                why.append('CONFORMAL_SET_INCLUDES_REFERABLE')
        if u[i] >= config.uncertainty_threshold and not poor[i]:
            why.append('HIGH_PREDICTIVE_UNCERTAINTY')
        mandatory_review[i] = bool(why)
        if poor[i]:
            why.append('INSUFFICIENT_IMAGE_QUALITY')
        reasons.append(why or ['PROVISIONAL_NONREFERABLE'])
    return mandatory_review, poor, reasons


def pointwise_action(probabilities, quality, uncertainty, config=PolicyConfig(), conformal_set=None):
    """Recommend an action for one image; no cohort-capacity guarantee is implied."""
    p, q, u = validate_scores([probabilities], [quality], [uncertainty])
    mandatory, poor, reasons = _guards(p, q, u, config, [conformal_set])
    if mandatory[0]:
        return Action.SPECIALIST_REVIEW, reasons[0]
    if poor[0]:
        return Action.RECAPTURE, reasons[0]
    return Action.AUTO_NONREFERABLE, reasons[0]


def _diagnostics(n, mandatory, poor, config):
    review_capacity = capacity(config.review_budget, n)
    recapture_capacity = capacity(config.recapture_budget, n)
    needed_review = int(mandatory.sum())
    flexible_poor = int((poor & ~mandatory).sum())
    needed_total = needed_review + flexible_poor
    spill_to_review = max(0, flexible_poor - recapture_capacity)
    return {
        'n': n, 'needed_review': needed_review,
        'needed_recapture': flexible_poor,
        'needed_recapture_or_review': flexible_poor,
        'needed_recapture_after_available_review': max(
            0, flexible_poor - max(0, review_capacity - needed_review)),
        'needed_total': needed_total,
        'review_capacity': review_capacity, 'recapture_capacity': recapture_capacity,
        'total_capacity': review_capacity + recapture_capacity,
        'mandatory_review_shortfall': max(0, needed_review - review_capacity),
        'review_shortfall_after_recapture': max(0, needed_review + spill_to_review - review_capacity),
        'total_shortfall': max(0, needed_total - review_capacity - recapture_capacity),
        'eligible_auto_nonreferable': n - needed_total,
        'unsafe_clearance_permitted': False,
    }


def _proxy_costs(p, q, u, config, method):
    """Prospective, untuned priorities; only predictions enter these costs."""
    weights = np.asarray(config.severity_costs)
    weighted_miss = (p[:, config.referable_grade:] * weights[config.referable_grade:]).sum(1)
    if method == 'plain':
        clearance = np.zeros(len(p))
    elif method == 'confidence_only':
        clearance = 1 - p.max(1)
    elif method == 'entropy_only':
        clearance = u
    elif method == 'severity_only':
        clearance = weighted_miss
    elif method == 'quality_only':
        clearance = 1 - q
    elif method == 'naive':
        clearance = (1 - q) + u
    else:
        clearance = weighted_miss + config.quality_weight * (1 - q) + config.uncertainty_weight * u
    # Recapture has delay/risk cost; it does not erase expected disease risk.
    recapture = config.recapture_cost + weighted_miss + .5 * q
    review = np.full(len(p), config.review_cost)
    return np.column_stack((clearance, recapture, review))


def allocate(probabilities, quality, uncertainty, config=PolicyConfig(), method='dual_budget',
             conformal_sets=None):
    """Minimize a prediction-only proxy with immutable safety guards and two caps.

    An overloaded cohort raises BudgetOverloaded; unsafe cases are never cleared.
    All strategies share mandatory guards. ``plain`` means guarded base routing;
    use no_deferral_predictions for the distinct, raw classification baseline.
    ``single_budget`` is review-only: poor-quality cases also require review,
    recapture is unused, and the review cap may therefore overload sooner.
    """
    p, q, u = validate_scores(probabilities, quality, uncertainty)
    if method not in METHODS:
        raise ValueError(f'Unknown policy: {method}')
    n = len(p)
    mandatory, poor, _ = _guards(p, q, u, config, conformal_sets)
    if method == 'single_budget':
        mandatory = mandatory | poor
    diagnostics = _diagnostics(n, mandatory, poor, config)
    if diagnostics['mandatory_review_shortfall'] or diagnostics['review_shortfall_after_recapture']:
        raise BudgetOverloaded(diagnostics)
    costs = _proxy_costs(p, q, u, config, method)
    bounds = [(0., 1.) for _ in range(3 * n)]
    for i in range(n):
        if mandatory[i]:
            bounds[3 * i] = (0., 0.)
            bounds[3 * i + 1] = (0., 0.)
        elif poor[i]:
            bounds[3 * i] = (0., 0.)
        else:
            # Recapture requires an image-quality indication, not an unused slot.
            bounds[3 * i + 1] = (0., 0.)
            if method == 'plain':
                bounds[3 * i + 2] = (0., 0.)
    rows = np.repeat(np.arange(n), 3)
    eq = coo_matrix((np.ones(3 * n), (rows, np.arange(3 * n))), shape=(n, 3 * n)).tocsr()
    columns = np.concatenate((3 * np.arange(n) + 1, 3 * np.arange(n) + 2))
    ub = coo_matrix((np.ones(2 * n), (np.repeat([0, 1], n), columns)), shape=(2, 3 * n)).tocsr()
    fit = linprog(costs.ravel(), A_ub=ub,
                  b_ub=[diagnostics['recapture_capacity'], diagnostics['review_capacity']],
                  A_eq=eq, b_eq=np.ones(n), bounds=bounds, method='highs')
    if not fit.success:
        raise RuntimeError('Guarded allocation solver failed after capacity feasibility audit')
    assignment = fit.x.reshape(n, 3)
    if not np.allclose(assignment, assignment.round(), atol=1e-6):
        raise RuntimeError('Allocation did not return an integral solution')
    choices = assignment.argmax(1)
    if ((choices == 1).sum() > diagnostics['recapture_capacity']
            or (choices == 2).sum() > diagnostics['review_capacity']
            or ((choices == 0) & (mandatory | poor)).any()
            or (mandatory & (choices != 2)).any()):
        raise RuntimeError('Allocation violated a capacity or mandatory guard')
    return [list(Action)[i] for i in choices]


def allocate_rows(rows, config=PolicyConfig(), method='dual_budget'):
    """Use engine rows directly; labels and identifiers never enter allocation."""
    probabilities, quality, uncertainty, conformal = [], [], [], []
    for row in rows:
        probabilities.append(row['grade_probabilities'])
        if isinstance(row.get('quality'), dict):
            quality.append(1 - row['quality']['poor_probability'])
        else:
            quality.append(row.get('quality_good_probability', row.get('quality_probability')))
        value = row.get('uncertainty', row.get('normalized_entropy', row.get('entropy')))
        uncertainty.append(value.get('entropy') if isinstance(value, dict) else value)
        conformal.append(row.get('conformal_set', (row.get('conformal') or {}).get('prediction_set')))
    return allocate(probabilities, quality, uncertainty, config, method, conformal)


def no_deferral_predictions(probabilities, config=PolicyConfig()):
    """Raw classifier baseline, deliberately returning predictions, not triage actions."""
    p = np.asarray(probabilities, dtype=float)
    validate_scores(p, np.ones(len(p)), np.zeros(len(p)))
    predicted = p.argmax(1)
    return {
        'predicted_grades': predicted.tolist(),
        'predicted_referable': (predicted >= config.referable_grade).tolist(),
        'is_no_deferral_baseline': True,
        'has_triage_actions': False,
    }


def screening_metrics(grades, probabilities, actions, config=PolicyConfig()):
    """Descriptive image-level outcomes; only evaluation consumes true labels."""
    p = np.asarray(probabilities, dtype=float)
    g = np.asarray(grades)
    validate_scores(p, np.ones(len(g)), np.zeros(len(g)))
    if g.shape != (len(p),) or g.dtype.kind not in 'iu' or not np.isin(g, np.arange(5)).all():
        raise ValueError('Integer true grades must have N entries in 0..4')
    a = np.asarray([Action(value).value for value in actions])
    if a.shape != g.shape:
        raise ValueError('Actions must have N entries')
    auto = a == Action.AUTO_NONREFERABLE.value
    review = a == Action.SPECIALIST_REVIEW.value
    recapture = a == Action.RECAPTURE.value
    referable = g >= config.referable_grade
    severe = g >= config.severe_grade
    missed = referable & auto
    severity = np.asarray(config.severity_costs)[g]
    miss_cost = float((severity * missed).sum())
    severe_recapture_cost = float((severity * severe * recapture).sum())
    review_cost = float(config.review_cost * review.sum())
    recapture_cost = float(config.recapture_cost * recapture.sum())

    def ratio(num, den):
        return float(num / den) if den else None

    return {
        'n': len(g), 'unit': 'image',
        'referable_count': int(referable.sum()), 'severe_count': int(severe.sum()),
        'auto_nonreferable_count': int(auto.sum()),
        'review_count': int(review.sum()), 'recapture_count': int(recapture.sum()),
        'coverage': float(auto.mean()), 'review_rate': float(review.mean()),
        'recapture_rate': float(recapture.mean()),
        'missed_referable_auto': int(missed.sum()),
        'missed_referable_per_1000_all': 1000 * float(missed.mean()),
        'missed_referable_fraction_of_referable': ratio(missed.sum(), referable.sum()),
        'false_omission_fraction_of_auto': ratio(missed.sum(), auto.sum()),
        'missed_severe_auto': int((severe & auto).sum()),
        'referable_deferred_to_recapture': int((referable & recapture).sum()),
        'severe_deferred_to_recapture': int((severe & recapture).sum()),
        'severe_recapture_fraction_of_severe': ratio((severe & recapture).sum(), severe.sum()),
        'proxy_missed_referable_cost': miss_cost,
        'proxy_severe_recapture_cost': severe_recapture_cost,
        'proxy_review_cost': review_cost, 'proxy_recapture_cost': recapture_cost,
        'proxy_total_cost_per_image': (miss_cost + severe_recapture_cost + review_cost + recapture_cost) / len(g),
        'review_capacity': capacity(config.review_budget, len(g)),
        'recapture_capacity': capacity(config.recapture_budget, len(g)),
        'review_budget_violation': bool(review.sum() > capacity(config.review_budget, len(g))),
        'recapture_budget_violation': bool(recapture.sum() > capacity(config.recapture_budget, len(g))),
        'interpretation': ('AUTO_NONREFERABLE is provisional clearance; every true referable clearance is a miss. '
                           'Deferrals are recommendations, not completed care. Costs are unvalidated research proxies.'),
    }
