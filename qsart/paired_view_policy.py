"""First-view-only routing for the retrospective paired-view experiment."""

from dataclasses import dataclass
import math

import numpy as np

from qsart.triage_v2 import capacity


FEATURE_NAMES = ('poor_quality', 'entropy', 'referable_risk', 'severe_risk', 'predicted_referable')
BUDGETS = (0., .10, .25, .50, .75, 1.)
ACQUISITION_COST = .20
RANDOM_SEED = 20261116
COMPARATORS = ('quality', 'entropy', 'referable_risk', 'random')


def first_view_features(probabilities, good_quality):
    p = np.asarray(probabilities, dtype=float)
    good = np.asarray(good_quality, dtype=float)
    if p.ndim != 2 or not len(p) or p.shape[1] != 5 or good.shape != (len(p),):
        raise ValueError('One five-grade vector and quality score are required per initial view')
    if (not np.isfinite(p).all() or not np.isfinite(good).all() or ((p < 0) | (p > 1)).any()
            or not np.allclose(p.sum(axis=1), 1., rtol=0., atol=1e-5) or ((good < 0) | (good > 1)).any()):
        raise ValueError('Initial-view probabilities and quality scores must be valid')
    entropy = -(p * np.log(np.where(p > 0, p, 1.))).sum(axis=1) / math.log(5)
    return np.column_stack([1. - good, entropy, p[:, 2:].sum(axis=1), p[:, 3:].sum(axis=1), p.argmax(axis=1) >= 2])


@dataclass(frozen=True)
class ValueModel:
    feature_mean: tuple[float, ...]
    feature_scale: tuple[float, ...]
    coefficients: tuple[float, ...]
    intercept: float

    def predict(self, features):
        return ((features - self.feature_mean) / self.feature_scale) @ self.coefficients + self.intercept


def fit_value(features, gross_gain):
    features = np.asarray(features, dtype=float)
    target = np.asarray(gross_gain, dtype=float)
    if (features.ndim != 2 or not len(features) or features.shape[1] != len(FEATURE_NAMES)
            or target.shape != (len(features),) or not np.isfinite(features).all() or not np.isfinite(target).all()):
        raise ValueError('Finite training predictors and gross gains are required')
    mean = features.mean(axis=0)
    scale = features.std(axis=0)
    scale[scale == 0.] = 1.
    standardized = (features - mean) / scale
    intercept = float(target.mean())
    beta = np.linalg.solve(standardized.T @ standardized + np.eye(features.shape[1]),
                           standardized.T @ (target - intercept))
    return ValueModel(tuple(mean.tolist()), tuple(scale.tolist()), tuple(beta.tolist()), intercept)


def budget_routes(features, model: ValueModel):
    """Apply frozen ranks using only an initial-view feature matrix."""
    count = len(features)
    scores = model.predict(features)
    order = {'voi': np.argsort(-scores, kind='stable')}
    order.update({name: np.argsort(-features[:, column], kind='stable')
                  for column, name in enumerate(COMPARATORS[:-1])})
    order['random'] = np.random.default_rng(RANDOM_SEED).permutation(count)
    order['voi_positive'] = order['voi'][scores[order['voi']] > ACQUISITION_COST]

    def route(method, index, ranked):
        budget = BUDGETS[index]
        slots = capacity(budget, count)
        selected = np.zeros(count, dtype=bool)
        selected[ranked[:slots]] = True
        return {'id': f'{method}:{index}', 'method': method, 'budget': budget,
                'slots': slots, 'selected': selected.tolist()}

    all_eyes = np.arange(count)
    return [route('first_view', 0, all_eyes),
            *(route(name, index, ranked) for index in range(1, len(BUDGETS) - 1) for name, ranked in order.items()),
            route('both_views', len(BUDGETS) - 1, all_eyes)]
