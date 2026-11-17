"""Development-only comparator tuning around the unchanged guarded v2 allocator.

Allocation prices are candidate decision parameters. Outcome accounting always
uses a separate, fixed ledger, so choosing a price cannot rewrite its own loss.
These helpers do not fit models, temperatures, conformal artifacts or guard
thresholds. Cross-fitting them on previously inspected selection data does not
create an independent end-to-end evaluation.
"""
from copy import deepcopy
from hashlib import sha256
from numbers import Integral
from collections.abc import Mapping, Sequence
import math

import numpy as np

from qsart.triage_v2 import (
    METHODS, BudgetOverloaded, PolicyConfig, allocate_rows, screening_metrics,
)


MODES = ('ordinal', 'lac', 'no_cp')
BUDGET_PAIRS = ((.30, .20), (.60, .40), (.90, .40), (1., .40))
DECISION_REVIEW_PRICES = (.05, .10, .25, .50, 1., 2.)
FOLD_PREFIX = 'policy-comparator-v1-20261022'
EVALUATION_LEDGER = {
    'severity_costs': (0., 0., 3., 6., 10.),
    'review_cost': .25,
    'recapture_cost': .20,
}
LP_TIE_LIMITATION = (
    'The unchanged linear-program allocator may have multiple equally optimal '
    'assignments; no new secondary assignment tie-breaking guarantee is added.'
)


def _group_keys(groups):
    values = np.asarray(groups, dtype=object)
    if values.ndim != 1 or not len(values):
        raise ValueError('A nonempty one-dimensional patient-group vector is required')
    keys = []
    kinds = set()
    for value in values:
        if isinstance(value, str) and value.strip():
            kinds.add('string')
        elif isinstance(value, Integral) and not isinstance(value, (bool, np.bool_)):
            kinds.add('integer')
        else:
            raise ValueError('Patient groups must be nonmissing strings or integers')
        keys.append(str(value))
    if len(kinds) != 1:
        raise ValueError('Patient group types must be consistent to avoid identifier collisions')
    return np.asarray(keys)


def patient_folds(groups, n_folds=3):
    """Assign complete patients by stable hash order, independent of row order.

    Subject identifiers are consistently typed strings or integers. With 42
    patients and three folds, each fold contains 14 patients. Labels are absent.
    """
    keys = _group_keys(groups)
    if isinstance(n_folds, (bool, np.bool_)) or not isinstance(n_folds, Integral) or n_folds < 2:
        raise ValueError('n_folds must be an integer of at least two')
    unique = set(keys.tolist())
    if len(unique) < n_folds:
        raise ValueError('At least n_folds distinct patients are required')
    ordered = sorted(unique, key=lambda key: (sha256(f'{FOLD_PREFIX}:{key}'.encode()).hexdigest(), key))
    lookup = {key: index % n_folds for index, key in enumerate(ordered)}
    return np.asarray([lookup[key] for key in keys], dtype=int)


def _copied_rows(rows):
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)) or not rows:
        raise ValueError('A nonempty sequence of prediction rows is required')
    if any(not isinstance(row, Mapping) for row in rows):
        raise ValueError('Prediction rows must be mappings')
    return deepcopy(list(rows))


def _prediction_set(value):
    if not isinstance(value, (list, tuple)) or any(
            isinstance(label, (bool, np.bool_)) or not isinstance(label, Integral)
            or label not in range(5) for label in value):
        raise ValueError('Explicit conformal sets must contain unique integer grades in 0..4')
    if len(set(value)) != len(value):
        raise ValueError('Explicit conformal sets cannot repeat grades')
    return list(value)


def prepare_mode_rows(rows, mode):
    """Copy cached rows and select an explicit CP ablation without using labels.

    Ordinal and LAC modes require their own available cached sets. Missing data
    never silently becomes no-CP or another conformal method. Empty explicit
    sets remain empty and therefore trigger the existing mandatory-review guard.
    """
    if mode not in MODES:
        raise ValueError(f'Unknown conformal mode: {mode}')
    output = _copied_rows(rows)
    for row in output:
        if mode == 'no_cp':
            row.pop('conformal_set', None)
            row.pop('conformal', None)
            continue
        conformal = row.get('conformal')
        if not isinstance(conformal, Mapping) or conformal.get('status') != 'available':
            raise ValueError(f'{mode} requires available explicit cached conformal output')
        if mode == 'ordinal':
            if conformal.get('method') != 'ordinal_greedy':
                raise ValueError('Ordinal mode requires the deployed ordinal_greedy method')
            selected = _prediction_set(conformal.get('prediction_set'))
            if 'conformal_set' in row and _prediction_set(row['conformal_set']) != selected:
                raise ValueError('Cached ordinal prediction sets disagree')
        else:
            selected = _prediction_set(conformal.get('baseline_lac_set'))
        row['conformal_set'] = selected
        row['conformal'] = dict(conformal)
        row['conformal']['prediction_set'] = list(selected)
        row['conformal']['method'] = 'ordinal_greedy' if mode == 'ordinal' else 'lac'
    return output


def _labels(labels, n):
    grades = np.asarray(labels)
    if grades.shape != (n,) or grades.dtype.kind not in 'iu' or not np.isin(grades, range(5)).all():
        raise ValueError('Integer evaluation grades must have N entries in 0..4')
    return grades


def evaluate_policy(rows, labels, method, review_budget, recapture_budget, decision_review_price=.25):
    """Allocate from scores alone, then measure labels with the fixed ledger.

    An overload has no action vector or outcome loss. Labels are checked for
    validity but are never supplied to the allocator or used to alter routing.
    """
    predictions = _copied_rows(rows)
    grades = _labels(labels, len(predictions))
    if method not in METHODS:
        raise ValueError(f'Unknown policy: {method}')
    if (isinstance(decision_review_price, (bool, np.bool_))
            or not math.isfinite(decision_review_price) or decision_review_price < 0):
        raise ValueError('decision_review_price must be finite and nonnegative')
    decision_config = PolicyConfig(review_budget=review_budget, recapture_budget=recapture_budget,
                                   review_cost=decision_review_price)
    result = {
        'method': method, 'n': len(predictions),
        'review_budget': review_budget, 'recapture_budget': recapture_budget,
        'decision_review_price': decision_review_price,
        'evaluation_ledger': deepcopy(EVALUATION_LEDGER),
    }
    try:
        actions = allocate_rows(predictions, decision_config, method)
    except BudgetOverloaded as exc:
        return {**result, 'status': 'overloaded', 'diagnostics': dict(exc.diagnostics)}
    evaluation_config = PolicyConfig(review_budget=review_budget, recapture_budget=recapture_budget,
                                     **EVALUATION_LEDGER)
    metrics = screening_metrics(grades, [row['grade_probabilities'] for row in predictions],
                                actions, evaluation_config)
    return {**result, 'status': 'evaluated', 'actions': [action.value for action in actions],
            'metrics': metrics}


def _candidate_rank(metrics, index):
    return (metrics['proxy_total_cost_per_image'], metrics['missed_severe_auto'],
            metrics['missed_referable_auto'], metrics['severe_deferred_to_recapture'],
            metrics['review_count'], -metrics['auto_nonreferable_count'], index)


def tune_policy(rows, labels, method, review_budget, recapture_budget):
    """Select one candidate using only this function's tuning population.

    The caller must keep evaluation patients separate. Every candidate, including
    overloads, is retained. Candidate actions are omitted to keep this tuning
    summary compact; evaluate_policy returns them for explicit evaluation.
    """
    candidates, eligible = [], []
    for index, price in enumerate(DECISION_REVIEW_PRICES):
        result = evaluate_policy(rows, labels, method, review_budget, recapture_budget, price)
        result.pop('actions', None)
        candidates.append(result)
        if result['status'] == 'evaluated':
            eligible.append((_candidate_rank(result['metrics'], index), result))
    common = {
        'method': method, 'review_budget': review_budget, 'recapture_budget': recapture_budget,
        'candidates': candidates, 'evaluation_ledger': deepcopy(EVALUATION_LEDGER),
        'tuning_scope': 'Only supplied tuning rows and grades; no evaluation patients or grades.',
        'lp_tie_limitation': LP_TIE_LIMITATION,
    }
    if not eligible:
        return {**common, 'status': 'untunable', 'diagnostics': candidates[0]['diagnostics']}
    _, chosen = min(eligible, key=lambda pair: pair[0])
    return {**common, 'status': 'tuned',
            'chosen_decision_review_price': chosen['decision_review_price'], 'metrics': chosen['metrics']}


def paired_patient_delta(before, after, groups, seed=20261022, repeats=1000):
    """Descriptive after-minus-before image mean, resampling whole patients.

    Arrays contain already evaluated per-image outcomes under paired policies.
    This conditional bootstrap does not refit thresholds or re-run capacities,
    and is not an interval for future clinical performance.
    """
    keys = _group_keys(groups)
    before, after = np.asarray(before, dtype=float), np.asarray(after, dtype=float)
    if before.shape != keys.shape or after.shape != keys.shape:
        raise ValueError('Paired outcome arrays must match the patient-group vector')
    if not np.isfinite(before).all() or not np.isfinite(after).all():
        raise ValueError('Paired outcomes must be finite')
    if isinstance(repeats, (bool, np.bool_)) or not isinstance(repeats, Integral) or repeats < 1:
        raise ValueError('repeats must be a positive integer')
    unique = sorted(set(keys.tolist()))
    delta = after - before
    sums = np.asarray([delta[keys == key].sum() for key in unique])
    counts = np.asarray([(keys == key).sum() for key in unique])
    draws = np.random.default_rng(seed).integers(0, len(unique), size=(repeats, len(unique)))
    samples = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    bounds = np.quantile(samples, [.025, .975])
    return {
        'difference_after_minus_before': float(delta.mean()),
        'ci95': bounds.tolist(), 'patients': len(unique), 'images': len(keys),
        'repeats': int(repeats), 'seed': int(seed),
        'scope': 'Descriptive conditional interval; fixed policies, patient resampling, no capacity rerun.',
    }
