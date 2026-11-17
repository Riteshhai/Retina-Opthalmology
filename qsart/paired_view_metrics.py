"""Retrospective endpoints for acquiring a second view of the same eye."""

from collections import defaultdict
from numbers import Integral

import numpy as np

from qsart.evaluation_policy import paired_patient_delta
from qsart.triage_v2 import PolicyConfig


CONFIG = PolicyConfig()
EXTRA_VIEW_COST = .20
BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_SEED = 20261116
EYE_SIDES = {'left': 'l', 'l': 'l', 'right': 'r', 'r': 'r'}
DELTA_METRICS = (
    'grade_accuracy', 'referable_misses', 'severe_misses', 'patient_referable_misses',
    'patient_severe_misses', 'patient_weighted_miss_per_patient', 'review_proxy_count', 'ledger_per_eye',
)
INTERVAL_ENDPOINTS = {
    'ledger_per_eye': 'ledger', 'referable_misses_per_eye': 'referable_miss',
    'severe_misses_per_eye': 'severe_miss', 'grade_accuracy': 'correct',
}


def _eye_grades(rows):
    grades = [row['grade'] for row in rows]
    if not grades or any(not isinstance(grade, Integral) or isinstance(grade, (bool, np.bool_))
                         or grade not in range(5) for grade in grades):
        raise ValueError('Expected a nonempty cohort with integer eye grades from 0 through 4')
    return np.asarray(grades, dtype=int)


def _selection_mask(selected, count):
    mask = np.asarray(selected)
    if mask.shape != (count,) or mask.dtype.kind != 'b':
        raise ValueError('Expected one boolean acquisition decision per eye')
    return mask


def _probabilities(values):
    probabilities = np.asarray(values, dtype=float)
    if (probabilities.shape != (len(values), 5) or not np.isfinite(probabilities).all()
            or (probabilities < 0).any() or (probabilities > 1).any()
            or not np.allclose(probabilities.sum(axis=1), 1., atol=1e-6, rtol=0)):
        raise ValueError('Expected five finite grade probabilities summing to one for each available view')
    return probabilities


def _patient_layout(rows):
    keys = [row['patient'] for row in rows]
    if not (all(isinstance(key, str) and key.strip() for key in keys)
            or all(isinstance(key, Integral) and not isinstance(key, (bool, np.bool_)) for key in keys)):
        raise ValueError('Expected one consistently typed, nonmissing patient key per eye')
    patients = defaultdict(dict)
    for index, (row, key) in enumerate(zip(rows, keys, strict=True)):
        eyes = patients[key]
        side = EYE_SIDES.get(row['eye'])
        if side is None or side in eyes:
            raise ValueError('Expected one distinct left and right eye per patient')
        eyes[side] = index
    if any(len(eyes) != 2 for eyes in patients.values()):
        raise ValueError('Expected one distinct left and right eye per patient')
    return keys, np.asarray([[eyes['l'], eyes['r']] for eyes in patients.values()])


def _evaluate(rows, grades, mask):
    probabilities = _probabilities([row['first'] for row in rows])
    if mask.any():
        second = _probabilities([rows[index]['second'] for index in np.flatnonzero(mask)])
        probabilities[mask] = (probabilities[mask] + second) / 2.
    predicted = probabilities.argmax(axis=1)
    review = predicted >= CONFIG.referable_grade
    referable_miss = (grades >= CONFIG.referable_grade) & ~review
    return {
        'predicted_grade': predicted, 'correct': predicted == grades,
        'referable_miss': referable_miss,
        'severe_miss': (grades >= CONFIG.severe_grade) & referable_miss,
        'review': review,
        'ledger': (np.asarray(CONFIG.severity_costs)[grades] * referable_miss
                   + CONFIG.review_cost * review + EXTRA_VIEW_COST * mask),
    }


def evaluate_policy(rows, selected):
    """Return per-eye outcomes and proxy costs for fixed acquisition decisions."""
    rows = list(rows)
    grades = _eye_grades(rows)
    return _evaluate(rows, grades, _selection_mask(selected, len(rows)))


def _summarize(grades, eyes, mask, outcomes):
    worst_truth = grades[eyes].max(axis=1)
    patient_review = outcomes['review'][eyes].any(axis=1)
    patient_miss = (worst_truth >= CONFIG.referable_grade) & ~patient_review
    return {
        'eyes': len(grades), 'patients': len(eyes), 'extra_views': int(mask.sum()),
        'grade_accuracy': float(outcomes['correct'].mean()),
        'referable_misses': int(outcomes['referable_miss'].sum()),
        'severe_misses': int(outcomes['severe_miss'].sum()),
        'referable_eyes': int((grades >= CONFIG.referable_grade).sum()),
        'severe_eyes': int((grades >= CONFIG.severe_grade).sum()),
        'patient_referable_misses': int(patient_miss.sum()),
        'patient_severe_misses': int(((worst_truth >= CONFIG.severe_grade) & patient_miss).sum()),
        'referable_patients': int((worst_truth >= CONFIG.referable_grade).sum()),
        'severe_patients': int((worst_truth >= CONFIG.severe_grade).sum()),
        'patient_weighted_miss_per_patient': float((np.asarray(CONFIG.severity_costs)[worst_truth] * patient_miss).mean()),
        'review_proxy_count': int(outcomes['review'].sum()),
        'ledger_per_eye': float(outcomes['ledger'].mean()),
    }


def summarize_policy(rows, selected):
    """Aggregate eye and patient outcomes and changes from first-view grading."""
    rows = list(rows)
    grades = _eye_grades(rows)
    _, eyes = _patient_layout(rows)
    mask = _selection_mask(selected, len(rows))
    no_acquisition = np.zeros(len(rows), dtype=bool)
    first = _summarize(grades, eyes, no_acquisition, _evaluate(rows, grades, no_acquisition))
    observed = _summarize(grades, eyes, mask, _evaluate(rows, grades, mask))
    observed['delta_vs_first'] = {metric: observed[metric] - first[metric] for metric in DELTA_METRICS}
    return observed


def paired_interval(rows, candidate, reference):
    """Return descriptive paired intervals at the same realized extra-view count."""
    rows = list(rows)
    grades = _eye_grades(rows)
    patients, _ = _patient_layout(rows)
    candidate_mask = _selection_mask(candidate, len(rows))
    reference_mask = _selection_mask(reference, len(rows))
    if candidate_mask.sum() != reference_mask.sum():
        raise ValueError('Paired intervals require matching realized extra-view counts')
    candidate_outcomes = _evaluate(rows, grades, candidate_mask)
    reference_outcomes = _evaluate(rows, grades, reference_mask)
    intervals = {}
    for metric, key in INTERVAL_ENDPOINTS.items():
        interval = paired_patient_delta(reference_outcomes[key], candidate_outcomes[key], patients,
                                        seed=BOOTSTRAP_SEED, repeats=BOOTSTRAP_DRAWS)
        intervals[metric] = {'difference_candidate_minus_reference': interval['difference_after_minus_before'],
                             'ci95': interval['ci95']}
    return intervals
