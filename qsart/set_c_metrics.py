"""Held-out Set-C outcomes with four-view patient accounting."""

from numbers import Integral

import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score

from qsart.evaluation import grade_metrics
from qsart.triage_v2 import Action, PolicyConfig
from qsart.uncertainty.conformal import coverage_metrics


CONFIG = PolicyConfig()
BOOTSTRAP_SEED = 20261116
BOOTSTRAP_DRAWS = 2000
ACTION_ORDER = (Action.AUTO_NONREFERABLE, Action.RECAPTURE, Action.SPECIALIST_REVIEW)
ACTION_INDEX = {action.value: index for index, action in enumerate(ACTION_ORDER)}
PATIENT_ENDPOINTS = (
    'patient_weighted_missed_referral_per_patient',
    'patient_proxy_per_patient',
    'fully_automatic_patient_fraction',
    'fully_automatic_referable_patients_per_patient',
    'fully_automatic_severe_patients_per_patient',
)


def _discrete(values, domain):
    values = np.asarray(values, dtype=object)
    if values.ndim != 1 or not len(values) or any(
        not isinstance(value, Integral) or isinstance(value, (bool, np.bool_)) or value not in domain
        for value in values
    ):
        raise ValueError('Expected a nonempty vector of integer labels in the declared domain')
    return values.astype(int)


def _patient_layout(grades, groups):
    grades = _discrete(grades, range(5))
    groups = np.asarray(groups, dtype=object)
    if groups.shape != grades.shape or not (
        all(isinstance(value, str) and value.strip() for value in groups)
        or all(isinstance(value, Integral) and not isinstance(value, (bool, np.bool_)) for value in groups)
    ):
        raise ValueError('Expected one consistently typed, nonmissing patient key per image')
    keys = groups.astype(str)
    _, counts = np.unique(keys, return_counts=True)
    if np.any(counts != 4):
        raise ValueError('Set-C patient outcomes require exactly four views per patient')
    return grades, np.argsort(keys, kind='stable').reshape(-1, 4)


def _action_codes(actions, n_images):
    values = np.asarray(actions, dtype=object)
    if values.shape != (n_images,):
        raise ValueError('Expected one action per image')
    return np.asarray([ACTION_INDEX[Action(value).value] for value in values])


def _patient_endpoints(worst_grades, patient_codes):
    full_auto = patient_codes == 0
    recapture = patient_codes == 1
    review = patient_codes == 2
    referable = worst_grades >= CONFIG.referable_grade
    severe = worst_grades >= CONFIG.severe_grade
    weights = np.asarray(CONFIG.severity_costs)[worst_grades]
    weighted_miss = weights * (full_auto & referable)
    proxy = weighted_miss + weights * (recapture & severe) + CONFIG.review_cost * review + CONFIG.recapture_cost * recapture
    return np.column_stack((weighted_miss, proxy, full_auto, full_auto & referable, full_auto & severe))


def _action_counts(codes):
    return {action.value: int((codes == index).sum()) for index, action in enumerate(ACTION_ORDER)}


def summarize_actions(grades, groups, actions):
    """Roll up authorized evaluation actions, with review taking precedence."""
    grades, views = _patient_layout(grades, groups)
    codes = _action_codes(actions, len(grades))
    patient_codes = codes[views].max(axis=1)
    worst_grades = grades[views].max(axis=1)
    auto_image = codes == 0
    any_auto = auto_image[views].any(axis=1)
    full_auto = patient_codes == 0
    referable = grades >= CONFIG.referable_grade
    severe = grades >= CONFIG.severe_grade
    patient_referable = worst_grades >= CONFIG.referable_grade
    patient_severe = worst_grades >= CONFIG.severe_grade
    full_misses = int((full_auto & patient_referable).sum())
    full_count = int(full_auto.sum())
    endpoints = _patient_endpoints(worst_grades, patient_codes)
    return {
        'n_images': len(grades),
        'n_patients': len(views),
        'image_actions': _action_counts(codes),
        'patient_actions': _action_counts(patient_codes),
        'missed_referable_images': int((auto_image & referable).sum()),
        'missed_severe_images': int((auto_image & severe).sum()),
        'severe_recapture_patients': int(((patient_codes == 1) & patient_severe).sum()),
        'fully_automatic_referable_patients': full_misses,
        'fully_automatic_severe_patients': int((full_auto & patient_severe).sum()),
        'patients_with_any_automatic_image': int(any_auto.sum()),
        'patients_with_partial_automatic_images': int(any_auto.sum()) - full_count,
        'referable_patients_with_any_automatic_image': int((any_auto & patient_referable).sum()),
        'patients_with_automatic_referable_image': int((auto_image & referable)[views].any(axis=1).sum()),
        'full_patient_clearance_miss_fraction': full_misses / full_count if full_count else None,
        **dict(zip(PATIENT_ENDPOINTS, endpoints.mean(axis=0).tolist(), strict=True)),
    }


def _workload(codes):
    return int((codes == 2).sum()), int((codes == 1).sum())


def paired_comparison(grades, groups, candidate, reference):
    """Paired descriptive intervals conditional on the observed matched workloads."""
    grades, views = _patient_layout(grades, groups)
    candidate = _action_codes(candidate, len(grades))
    reference = _action_codes(reference, len(grades))
    candidate_patients = candidate[views].max(axis=1)
    reference_patients = reference[views].max(axis=1)
    matched_image = _workload(candidate) == _workload(reference)
    matched_patient = _workload(candidate_patients) == _workload(reference_patients)
    intervals = None
    if matched_image and matched_patient:
        worst_grades = grades[views].max(axis=1)
        differences = (_patient_endpoints(worst_grades, candidate_patients)
                       - _patient_endpoints(worst_grades, reference_patients))
        n = len(views)
        indices = np.random.default_rng(BOOTSTRAP_SEED).integers(0, n, size=(BOOTSTRAP_DRAWS, n))
        bounds = np.quantile(differences[indices].mean(axis=1), [.025, .975], axis=0)
        intervals = {
            name: {'difference_candidate_minus_reference': float(difference),
                   'ci95': interval.tolist()}
            for name, difference, interval in zip(PATIENT_ENDPOINTS, differences.mean(axis=0), bounds.T, strict=True)
        }
    return {
        'matched_image_workload': matched_image,
        'matched_patient_workload': matched_patient,
        'identical_image_actions': bool(np.array_equal(candidate, reference)),
        'intervals': intervals,
    }


def _probability_vector(values, shape):
    probabilities = np.asarray(values, dtype=float)
    if (probabilities.shape != shape or not np.isfinite(probabilities).all()
            or (probabilities < 0).any() or (probabilities > 1).any()):
        raise ValueError('Expected finite probabilities in the declared shape and range')
    if probabilities.ndim == 2 and not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6, rtol=0):
        raise ValueError('Each five-grade probability row must sum to one')
    return probabilities


def _calibration(grades, probabilities, grading):
    return {
        'nll': float(-np.log(np.clip(probabilities[np.arange(len(grades)), grades], 1e-15, 1.0)).mean()),
        **{key: grading[key] for key in ('brier', 'ece', 'reliability_bins')},
    }


def _binary_summary(truth, predicted, probability):
    return {
        'n': len(truth),
        'positive_count': int(truth.sum()),
        'accuracy': float((predicted == truth).mean()),
        'confusion_matrix': confusion_matrix(truth, predicted, labels=[False, True]).tolist(),
        'auroc': float(roc_auc_score(truth, probability)) if np.unique(truth).size == 2 else None,
        'auprc': float(average_precision_score(truth, probability)) if truth.any() else None,
    }


def _check_eyes(images, grades, views):
    eye_labels = {'left': 'l', 'right': 'r', 'l': 'l', 'r': 'r'}
    expected = {('l', 1), ('l', 2), ('r', 1), ('r', 2)}
    for patient_views in views:
        eye_grades = {(eye_labels.get(images[index]['eye']), images[index]['view']): grades[index]
                      for index in patient_views}
        if eye_grades.keys() != expected:
            raise ValueError('Expected four distinct eye and view combinations per patient')
        if any(eye_grades[eye, 1] != eye_grades[eye, 2] for eye in ('l', 'r')):
            raise ValueError('Released DR grade must agree within each eye')


def summarize_predictions(images):
    """Report model predictions against the frozen images and their reference labels."""
    images = list(images)
    if not images or len({row['id'] for row in images}) != len(images):
        raise ValueError('Expected nonempty unique image identities')
    groups = [row['patient'] for row in images]
    grades, views = _patient_layout([row['grade'] for row in images], groups)
    _check_eyes(images, grades, views)
    n = len(grades)
    production = _probability_vector([row['p'] for row in images], (n, 5))
    raw = _probability_vector([row['raw'] for row in images], (n, 5))
    good_quality = _probability_vector([row['good_quality'] for row in images], (n,))
    _probability_vector([row['entropy'] for row in images], (n,))
    quality_labels = _discrete([row['quality_label'] for row in images], range(2)) == 1
    patient_grades = grades[views].max(axis=1)
    predicted_patient_grades = production.argmax(axis=1)[views].max(axis=1)
    referable_patient_ranking = production[:, CONFIG.referable_grade:].sum(axis=1)[views].max(axis=1)
    production_grading = grade_metrics(grades, production)
    raw_grading = grade_metrics(grades, raw)
    conformal = {}
    for method in ('ordinal', 'lac'):
        metrics = coverage_metrics([row[method] for row in images], grades, groups=groups)
        metrics['covered_images'] = round(metrics['marginal_coverage'] * n)
        metrics['covered_patients'] = round(metrics['patient_all_views_coverage'] * len(views))
        conformal[method] = metrics
    return {
        'image_grading': production_grading,
        'patient_grading': {
            'n': len(views),
            'worst_grade_accuracy': float((patient_grades == predicted_patient_grades).mean()),
            'worst_grade_confusion_matrix': confusion_matrix(
                patient_grades, predicted_patient_grades, labels=np.arange(5)).tolist(),
            'referable': _binary_summary(patient_grades >= CONFIG.referable_grade,
                                         predicted_patient_grades >= CONFIG.referable_grade,
                                         referable_patient_ranking),
        },
        'calibration': {
            'production': _calibration(grades, production, production_grading),
            'raw': _calibration(grades, raw, raw_grading),
        },
        'quality': {
            **_binary_summary(quality_labels, good_quality >= CONFIG.minimum_quality, good_quality),
            'good_probability_threshold': CONFIG.minimum_quality,
            'brier': float(np.square(good_quality - quality_labels).mean()),
        },
        'conformal': conformal,
    }
