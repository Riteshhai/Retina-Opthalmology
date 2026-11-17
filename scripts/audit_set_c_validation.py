"""Independently audit the frozen Set-C score witness and published aggregates."""

from collections import Counter, defaultdict
from decimal import Decimal
import hashlib
from itertools import groupby, product
import json
import math
from pathlib import Path
import re
import subprocess

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = 'set_c_validation_v1'
REPORTS = ROOT / 'reports' / EXPERIMENT
WITNESS = ROOT / 'artifacts' / EXPERIMENT / 'scores_and_actions.json'
READINESS = ROOT / 'reports/deepdrid_online_readiness_v1'
READINESS_WITNESS = ROOT / 'artifacts/deepdrid_online_readiness_v1/witness.json'
SIDECAR = ROOT / 'checkpoints/spec_v1/conformal_r1.json'
CALIBRATION = ROOT / 'checkpoints/development_v1/selected/calibration.json'
ARTIFACTS = {'readiness_report': READINESS / 'evaluation.json', 'readiness_audit': READINESS / 'independent_audit.json',
             'readiness_witness': READINESS_WITNESS, 'dr': CALIBRATION.with_name('dr.pt'),
             'quality': CALIBRATION.with_name('quality.pt'), 'calibration': CALIBRATION,
             'model_card': CALIBRATION.with_name('model_card.json'),
             'selection_lock': CALIBRATION.with_name('selection_lock.json'), 'conformal': SIDECAR}
AUDITOR_SOURCE = 'scripts/audit_set_c_validation.py'
MODES = ('ordinal', 'lac', 'no_cp')
METHODS = ('plain', 'confidence_only', 'entropy_only', 'severity_only',
           'quality_only', 'single_budget', 'naive', 'dual_budget')
BUDGETS = ((.30, .20), (.60, .40), (.90, .40), (1., .40))
AUTO, RECAPTURE, REVIEW = 'AUTO_NONREFERABLE', 'RECAPTURE', 'SPECIALIST_REVIEW'
ACTIONS = (AUTO, RECAPTURE, REVIEW)
WEIGHTS = (0, 0, 3, 6, 10)
ENDPOINTS = ('patient_weighted_missed_referral_per_patient', 'patient_proxy_per_patient',
             'fully_automatic_patient_fraction', 'fully_automatic_referable_patients_per_patient',
             'fully_automatic_severe_patients_per_patient')


def require(condition, detail):
    if not condition:
        raise ValueError(detail)


def read(path):
    return json.loads(path.read_bytes())


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def compare(actual, expected, field):
    """Check every independently computed field, tolerating float rounding only."""
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and expected.keys() <= actual.keys(), f'Missing fields in {field}')
        for key, value in expected.items():
            compare(actual[key], value, f'{field}.{key}')
    elif isinstance(expected, (list, tuple)):
        require(isinstance(actual, list) and len(actual) == len(expected), f'Length differs in {field}')
        for observed, value in zip(actual, expected, strict=True):
            compare(observed, value, field)
    elif isinstance(expected, float):
        require(type(actual) in (int, float) and math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-12),
                f'Value differs in {field}')
    else:
        require(type(actual) is type(expected) and actual == expected, f'Value differs in {field}')


def bind_execution(report, witness):
    commit = report['source_commit']
    require(re.fullmatch('[0-9a-f]{40}', commit) is not None, 'Invalid locked source revision')
    require(witness['source_commit'] == commit and report['experiment_id'] == witness['experiment_id'] == EXPERIMENT
            and report['model_inference_performed'] is True, 'Report and score witness identities differ')
    sources = report['source_sha256']
    require(AUDITOR_SOURCE in sources, 'Independent auditor absent from the execution lock')
    for name, expected in sources.items():
        source = (ROOT / name).resolve()
        require(source.is_relative_to(ROOT), 'Bound source path escapes the repository')
        committed = subprocess.check_output(['git', 'show', f'{commit}:{name}'], cwd=ROOT)
        require(source.read_bytes() == committed and hashlib.sha256(committed).hexdigest() == expected,
                f'Locked source bytes differ: {name}')
    require(report['private_witness_sha256'] == digest(WITNESS), 'Private score witness digest differs')
    require(report['budget_pairs'] == list(map(list, BUDGETS)), 'Locked dual-capacity grid differs')


def authenticate_rows(images):
    source, receipt = read(READINESS / 'evaluation.json'), read(READINESS / 'independent_audit.json')
    source_hash, receipt_hash = digest(READINESS / 'evaluation.json'), digest(READINESS / 'independent_audit.json')
    witness_hash = digest(READINESS_WITNESS)
    require(receipt['audit_status'] == 'passed' and receipt['evaluation_sha256'] == source_hash
            and source['witness_sha256'] == receipt['private_witness_sha256'] == witness_hash
            and not source['model_inference_performed'] and not receipt['model_inference_performed'],
            'Readiness receipt, report and witness disagree')
    roster = read(READINESS_WITNESS)
    expected = {row['image_id']: row for row in roster['online'] if row['eligible']}
    require(len(images) == len(expected) == 396 and [row['id'] for row in images] == list(expected),
            'Scored rows differ from the frozen eligible cohort')
    columns = {'patient': 'patient_id', 'eye': 'eye', 'view': 'view', 'grade': 'dr_grade',
               'quality_label': 'overall_quality', 'sha256': 'sha256'}
    groups = defaultdict(list)
    for i, row in enumerate(images):
        original = expected[row['id']]
        compare(row, {key: original[source] for key, source in columns.items()}, 'source roster')
        groups[row['patient']].append(i)
    require(len(groups) == 99, 'Scored patient count differs from the frozen cohort')
    for indices in groups.values():
        require({(images[i]['eye'], images[i]['view']) for i in indices} == set(product('lr', (1, 2)))
                and len(indices) == 4, 'A scored patient has an incomplete eye/view roster')
        for eye in 'lr':
            require(len({images[i]['grade'] for i in indices if images[i]['eye'] == eye}) == 1,
                    'A scored patient has inconsistent grades within an eye')
    return [indices for _, indices in sorted(groups.items())], (source_hash, receipt_hash, witness_hash)


def is_probability(value):
    return type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1


def soften(raw, temperature):
    values = [math.log(max(1e-30, p)) / temperature for p in raw]
    largest = max(values)
    weights = [math.exp(value - largest) for value in values]
    total = sum(weights)
    return [weight / total for weight in weights]


def candidate_scores(p):
    """Independently derive growth order and cumulative mass around the mode."""
    mode = p.index(max(p))
    mass, low, high = p[mode], mode, mode
    ordinal = [0.] * 5
    ordinal[mode] = mass
    while low or high < 4:
        if low and (high == 4 or p[low - 1] > p[high + 1]):
            low -= 1
            new = low
        else:
            high += 1
            new = high
        mass += p[new]
        ordinal[new] = min(1., mass)
    return {'ordinal': ordinal, 'lac': [1. - value for value in p]}


def verify_scores(images, sidecar, base_calibration):
    first = sidecar['models']['first_pass']['0.1']
    thresholds = {'ordinal': first['ordinal_greedy'], 'lac': first['lac']}
    production_temp = base_calibration['dr']['heads']['dr_grade']['temperature']
    independent_temp = sidecar['independent_temperature']['first_pass']
    for row in images:
        for probabilities in (row['raw'], row['p']):
            require(len(probabilities) == 5 and all(map(is_probability, probabilities))
                    and abs(sum(probabilities) - 1) <= 2e-6, 'Invalid five-grade probability vector')
        require(is_probability(row['good_quality']) and is_probability(row['entropy']), 'Invalid quality or entropy')
        require(max(abs(a - b) for a, b in zip(row['p'], soften(row['raw'], production_temp), strict=True)) <= 5e-7,
                'Production DR temperature replay differs')
        entropy = -sum(p * math.log(max(p, 1e-12)) for p in row['p']) / math.log(5)
        require(abs(row['entropy'] - entropy) <= 5e-7, 'Normalized entropy replay differs')
        independent = soften(row['raw'], independent_temp)
        for mode, scores in candidate_scores(independent).items():
            calibration = thresholds[mode]
            qhat = math.inf if calibration['qhat'] is None else calibration['qhat']
            included = [grade for grade, score in enumerate(scores) if score <= qhat]
            if not included and calibration['empty_policy'] == 'argmax':
                included = [independent.index(max(independent))]
            require(row[mode] == included, f'{mode} conformal membership differs')


def guard(row, mode):
    p, q, uncertainty = row['p'], row['good_quality'], row['entropy']
    grades = row[mode] if mode != 'no_cp' else None
    if (p.index(max(p)) >= 2 or sum(p[2:]) >= .50 or sum(p[3:]) >= .25
            or (grades is not None and (not grades or max(grades) >= 2)) or (q >= .50 and uncertainty >= .75)):
        return REVIEW
    return RECAPTURE if q < .50 else AUTO


def counts(actions):
    population = Counter(actions)
    return {action: population[action] for action in ACTIONS}


def capacity(fraction, population):
    return int(Decimal(str(fraction)) * population)


def get_diagnostics(routes, budget):
    n = len(routes)
    review, recapture = (capacity(rate, n) for rate in budget)
    forced, flexible = routes.count(REVIEW), routes.count(RECAPTURE)
    total = forced + flexible
    return {'n': n, 'needed_review': forced, 'needed_recapture': flexible, 'needed_recapture_or_review': flexible,
            'needed_recapture_after_available_review': max(0, flexible - max(0, review - forced)),
            'needed_total': total, 'review_capacity': review, 'recapture_capacity': recapture,
            'total_capacity': review + recapture, 'mandatory_review_shortfall': max(0, forced - review),
            'review_shortfall_after_recapture': max(0, forced + max(0, flexible - recapture) - review),
            'total_shortfall': max(0, total - review - recapture), 'eligible_auto_nonreferable': n - total,
            'unsafe_clearance_permitted': False}


def patient_action(actions):
    return max(actions, key=ACTIONS.index)


def patient_capacities(routes, groups, budget):
    patient_routes = [patient_action(routes[i] for i in group) for group in groups]
    review, recapture = (capacity(rate, len(groups)) for rate in budget)
    forced, flexible = patient_routes.count(REVIEW), patient_routes.count(RECAPTURE)
    minimum_reviews = forced + max(0, flexible - recapture)
    shortfall = max(0, minimum_reviews - review)
    return {'review_capacity': review, 'recapture_capacity': recapture, 'mandatory_review': forced,
            'recapture_only': flexible, 'minimum_review_after_spill': minimum_reviews, 'review_shortfall': shortfall,
            'status': 'overloaded' if shortfall else 'feasible',
            'maximum_full_automatic_patients': None if shortfall else patient_routes.count(AUTO)}


def costs(row, method):
    p, q, u = row['p'], row['good_quality'], row['entropy']
    disease = sum(weight * probability for weight, probability in zip(WEIGHTS, p, strict=True))
    clearance = {'plain': 0., 'confidence_only': 1. - max(p), 'entropy_only': u, 'severity_only': disease,
                 'quality_only': 1. - q, 'naive': 1. - q + u}.get(method, disease + 1. - q + .50 * u)
    return {AUTO: clearance, RECAPTURE: .20 + disease + .50 * q, REVIEW: .25}


def verify_assignment(actions, routes, images, method, diagnostics):
    allowed = {REVIEW: {REVIEW}, RECAPTURE: {RECAPTURE, REVIEW},
               AUTO: {AUTO} if method == 'plain' else {AUTO, REVIEW}}
    require(len(actions) == len(images) and all(action in allowed[route]
                for route, action in zip(routes, actions, strict=True)), 'Saved assignment violates a guard')
    review, recapture = (diagnostics[f'{kind}_capacity'] for kind in ('review', 'recapture'))
    require(actions.count(REVIEW) <= review and actions.count(RECAPTURE) <= recapture,
            'Saved assignment exceeds a capacity')
    # Force the cheapest necessary recapture spills. Then sort the remaining optional review savings.
    individual = [costs(row, method) for row in images]
    poor_savings = sorted((cost[RECAPTURE] - cost[REVIEW]
                           for route, cost in zip(routes, individual, strict=True) if route == RECAPTURE), reverse=True)
    auto_savings = [cost[AUTO] - cost[REVIEW]
                    for route, cost in zip(routes, individual, strict=True) if route == AUTO and method != 'plain']
    spill = max(0, len(poor_savings) - recapture)
    optional = sorted(poor_savings[spill:] + auto_savings, reverse=True)
    slots = review - routes.count(REVIEW) - spill
    baseline = sum(cost[route] for route, cost in zip(routes, individual, strict=True))
    optimum = baseline - sum(poor_savings[:spill]) - sum(max(0., gain) for gain in optional[:slots])
    actual = sum(cost[action] for action, cost in zip(actions, individual, strict=True))
    error = abs(actual - optimum)
    require(error <= 2e-6, 'Saved assignment cost differs from the independent optimum')
    return error


def endpoints(grade, action):
    weighted_miss = WEIGHTS[grade] if action == AUTO else 0
    proxy = weighted_miss + ({AUTO: 0., RECAPTURE: .20, REVIEW: .25}[action])
    if action == RECAPTURE and grade >= 3:
        proxy += WEIGHTS[grade]
    return weighted_miss, proxy, int(action == AUTO), int(action == AUTO and grade >= 2), int(action == AUTO and grade >= 3)


def measure(actions, grades, groups):
    patient, rows = [], []
    any_auto = partial = exposed = auto_miss = severe_recapture = 0
    for group in groups:
        actions_in_group = [actions[i] for i in group]
        disposition = patient_action(actions_in_group)
        grade = max(grades[i] for i in group)
        patient.append(disposition)
        rows.append(endpoints(grade, disposition))
        any_auto += AUTO in actions_in_group
        partial += AUTO in actions_in_group and disposition != AUTO
        exposed += AUTO in actions_in_group and grade >= 2
        auto_miss += any(actions[i] == AUTO and grades[i] >= 2 for i in group)
        severe_recapture += disposition == RECAPTURE and grade >= 3
    n = len(groups)
    total = [sum(column) for column in zip(*rows, strict=True)]
    image_misses = Counter(grade for grade, action in zip(grades, actions, strict=True) if action == AUTO)
    result = {'n_images': len(grades), 'n_patients': n, 'image_actions': counts(actions),
              'patient_actions': counts(patient), 'missed_referable_images': sum(image_misses[g] for g in range(2, 5)),
              'missed_severe_images': image_misses[3] + image_misses[4], 'severe_recapture_patients': severe_recapture,
              'fully_automatic_referable_patients': total[3], 'fully_automatic_severe_patients': total[4],
              'patients_with_any_automatic_image': any_auto, 'patients_with_partial_automatic_images': partial,
              'referable_patients_with_any_automatic_image': exposed, 'patients_with_automatic_referable_image': auto_miss,
              'full_patient_clearance_miss_fraction': total[3] / total[2] if total[2] else None,
              **{name: value / n for name, value in zip(ENDPOINTS, total, strict=True)}}
    return result, np.asarray(rows)


def compare_pair(candidate, reference, weights):
    before, after = reference['metrics'], candidate['metrics']
    matching = {}
    for unit in ('image', 'patient'):
        key = f'{unit}_actions'
        matching[f'matched_{unit}_workload'] = all(after[key][action] == before[key][action]
                                                   for action in (REVIEW, RECAPTURE))
    intervals = None
    if all(matching.values()):
        differences = candidate['endpoints'] - reference['endpoints']
        intervals = {}
        for name, values in zip(ENDPOINTS, differences.T, strict=True):
            replicate = weights @ values / len(values)
            intervals[name] = {'difference_candidate_minus_reference': math.fsum(values) / len(values),
                               'ci95': np.quantile(replicate, [.025, .975]).tolist()}
    return {**matching, 'identical_image_actions': candidate['actions'] == reference['actions'], 'intervals': intervals}


def matrix(truth, predicted, classes):
    pairs = Counter(zip(truth, predicted, strict=True))
    return [[pairs[row, column] for column in range(classes)] for row in range(classes)]


def binary_ranking(truth, risk):
    positives, negatives = sum(truth), len(truth) - sum(truth)
    ranked = sorted(zip(risk, truth, strict=True), reverse=True)
    tp = fp = 0
    area = concordant = 0.
    for _, tied in groupby(ranked, key=lambda pair: pair[0]):
        labels = [label for _, label in tied]
        new_tp = sum(labels)
        new_fp = len(labels) - new_tp
        concordant += tp * new_fp + .5 * new_tp * new_fp
        tp += new_tp
        fp += new_fp
        area += new_tp * tp / (tp + fp)
    return (concordant / (positives * negatives) if positives and negatives else None,
            area / positives if positives else None)


def binary(truth, predicted, risk):
    roc, pr = binary_ranking(truth, risk)
    table = matrix(truth, predicted, 2)
    return {'n': len(truth), 'positive_count': sum(truth), 'accuracy': (table[0][0] + table[1][1]) / len(truth),
            'confusion_matrix': table, 'auroc': roc, 'auprc': pr}


def reliability(grades, probabilities):
    bins = [[] for _ in range(10)]
    squared, likelihood = [], []
    for label, scores in zip(grades, probabilities, strict=True):
        major = max(scores)
        bin_no = next(i for i in range(10) if major < (i + 1) / 10 or i == 9)
        bins[bin_no].append((scores.index(major) == label, major))
        likelihood.append(-math.log(max(scores[label], 1e-15)))
        squared.append(sum((probability - (grade == label)) ** 2 for grade, probability in enumerate(scores)))
    result = []
    error = 0.
    for items in bins:
        n = len(items)
        accuracy = sum(correct for correct, _ in items) / n if n else None
        confidence = math.fsum(probability for _, probability in items) / n if n else None
        if n:
            error += n / len(grades) * abs(accuracy - confidence)
        result.append({'count': n, 'accuracy': accuracy, 'confidence': confidence})
    return {'nll': math.fsum(likelihood) / len(grades), 'brier': math.fsum(squared) / len(grades),
            'ece': error, 'reliability_bins': result}


def grade_summary(grades, predicted, risk, calibration):
    table = matrix(grades, predicted, 5)
    n = len(grades)
    rows, columns = [sum(row) for row in table], [sum(col) for col in zip(*table, strict=True)]
    per_class = []
    for g, (support, pred_count) in enumerate(zip(rows, columns, strict=True)):
        tp = table[g][g]
        per_class.append({'grade': g, 'support': support, 'precision': tp / pred_count if pred_count else 0.,
                          'recall': tp / support if support else 0.,
                          'f1': 2 * tp / (pred_count + support) if pred_count + support else 0.})
    weighted = sum(table[i][j] * (i - j) ** 2 for i, j in product(range(5), repeat=2))
    expected = sum(rows[i] * columns[j] * (i - j) ** 2 for i, j in product(range(5), repeat=2)) / n
    roc, pr = binary_ranking([grade >= 2 for grade in grades], risk)
    return {'n': n, 'accuracy': sum(table[i][i] for i in range(5)) / n, 'confusion_matrix': table,
            'per_class': per_class, 'macro_f1': sum(row['f1'] for row in per_class) / 5,
            'quadratic_weighted_kappa': 1. - weighted / expected if expected else None,
            'referable_auroc': roc, 'referable_auprc': pr,
            **{key: value for key, value in calibration.items() if key != 'nll'}}


def coverage(images, groups, grades, mode):
    sets = [row[mode] for row in images]
    hit = [grade in values for grade, values in zip(grades, sets, strict=True)]
    sizes = Counter(map(len, sets))
    hits = sum(hit)
    covered_patients = sum(all(hit[i] for i in group) for group in groups)
    per_class = Counter(grades)
    hits_by_grade = Counter(grade for grade, covered in zip(grades, hit, strict=True) if covered)
    return {'n_images': len(images), 'n_patients': len(groups), 'marginal_coverage': hits / len(images),
            'mean_set_size': sum(size * count for size, count in sizes.items()) / len(images),
            'empty_set_count': sizes[0], 'set_size_counts': {str(i): sizes[i] for i in range(6)},
            'contiguous_nonempty_fraction': sum(bool(s) and max(s) - min(s) + 1 == len(s) for s in sets) / len(images),
            'per_class': {str(g): {'n': per_class[g], 'coverage': hits_by_grade[g] / per_class[g] if per_class[g] else None}
                          for g in range(5)}, 'patient_all_views_coverage': covered_patients / len(groups),
            'covered_images': hits, 'covered_patients': covered_patients}


def prediction_metrics(images, groups, grades):
    probabilities = [row['p'] for row in images]
    predicted = [p.index(max(p)) for p in probabilities]
    risks = [sum(p[2:]) for p in probabilities]
    patient_truth = [max(grades[i] for i in group) for group in groups]
    patient_predicted = [max(predicted[i] for i in group) for group in groups]
    patient_risks = [max(risks[i] for i in group) for group in groups]
    patient_matrix = matrix(patient_truth, patient_predicted, 5)
    is_good = [row['quality_label'] == 1 for row in images]
    quality = [row['good_quality'] for row in images]
    calibration = reliability(grades, probabilities)
    return {'image_grading': grade_summary(grades, predicted, risks, calibration),
            'patient_grading': {'n': len(groups), 'worst_grade_confusion_matrix': patient_matrix,
                                'worst_grade_accuracy': sum(patient_matrix[i][i] for i in range(5)) / len(groups),
                                'referable': binary([g >= 2 for g in patient_truth], [g >= 2 for g in patient_predicted],
                                                     patient_risks)},
            'calibration': {'production': calibration, 'raw': reliability(grades, [row['raw'] for row in images])},
            'quality': {**binary(is_good, [value >= .50 for value in quality], quality), 'good_probability_threshold': .50,
                        'brier': sum((score - truth) ** 2 for score, truth in zip(quality, is_good, strict=True)) / len(images)},
            'conformal': {mode: coverage(images, groups, grades, mode) for mode in ('ordinal', 'lac')}}


def audit(evaluation, private):
    report, witness = read(evaluation), read(private)
    bind_execution(report, witness)
    images = witness['images']
    groups, _ = authenticate_rows(images)
    hashes = {name: digest(path) for name, path in ARTIFACTS.items()}
    provenance = report['input_provenance']
    compare(provenance, {'artifact_sha256': hashes, 'probability_path': 'first_pass', 'device': 'cpu',
                         'conformal_alpha': .10, 'source_release': read(READINESS / 'evaluation.json')['source_release']},
            'inference provenance')
    grades = [row['grade'] for row in images]
    compare(report, {'images': len(images), 'patients': len(groups)}, 'population')
    verify_scores(images, read(SIDECAR), read(CALIBRATION))
    compare(report['prediction_metrics'], prediction_metrics(images, groups, grades), 'prediction metrics')
    all_routes = {mode: [guard(row, mode) for row in images] for mode in MODES}
    compare(witness['pointwise'], all_routes, 'pointwise witness')
    for mode, routes in all_routes.items():
        summary, _ = measure(routes, grades, groups)
        compare(report['pointwise'][mode], summary, f'{mode} pointwise outcomes')

    fixed = [{'id': f'{mode}:{method}:{i}', 'mode': mode, 'method': method, 'budget': list(budget)}
             for mode, (i, budget), method in product(MODES, enumerate(BUDGETS), METHODS)]
    feasible = {}
    max_error = 0.
    for expected, observed, saved in zip(fixed, report['scenarios'], witness['scenarios'], strict=True):
        identity, mode, method, budget = (expected[k] for k in ('id', 'mode', 'method', 'budget'))
        compare(observed, expected, f'{identity} report identity')
        compare(saved, expected, f'{identity} witness identity')
        pointwise = all_routes[mode]
        capacities = patient_capacities(pointwise, groups, budget)
        compare(observed['patient_capacity'], capacities, f'{identity} patient capacities')
        routes = [REVIEW if action == RECAPTURE else action for action in pointwise] if method == 'single_budget' else pointwise
        diagnostics = get_diagnostics(routes, budget)
        if diagnostics['review_shortfall_after_recapture']:
            compare(saved, {'actions': None, 'status': 'overload'}, 'overload witness')
            compare(observed, {'status': 'overload', 'outcomes': None, 'diagnostics': diagnostics}, 'overload report')
            continue
        compare(observed, {'status': 'evaluated', 'diagnostics': None}, 'feasible report')
        require(saved['status'] == 'evaluated' and isinstance(saved['actions'], list), 'Feasible witness has no actions')
        actions = saved['actions']
        max_error = max(max_error, verify_assignment(actions, routes, images, method, diagnostics))
        metrics, endpoints_by_patient = measure(actions, grades, groups)
        compare(observed['outcomes'], metrics, f'{identity} outcomes')
        feasible[identity] = {'metrics': metrics, 'endpoints': endpoints_by_patient, 'actions': actions}

    identities = [{'mode': mode, 'budget': list(budget), 'candidate': f'{mode}:dual_budget:{i}',
                   'reference': f'{mode}:{method}:{i}'}
                  for mode, (i, budget), method in product(MODES, enumerate(BUDGETS), METHODS[:-1])]
    samples = np.random.default_rng(20261116).integers(0, len(groups), size=(2000, len(groups)))
    weights = np.stack([np.bincount(draw, minlength=len(groups)) for draw in samples])
    matched, identical, unavailable, matched_images, matched_patients = 0, 0, 0, 0, 0
    for expected, observed in zip(identities, report['comparisons'], strict=True):
        compare(observed, expected, 'comparison identity')
        reference, candidate = (feasible.get(expected[key]) for key in ('reference', 'candidate'))
        if reference is None or candidate is None:
            require(observed['status'] == 'unavailable' and observed.get('intervals') is None,
                    'Comparison with an overload contains an interval')
            unavailable += 1
            continue
        pair = compare_pair(candidate, reference, weights)
        compare(observed, {'status': 'comparison', **pair}, 'paired comparison outcomes')
        matched_images += pair['matched_image_workload']
        matched_patients += pair['matched_patient_workload']
        matched += pair['intervals'] is not None
        identical += pair['identical_image_actions']
    return {'audit_status': 'passed', 'experiment_id': EXPERIMENT, 'source_commit': report['source_commit'],
            'evaluation_sha256': digest(evaluation), 'private_witness_sha256': digest(private),
            'auditor_sha256': digest(ROOT / AUDITOR_SOURCE), 'input_artifact_sha256': hashes,
            'images': len(images), 'patients': len(groups), 'pointwise_modes_verified': len(all_routes),
            'scenarios_verified': len(fixed), 'evaluated_allocations': len(feasible),
            'overloaded_allocations': len(fixed) - len(feasible), 'comparisons_verified': len(identities),
            'unavailable_comparisons': unavailable, 'matched_image_workload_pairs': matched_images,
            'matched_patient_workload_pairs': matched_patients, 'exact_workload_matched_pairs': matched,
            'identical_image_action_pairs': identical, 'max_abs_allocation_objective_error': max_error,
            'model_inference_replayed': False}


def main():
    receipt_path = REPORTS / 'independent_audit.json'
    require(not receipt_path.exists(), 'Preserve the completed independent audit receipt')
    receipt = audit(REPORTS / 'evaluation.json', WITNESS)
    with receipt_path.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(receipt, indent=2, allow_nan=False) + '\n')
    print(json.dumps(receipt, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
