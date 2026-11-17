"""Independently reconstruct the retrospective same-eye acquisition study."""

from collections import defaultdict
import csv
from decimal import Decimal
from hashlib import file_digest, sha256
from itertools import product
import json
import math
from pathlib import Path
import re
import subprocess

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
STUDY = 'paired_view_v1'
OUT = ROOT / 'reports' / STUDY
WITNESS = ROOT / 'artifacts' / STUDY / 'scores_and_routes.json'
AUDITOR = 'scripts/audit_paired_view_v1.py'
REQUIRED_SOURCES = {AUDITOR, 'docs/research/PAIRED_VIEW_V1_PROTOCOL.md', 'scripts/run_paired_view_v1.py',
                    'qsart/paired_view_inputs.py', 'qsart/paired_view_metrics.py', 'qsart/paired_view_policy.py'}
PATHS = {
    'manifest': 'data/processed/deepdrid/development_v1.csv',
    'dataset_report': 'reports/development_v1/data_summary.json',
    'dr': 'checkpoints/development_v1/selected/dr.pt',
    'quality': 'checkpoints/development_v1/selected/quality.pt',
    'calibration': 'checkpoints/development_v1/selected/calibration.json',
    'model_card': 'checkpoints/development_v1/selected/model_card.json',
    'selection_lock': 'checkpoints/development_v1/selected/selection_lock.json',
}
BUDGETS = (0., .10, .25, .50, .75, 1.)
METHODS = ('voi', 'quality', 'entropy', 'referable_risk', 'random', 'voi_positive')
FEATURE_NAMES = ('poor_quality', 'entropy', 'referable_risk', 'severe_risk', 'predicted_referable')
AUTHOR_COMMIT = '56d8af71319803150d0e7b90533e0918c7bf85ee'
WEIGHTS = (0., 0., 3., 6., 10.)
SECOND_COST = .20
REVIEW_PROXY = .25
SEED = 20261116
DIFFERENCES = ('grade_accuracy', 'referable_misses', 'severe_misses', 'patient_referable_misses',
               'patient_severe_misses', 'patient_weighted_miss_per_patient', 'review_proxy_count', 'ledger_per_eye')


def require(condition, detail):
    if not condition:
        raise ValueError(detail)


def digest(path):
    with path.open('rb') as stream:
        return file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def compare(actual, expected, field):
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and expected.keys() <= actual.keys(), f'Missing fields in {field}')
        for key, value in expected.items():
            compare(actual[key], value, f'{field}.{key}')
    elif isinstance(expected, (list, tuple)):
        require(isinstance(actual, list) and len(actual) == len(expected), f'Length differs in {field}')
        for observed, value in zip(actual, expected, strict=True):
            compare(observed, value, field)
    elif isinstance(expected, float):
        require(type(actual) in (int, float) and math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-10),
                f'Numeric value differs in {field}')
    else:
        require(type(actual) is type(expected) and actual == expected, f'Value differs in {field}')


def verify_binding(report, witness, witness_path):
    source = report['source_commit']
    paths = report['source_sha256']
    require(isinstance(source, str) and re.fullmatch('[0-9a-f]{40}', source)
            and REQUIRED_SOURCES <= paths.keys() and report['partition'] == 'validation'
            and witness['source_commit'] == source and report['experiment_id'] == witness['experiment_id'] == STUDY
            and report['model_inference_performed'] is True, 'Invalid frozen study identity')
    for path, digest_expected in paths.items():
        original = (ROOT / path).resolve()
        require(original.is_relative_to(ROOT), 'Bound source path escapes the repository')
        committed = subprocess.check_output(['git', 'show', f'{source}:{path}'], cwd=ROOT)
        require(committed == original.read_bytes() and sha256(committed).hexdigest() == digest_expected,
                'Frozen source bytes differ')
    require(report['private_witness_sha256'] == digest(witness_path), 'Private paired-view witness digest differs')
    provenance = report['input_provenance']
    expected = {key: digest(ROOT / path) for key, path in PATHS.items()}
    require(provenance['input_sha256'] == expected, 'Study inputs differ from the frozen provenance')
    calibration = read(ROOT / PATHS['calibration'])
    model_card = read(ROOT / PATHS['model_card'])
    data_report = read(ROOT / PATHS['dataset_report'])['development_v1']
    require(all(calibration[task]['checkpoint_sha256'] == expected[task]
                and calibration[task]['manifest_sha256'] == expected['manifest']
                and calibration[task]['split'] == 'calibration' for task in ('dr', 'quality'))
            and data_report['manifest_sha256'] == expected['manifest']
            and all(model_card[field] == provenance[field] for field in ('model_version', 'quality_model_version')),
            'First-pass calibration binding differs')
    compare(provenance, {'source_release_commit': AUTHOR_COMMIT, 'device': 'cpu', 'second_pass_inference': False,
                         'first_view_suffix': 1, 'second_view_suffix': 2,
                         'probability_path': 'calibrated_first_pass_per_physical_photo'}, 'inference provenance')
    return expected


def verify_roster(fit_rows, selection_rows):
    with (ROOT / PATHS['manifest']).open(encoding='utf-8-sig', newline='') as stream:
        source = list(csv.DictReader(stream))
    by_split = defaultdict(dict)
    for row in source:
        eye, identity, split = row['eye'], row['image_id'], row['split']
        match = re.fullmatch(r'(.+)_([lr])([12])', identity)
        require(match is not None and {'l': 'left', 'r': 'right'}[match[2]] == eye
                and split in ('train', 'validation', 'calibration') and row['official_split'] == 'official_train'
                and row['source_commit'] == AUTHOR_COMMIT and row['annotation_status'] == 'verified',
                'Unexpected paired-view source annotation')
        key = (row['subject_id'], eye, match[3])
        require(key not in by_split[split], 'Repeated paired-view source key')
        by_split[split][key] = row
    subsets = {split: {key[0] for key in records} for split, records in by_split.items()}
    require(len(source) == 1184 and len({row['sha256'] for row in source}) == len(source)
            and len(set.union(*subsets.values())) == sum(map(len, subsets.values())),
            'Development photo or patient splits overlap')
    summary = {}
    for split, scored, n in [('train', fit_rows, 212), ('validation', selection_rows, 42), ('calibration', None, 42)]:
        patients = sorted(subsets[split])
        records = by_split[split]
        require(len(patients) == n and len(records) == 4*n, 'Development patient or photo count differs')
        require(set(records) == set(product(patients, ('left', 'right'), ('1', '2'))),
                'Expected complete same-eye pairs in each development split')
        summary[split] = {'split': split, 'photos': 4*n, 'eyes': 2*n, 'patients': n}
        if scored is None:
            continue
        for row, (patient, eye) in zip(scored, product(patients, ('left', 'right')), strict=True):
            one, two = records[(patient, eye, '1')], records[(patient, eye, '2')]
            require(one['dr_grade'] == two['dr_grade'], 'Released grade differs between same-eye views')
            compare(row, {'patient': patient, 'eye': eye, 'grade': int(one['dr_grade']),
                          'first_id': one['image_id'], 'second_id': two['image_id'],
                          'first_sha256': one['sha256'], 'second_sha256': two['sha256']}, 'private pair roster')
            for view in ('first', 'second'):
                scores = row[view]
                require(isinstance(scores, list) and len(scores) == 5 and all(0 <= p <= 1 for p in scores)
                        and math.isclose(sum(scores), 1., rel_tol=1e-5), 'Malformed calibrated view scores')
            require(0 <= row['first_quality'] <= 1 and row['grade'] in range(5), 'Malformed fitted row')
    return summary


def features(row):
    p = row['first']
    entropy = -sum(x * math.log(x) for x in p if x > 0) / math.log(5.)
    return [1 - row['first_quality'], entropy, sum(p[2:]), sum(p[3:]), float(p.index(max(p)) >= 2)]


def eye_outcome(row, acquired):
    p = [(a + b)/2 for a, b in zip(row['first'], row['second'], strict=True)] if acquired else row['first']
    prediction = p.index(max(p))
    grade = row['grade']
    review = prediction >= 2
    cost = WEIGHTS[grade] * (not review) + REVIEW_PROXY * review + SECOND_COST * acquired
    return {'predicted_grade': prediction, 'correct': prediction == grade, 'review': review,
            'referable_miss': grade >= 2 and not review, 'severe_miss': grade >= 3 and not review, 'ledger': cost}


def fit_model(rows):
    design = np.array([features(row) for row in rows], dtype=float)
    targets = np.array([eye_outcome(row, False)['ledger'] - eye_outcome(row, True)['ledger'] + SECOND_COST
                        for row in rows])
    mean, scale, intercept = design.mean(axis=0), design.std(axis=0), float(targets.mean())
    scale[scale == 0.] = 1.
    standardized = (design - mean)/scale
    # An augmented least-squares system independently implements unit ridge penalty.
    coefficients, *_ = np.linalg.lstsq(np.vstack((standardized, np.eye(5))),
                                       np.concatenate((targets - intercept, np.zeros(5))), rcond=None)
    return {'feature_mean': mean.tolist(), 'feature_scale': scale.tolist(), 'coefficients': coefficients.tolist(),
            'intercept': intercept}, {'mean_gross_gain': intercept, 'positive_after_cost': int((targets > .20).sum())}


def replay_rankings(rows, fitting):
    matrix = np.array([features(row) for row in rows], dtype=float)
    scores = (matrix - fitting['feature_mean']) / fitting['feature_scale'] @ fitting['coefficients'] + fitting['intercept']
    indices = {'voi': sorted(range(len(rows)), key=lambda i: (-scores[i], i)),
               'random': np.random.default_rng(SEED).permutation(len(rows)).tolist()}
    for column, name in enumerate(('quality', 'entropy', 'referable_risk')):
        indices[name] = sorted(range(len(rows)), key=lambda i: (-matrix[i, column], i))
    indices['voi_positive'] = [i for i in indices['voi'] if scores[i] > SECOND_COST]

    def entry(method, index, route):
        budget = BUDGETS[index]
        slots = int(Decimal(str(budget)) * len(rows))
        selected = set(route[:slots])
        return {'id': f'{method}:{index}', 'method': method, 'budget': budget, 'slots': slots,
                'selected': [i in selected for i in range(len(rows))]}

    return [entry('first_view', 0, []), *[entry(method, index, indices[method])
            for index, method in product(range(1, 5), METHODS)], entry('both_views', 5, list(range(len(rows))))]


def measure(rows, selected):
    per_eye = [eye_outcome(row, acquire) for row, acquire in zip(rows, selected, strict=True)]
    groups = defaultdict(list)
    for index, row in enumerate(rows):
        groups[row['patient']].append(index)
    worst_grade = [max(rows[i]['grade'] for i in group) for group in groups.values()]
    worst_prediction = [max(per_eye[i]['predicted_grade'] for i in group) for group in groups.values()]
    referable_patient_misses = [grade for grade, predicted in zip(worst_grade, worst_prediction, strict=True)
                               if grade >= 2 and predicted < 2]
    return {'eyes': len(rows), 'patients': len(groups), 'extra_views': sum(selected),
            'grade_accuracy': sum(x['correct'] for x in per_eye)/len(rows),
            'referable_misses': sum(x['referable_miss'] for x in per_eye),
            'severe_misses': sum(x['severe_miss'] for x in per_eye),
            'referable_eyes': sum(row['grade'] >= 2 for row in rows),
            'severe_eyes': sum(row['grade'] >= 3 for row in rows),
            'patient_referable_misses': len(referable_patient_misses),
            'patient_severe_misses': sum(grade >= 3 for grade in referable_patient_misses),
            'referable_patients': sum(grade >= 2 for grade in worst_grade),
            'severe_patients': sum(grade >= 3 for grade in worst_grade),
            'patient_weighted_miss_per_patient': sum(WEIGHTS[grade] for grade in referable_patient_misses)/len(groups),
            'review_proxy_count': sum(x['review'] for x in per_eye),
            'ledger_per_eye': sum(x['ledger'] for x in per_eye)/len(rows)}, per_eye


def bootstrap(rows, candidate, reference, multiplicity):
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[row['patient']].append(i)
    patient_order = [groups[key] for key in sorted(groups)]
    resampled_sizes = multiplicity @ np.array(list(map(len, patient_order)))
    specs = {'ledger_per_eye': 'ledger', 'referable_misses_per_eye': 'referable_miss',
             'severe_misses_per_eye': 'severe_miss', 'grade_accuracy': 'correct'}
    result = {}
    for output, source in specs.items():
        values = np.array([sum(candidate[i][source] - reference[i][source] for i in group)
                           for group in patient_order], dtype=float)
        differences = (multiplicity @ values) / resampled_sizes
        result[output] = {'difference_candidate_minus_reference': float(values.sum()/len(rows)),
                          'ci95': np.quantile(differences, (.025, .975)).tolist()}
    return result


def verify_outcomes(rows, routes, scenarios, comparisons):
    baseline, _ = measure(rows, [False]*len(rows))
    lookup = {}
    for route, scenario in zip(routes, scenarios, strict=True):
        compare(scenario, {key: value for key, value in route.items() if key != 'selected'}, 'scenario')
        outcomes, per_eye = measure(rows, route['selected'])
        compare(scenario['outcomes'], {**outcomes, 'delta_vs_first': {key: outcomes[key] - baseline[key]
                                                                    for key in DIFFERENCES}}, 'outcomes')
        lookup[route['id']] = (outcomes['extra_views'], per_eye)
    n = len({row['patient'] for row in rows})
    draws = np.random.default_rng(SEED).integers(0, n, size=(2000, n))
    multiplicity = np.stack([np.bincount(sample, minlength=n) for sample in draws])
    expected_pairs = [(f'voi:{i}', f'{method}:{i}', BUDGETS[i]) for i, method in product(range(1, 5), METHODS[1:5])]
    for comparison, (candidate, reference, budget) in zip(comparisons, expected_pairs, strict=True):
        count, candidate_values = lookup[candidate]
        reference_count, reference_values = lookup[reference]
        require(count == reference_count, 'Candidate and reference acquisition workloads differ')
        compare(comparison, {'candidate': candidate, 'reference': reference, 'budget': budget, 'extra_views': count,
                             'intervals': bootstrap(rows, candidate_values, reference_values, multiplicity)},
                'matched acquisition comparison')


def audit(report_path, witness_path):
    report, witness = read(report_path), read(witness_path)
    hashes = verify_binding(report, witness, witness_path)
    fit_rows, selection = witness['fit_rows'], witness['selection_rows']
    populations = verify_roster(fit_rows, selection)
    compare(report, {'fit_population': populations['train'], 'selection_population': populations['validation'],
                     'budgets': BUDGETS}, 'study population and budgets')
    compare(report['input_provenance']['partitions'], {
        split: {key: value for key, value in population.items() if key != 'split'}
        | {'scored': split != 'calibration'} for split, population in populations.items()
    }, 'input partition roles')
    model, gain = fit_model(fit_rows)
    compare(witness['model'], model, 'signed gross-benefit model witness')
    compare(report['model'], {**model, 'feature_names': FEATURE_NAMES, 'fit_gain': gain},
            'signed gross-benefit model report')
    routes = replay_rankings(selection, model)
    compare(witness['routes'], routes, 'routing witness')
    verify_outcomes(selection, routes, report['scenarios'], report['comparisons'])
    return {'audit_status': 'passed', 'experiment_id': STUDY, 'source_commit': report['source_commit'],
            'evaluation_sha256': digest(report_path), 'private_witness_sha256': digest(witness_path),
            'auditor_sha256': digest(ROOT / AUDITOR), 'input_sha256': hashes,
            'fit_patients': populations['train']['patients'], 'selection_patients': populations['validation']['patients'],
            'scenarios_verified': len(routes), 'comparisons_verified': len(report['comparisons']),
            'routing_reconstructed_from_first_view': True, 'model_inference_replayed': False}


def main():
    receipt_path = OUT / 'independent_audit.json'
    require(not receipt_path.exists(), 'Preserve the completed audit receipt')
    receipt = audit(OUT / 'evaluation.json', WITNESS)
    with receipt_path.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(receipt, indent=2, allow_nan=False) + '\n')
    print(json.dumps(receipt, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
