"""Reconstruct the patient triage diagnostic from the frozen image decisions."""

import argparse
from collections import Counter
import csv
from decimal import Decimal
from fractions import Fraction
import hashlib
from io import StringIO
from itertools import groupby, product
import json
from pathlib import Path
import re
import subprocess


ROOT = Path(__file__).resolve().parents[1]
SLUG = 'patient_triage_v1'
PROTOCOL = 'docs/research/PATIENT_TRIAGE_V1_PROTOCOL.md'
INPUT_PATHS = {
    'manifest': 'data/processed/deepdrid/development_v1.csv',
    'ordinal_witness': 'artifacts/ordinal_study_v1/scores_and_sets.json',
    'action_witness': 'artifacts/referability_gate_v1/sets_and_actions.json',
    'image_report': 'reports/referability_gate_v1/evaluation.json',
    'image_audit': 'reports/referability_gate_v1/independent_audit.json',
}
PRIORITY = ('AUTO_NONREFERABLE', 'RECAPTURE', 'SPECIALIST_REVIEW')
AUTO, RECAPTURE, REVIEW = PRIORITY


def require(test, message):
    if not test:
        raise ValueError(message)


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def verify_source(report):
    commit = report['source_commit']
    require(re.fullmatch('[0-9a-f]{40}', commit), 'Invalid locked source revision')
    sources = {PROTOCOL, 'qsart/triage_v2.py', 'scripts/run_patient_triage.py', 'scripts/audit_patient_triage.py'}
    require(set(report['source_sha256']) == sources, 'Unexpected source inventory')
    for name in sources:
        bound = subprocess.check_output(['git', 'show', f'{commit}:{name}'], cwd=ROOT)
        require(bound == (ROOT / name).read_bytes() and digest(bound) == report['source_sha256'][name],
                'Source bytes differ from the lock')
    require(report['protocol_sha256'] == report['source_sha256'][PROTOCOL], 'Protocol hash differs')


def load_inputs(report):
    payloads = {name: (ROOT / path).read_bytes() for name, path in INPUT_PATHS.items()}
    require(report['input_sha256'] == {name: digest(data) for name, data in payloads.items()},
            'Input receipt hashes differ')
    old, receipt, saved, ordinal = (json.loads(payloads[name]) for name in
                                    ('image_report', 'image_audit', 'action_witness', 'ordinal_witness'))
    require(receipt['audit_status'] == 'passed' and receipt['evaluation_sha256'] == digest(payloads['image_report'])
            and old['private_output_sha256'] == receipt['private_output_sha256'] == digest(payloads['action_witness'])
            and old['input_sha256']['manifest'] == digest(payloads['manifest'])
            and old['input_sha256']['ordinal_cache'] == digest(payloads['ordinal_witness'])
            and old['model_version'] == report['model_version'], 'Previous evidence bindings differ')
    manifest = list(csv.DictReader(StringIO(payloads['manifest'].decode('utf-8-sig'), newline='')))
    rows = [row for row in manifest if row['split'] == 'validation']
    subjects = [row['subject_id'] for row in rows]
    grades = [int(row['dr_grade']) for row in rows]
    require(len(rows) == len({row['image_id'] for row in rows}) == 168
            and all(row['annotation_status'] == 'verified' for row in rows)
            and set(grades).issubset(range(5)) and grades == ordinal['selection_grades']
            and subjects == ordinal['selection_subject_ids']
            and not set(subjects).intersection(row['subject_id'] for row in manifest if row['split'] != 'validation'),
            'Selection rows or role separation differ')
    patient_keys = [(row['subject_id'], row['eye'], int(row['original_patient_DR_Level']), grade)
                    for row, grade in zip(rows, grades, strict=True)]
    groups = sorted(patient_keys)
    referable = 0
    for _, records in groupby(groups, key=lambda value: value[0]):
        records = list(records)
        for eye in ('left', 'right'):
            truth = [record[3] for record in records if record[1] == eye]
            require(len(truth) == 2 and len(set(truth)) == 1, 'Eye view count or grade differs')
        worst = max(record[3] for record in records)
        require(len(records) == 4 and {record[2] for record in records} == {worst}, 'Patient label differs')
        referable += worst >= 2
    require(report['population'] == {'images': 168, 'patients': 42, 'views_per_eye': 2,
                                     'referable_patients': referable, 'referable_grade': 2},
            'Patient cohort counts differ')
    return old, receipt, saved, list(zip(subjects, grades, strict=True))


def independent_rollup(actions, labels, source_metrics):
    image = Counter(actions)
    require(len(actions) == 168 and set(image).issubset(PRIORITY), 'Unexpected image action witness')
    joined = sorted((subject, grade, action) for (subject, grade), action in zip(labels, actions, strict=True))
    patients = []
    image_misses = 0
    for _, rows in groupby(joined, key=lambda row: row[0]):
        pairs = [(grade, action) for _, grade, action in rows]
        automatic = [grade for grade, action in pairs if action == AUTO]
        missed = sum(grade >= 2 for grade in automatic)
        image_misses += missed
        patients.append((max((action for _, action in pairs), key=PRIORITY.index),
                         bool(automatic), any(grade >= 2 for grade, _ in pairs), bool(missed)))
    count = Counter(action for action, *_ in patients)
    exposure = sum(automatic for _, automatic, _, _ in patients)
    missed_patients = sum(miss for _, _, _, miss in patients)
    fully_automatic_misses = sum(action == AUTO and referable for action, _, referable, _ in patients)
    require([source_metrics[key] for key in ('auto_nonreferable_count', 'recapture_count',
                                             'review_count', 'missed_referable_auto')]
            == [image[AUTO], image[RECAPTURE], image[REVIEW], image_misses], 'Prior image counts differ')
    return {
        'image_actions': {action: image[action] for action in PRIORITY},
        'missed_referable_images': image_misses,
        'patient_actions': {action: count[action] for action in PRIORITY},
        'patients_with_any_automatic_image': exposure,
        'patients_with_partial_automatic_images': exposure - count[AUTO],
        'referable_patients_with_any_automatic_image': sum(auto and ref for _, auto, ref, _ in patients),
        'patients_with_automatic_referable_image': missed_patients,
        'fully_automatic_referable_patients': fully_automatic_misses,
        'automatic_image_miss_fraction_among_exposed_patients': missed_patients / exposure if exposure else None,
        'full_patient_clearance_miss_fraction': fully_automatic_misses / count[AUTO] if count[AUTO] else None,
    }


def independent_capacity(pointwise, review, recapture):
    count = pointwise['patient_actions']
    review_cap = int(Fraction(str(review)) * 42)
    recapture_cap = int(Fraction(str(recapture)) * 42)
    possible_reviews = [count[REVIEW] + spill for spill in range(count[RECAPTURE] + 1)
                        if count[RECAPTURE] - spill <= recapture_cap]
    feasible = any(value <= review_cap for value in possible_reviews)
    return {
        'review_capacity': review_cap, 'recapture_capacity': recapture_cap,
        'mandatory_review': count[REVIEW], 'recapture_only': count[RECAPTURE],
        'minimum_review_after_spill': min(possible_reviews),
        'review_shortfall': max(0, min(possible_reviews) - review_cap),
        'status': 'feasible' if feasible else 'overloaded',
        'maximum_full_automatic_patients': count[AUTO] if feasible else None,
    }


def verify_fits(report, old, receipt, witness, labels):
    alphas = (.10, .05)
    methods = ('ordinal_projected', 'lac_projected', 'binary_lac')
    budget_pairs = ((.30, .20), (.60, .40), (.90, .40), (1., .40))
    points, allocations = [], []
    feasible = mixed_status = excess = total = 0
    for (alpha, method), fit, source, saved in zip(product(alphas, methods), report['fits'], old['fits'],
                                                  witness['fits'], strict=True):
        require(all(item['alpha'] == alpha and item['method'] == method for item in (fit, source, saved)),
                'Fit identity mismatch')
        pointwise = independent_rollup(saved['pointwise_actions'], labels, source['pointwise_metrics'])
        require(fit['pointwise'] == pointwise, 'Pointwise patient counts differ')
        points.append(pointwise)
        for (review, recapture), scenario, prior, row in zip(budget_pairs, fit['scenarios'], source['allocations'],
                                                              saved['allocations'], strict=True):
            total += 1
            require(all((item['review_budget'], item['recapture_budget']) == (review, recapture)
                        for item in (scenario, prior, row)), 'Budget identity mismatch')
            caps = independent_capacity(pointwise, review, recapture)
            require(scenario['patient_capacity'] == caps and scenario['image_status'] == prior['status'],
                    'Capacity enumeration or image allocation status differs')
            feasible += caps['status'] == 'feasible'
            observed = scenario['image_allocation_patient_rollup']
            if prior['status'] == 'overloaded':
                require(observed is None and prior['metrics'] is None and row['actions'] is None,
                        'Overload has a reported patient allocation')
                continue
            require(prior['status'] == 'evaluated', 'Unknown image allocation status')
            independent = independent_rollup(row['actions'], labels, prior['metrics'])
            review_excess = max(0, independent['patient_actions'][REVIEW] - caps['review_capacity'])
            recapture_excess = max(0, independent['patient_actions'][RECAPTURE] - caps['recapture_capacity'])
            require(observed == {**independent, 'review_excess_over_patient_cap': review_excess,
                                 'recapture_excess_over_patient_cap': recapture_excess},
                    'Observed patient rollup or cap excess differs')
            allocations.append(independent)
            mixed_status += caps['status'] == 'overloaded'
            excess += bool(review_excess or recapture_excess)

    summary = {
        'pointwise_fits': len(points), 'image_allocations_evaluated': len(allocations),
        'image_allocations_overloaded': total - len(allocations),
        'patient_capacity_feasible': feasible, 'patient_capacity_overloaded': total - feasible,
        'image_evaluated_patient_overloaded': mixed_status,
        'image_rollups_exceeding_patient_caps': excess,
    }
    for label, views in (('pointwise', points), ('image_allocations', allocations)):
        summary[f'{label}_max_automatic_images'] = max(view['image_actions'][AUTO] for view in views)
        summary[f'{label}_max_patients_with_any_automatic_image'] = max(
            view['patients_with_any_automatic_image'] for view in views)
        summary[f'{label}_max_fully_automatic_patients'] = max(view['patient_actions'][AUTO] for view in views)
    require(report['summary'] == summary and len(allocations) == receipt['evaluated_allocations']
            and total == receipt['allocation_scenarios_verified'], 'Scenario totals differ')
    return len(points), len(allocations), total


def verify_planning(planning):
    def lower_tail(count, misses, risk):
        complement = 1 - risk
        mass_at_zero = complement ** count
        return mass_at_zero if not misses else mass_at_zero + count * risk * complement ** (count - 1)

    require(planning['one_sided_confidence'] == .95, 'Unexpected planning confidence')
    count_checks = 0
    for target, row in zip((.05, .02, .01), planning['counts'], strict=True):
        require(row['target_upper_risk'] == target, 'Unexpected planning risk')
        for misses, label in enumerate(('zero_misses', 'one_miss')):
            n = row[label]
            risk, tail = Decimal(str(target)), Decimal('.05')
            require(lower_tail(n, misses, risk) <= tail < lower_tail(n - 1, misses, risk),
                    'Planning count does not minimize the exact binomial criterion')
            count_checks += 1
    return count_checks


def audit(output):
    output = (ROOT / output).resolve()
    require(output.is_relative_to(ROOT / 'reports') and output != ROOT / 'reports', 'Invalid report directory')
    target = output / 'independent_audit.json'
    require(not target.exists(), 'Preserve the existing audit receipt')
    payload = (output / 'evaluation.json').read_bytes()
    report = json.loads(payload)
    require(report['experiment_id'] == SLUG and report['partition'] == 'validation', 'Unexpected report')
    verify_source(report)
    old, receipt, witness, labels = load_inputs(report)
    points, allocations, scenarios = verify_fits(report, old, receipt, witness, labels)
    checks = verify_planning(report['planning'])
    audit_receipt = {
        'audit_status': 'passed', 'experiment_id': SLUG, 'source_commit': report['source_commit'],
        'evaluation_sha256': digest(payload), 'auditor_sha256': digest(Path(__file__).read_bytes()),
        'manifest_images': len(labels), 'manifest_patients': len({subject for subject, _ in labels}),
        'pointwise_rollups_verified': points, 'image_budget_rollups_verified': allocations,
        'patient_capacity_scenarios_enumerated': scenarios, 'binomial_minima_verified': checks,
    }
    with target.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(audit_receipt, stream, indent=2)
        stream.write('\n')
    print(json.dumps(audit_receipt))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('reports') / SLUG)
    audit(parser.parse_args().output)
