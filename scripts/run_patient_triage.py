"""Summarize frozen image recommendations by patient and check patient capacity."""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from scipy.stats import binom

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qsart.triage_v2 import Action, capacity


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = 'patient_triage_v1'
PROTOCOL = 'docs/research/PATIENT_TRIAGE_V1_PROTOCOL.md'
SOURCE_PATHS = (
    PROTOCOL, 'scripts/run_patient_triage.py', 'scripts/audit_patient_triage.py', 'qsart/triage_v2.py',
)
INPUTS = {
    'image_report': ('reports/referability_gate_v1/evaluation.json',
                     '53de650c64c6313cf612a355aea51dc99146ef344a4affdc5a0bed410bc33cfe'),
    'image_audit': ('reports/referability_gate_v1/independent_audit.json',
                    '4e75c7e556b684f63929d2445f3e8d728d0911c9ceb7bab26b5118291dbf3b2d'),
    'action_witness': ('artifacts/referability_gate_v1/sets_and_actions.json',
                       '6d416b0e6275b4b495c5213add27c13e550db7ff31a6023682fc96e33833f282'),
    'ordinal_witness': ('artifacts/ordinal_study_v1/scores_and_sets.json',
                        '0ef95338994b3a587dd38b3034f5c059f063a77901b5af3f17f39292d5f40b4c'),
    'manifest': ('data/processed/deepdrid/development_v1.csv',
                 '45e9f2d34ed54a12056b57a808d03cfaf29ccc970d148e9eb107b79cf5ac3e0d'),
}
ALPHAS = (.10, .05)
METHODS = ('ordinal_projected', 'lac_projected', 'binary_lac')
BUDGETS = ((.30, .20), (.60, .40), (.90, .40), (1., .40))
AUTO, RECAPTURE, REVIEW = (Action.AUTO_NONREFERABLE, Action.RECAPTURE, Action.SPECIALIST_REVIEW)
REFERABLE = 2
TAIL = .05
PLANNING_RISKS = (.05, .02, .01)


def sha256(value):
    return hashlib.sha256(value).hexdigest()


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT)


def bind_source(commit):
    if git('rev-parse', 'HEAD').decode().strip() != commit or git('status', '--porcelain'):
        raise ValueError('Execution requires the declared clean source commit')
    hashes = {}
    for path in SOURCE_PATHS:
        value = (ROOT / path).read_bytes()
        if value != git('show', f'{commit}:{path}'):
            raise ValueError(f'Execution source differs from the declared commit: {path}')
        hashes[path] = sha256(value)
    return hashes


def load_inputs():
    paths = {name: ROOT / path for name, (path, _) in INPUTS.items()}
    hashes = {name: sha256(path.read_bytes()) for name, path in paths.items()}
    if hashes != {name: digest for name, (_, digest) in INPUTS.items()}:
        raise ValueError('Frozen report, witness or manifest bytes differ')
    report, audit, actions, ordinal = (json.loads(paths[name].read_bytes())
                                       for name in ('image_report', 'image_audit', 'action_witness', 'ordinal_witness'))
    if (
        report['experiment_id'] != audit['experiment_id'] or audit['experiment_id'] != actions['experiment_id']
        or report['experiment_id'] != 'referability_gate_v1' or report['partition'] != 'validation'
        or report['policy']['referable_grade'] != REFERABLE
        or audit['audit_status'] != 'passed' or audit['evaluation_sha256'] != hashes['image_report']
        or report['private_output_sha256'] != audit['private_output_sha256']
        or audit['private_output_sha256'] != hashes['action_witness']
        or report['input_sha256']['ordinal_cache'] != hashes['ordinal_witness']
        or report['input_sha256']['manifest'] != hashes['manifest']
        or tuple(map(tuple, report['budget_pairs'])) != BUDGETS
    ):
        raise ValueError('Prior report, audit and private input bindings do not agree')
    return paths['manifest'], report, audit, actions, ordinal, hashes


def selection_patients(path, ordinal):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    if {row['split'] for row in rows} != {'train', 'calibration', 'validation'}:
        raise ValueError('Expected the frozen development manifest')
    selected = [row for row in rows if row['split'] == 'validation']
    grades = [int(row['dr_grade']) for row in selected]
    subjects = [row['subject_id'] for row in selected]
    if (
        len(selected) != 168 or len({row['image_id'] for row in selected}) != len(selected)
        or any(row['annotation_status'] != 'verified' for row in selected)
        or any(grade not in range(5) for grade in grades)
        or grades != ordinal['selection_grades'] or subjects != ordinal['selection_subject_ids']
        or set(subjects) & {row['subject_id'] for row in rows if row['split'] != 'validation'}
    ):
        raise ValueError('Selection labels, patient roles or row order differ')
    indices = defaultdict(list)
    for index, subject in enumerate(subjects):
        indices[subject].append(index)
    if len(indices) != 42:
        raise ValueError('Expected 42 patient groups')
    for group in indices.values():
        eyes = [selected[index]['eye'] for index in group]
        if Counter(eyes) != {'left': 2, 'right': 2}:
            raise ValueError('Expected two views of each eye per patient')
        for eye in ('left', 'right'):
            if len({grades[index] for index in group if selected[index]['eye'] == eye}) != 1:
                raise ValueError('Eye grades differ across views')
        worst = max(grades[index] for index in group)
        if {int(selected[index]['original_patient_DR_Level']) for index in group} != {worst}:
            raise ValueError('The worst eye and original patient grades differ')
    return grades, list(indices.values())


def patient_rollup(actions, grades, patient_indices):
    if len(actions) != len(grades):
        raise ValueError('Expected one action for every ordered image')
    actions = [Action(value) for value in actions]
    image_counts = Counter(actions)
    patient_counts = Counter()
    any_auto = referable_exposed = patients_with_missed_image = fully_auto_misses = 0
    for indices in patient_indices:
        patient_actions = {actions[index] for index in indices}
        patient_action = REVIEW if REVIEW in patient_actions else RECAPTURE if RECAPTURE in patient_actions else AUTO
        patient_counts[patient_action] += 1
        referable = any(grades[index] >= REFERABLE for index in indices)
        any_auto += AUTO in patient_actions
        referable_exposed += AUTO in patient_actions and referable
        patients_with_missed_image += any(actions[index] == AUTO and grades[index] >= REFERABLE for index in indices)
        fully_auto_misses += patient_action == AUTO and referable
    fully_auto = patient_counts[AUTO]
    return {
        'image_actions': {action.value: image_counts[action] for action in Action},
        'missed_referable_images': sum(action == AUTO and grade >= REFERABLE
                                        for action, grade in zip(actions, grades, strict=True)),
        'patient_actions': {action.value: patient_counts[action] for action in Action},
        'patients_with_any_automatic_image': any_auto,
        'patients_with_partial_automatic_images': any_auto - fully_auto,
        'referable_patients_with_any_automatic_image': referable_exposed,
        'patients_with_automatic_referable_image': patients_with_missed_image,
        'fully_automatic_referable_patients': fully_auto_misses,
        'automatic_image_miss_fraction_among_exposed_patients': patients_with_missed_image / any_auto if any_auto else None,
        'full_patient_clearance_miss_fraction': fully_auto_misses / fully_auto if fully_auto else None,
    }


def check_image_counts(rollup, expected):
    counts = rollup['image_actions']
    checks = {'n': sum(counts.values()), 'auto_nonreferable_count': counts[AUTO],
              'recapture_count': counts[RECAPTURE], 'review_count': counts[REVIEW],
              'missed_referable_auto': rollup['missed_referable_images']}
    if any(expected[key] != value for key, value in checks.items()):
        raise ValueError('Action witness does not reproduce the prior image counts')


def patient_capacity(pointwise_counts, review, recapture):
    patients = sum(pointwise_counts.values())
    review_cap, recapture_cap = capacity(review, patients), capacity(recapture, patients)
    required_review, recapture_only = pointwise_counts[REVIEW], pointwise_counts[RECAPTURE]
    min_review = required_review + max(0, recapture_only - recapture_cap)
    shortfall = max(0, min_review - review_cap)
    return {
        'review_capacity': review_cap, 'recapture_capacity': recapture_cap,
        'mandatory_review': required_review, 'recapture_only': recapture_only,
        'minimum_review_after_spill': min_review, 'review_shortfall': shortfall,
        'status': 'overloaded' if shortfall else 'feasible',
        'maximum_full_automatic_patients': pointwise_counts[AUTO] if not shortfall else None,
    }


def planning_count(misses, target):
    if type(misses) is not int or misses < 0 or not 0 < target < 1:
        raise ValueError('Expected a nonnegative miss count and a risk in (0, 1)')
    count = misses + 1
    while binom.cdf(misses, count, target) > TAIL:
        count += 1
    return count


def evaluate_fits(report, witness, grades, patients):
    cases = [(alpha, method) for alpha in ALPHAS for method in METHODS]
    fits = []
    for (alpha, method), prior, saved in zip(cases, report['fits'], witness['fits'], strict=True):
        if any((fit['alpha'], fit['method']) != (alpha, method) for fit in (prior, saved)):
            raise ValueError('The six frozen fit identities differ')
        pointwise = patient_rollup(saved['pointwise_actions'], grades, patients)
        check_image_counts(pointwise, prior['pointwise_metrics'])
        scenarios = []
        for (review, recapture), old, row in zip(BUDGETS, prior['allocations'], saved['allocations'], strict=True):
            if any((item['review_budget'], item['recapture_budget']) != (review, recapture) for item in (old, row)):
                raise ValueError('Saved budget pairs differ')
            caps = patient_capacity(pointwise['patient_actions'], review, recapture)
            if old['status'] == 'overloaded' and old['metrics'] is None and row['actions'] is None:
                observed = None
            elif old['status'] == 'evaluated' and old['metrics'] is not None and row['actions'] is not None:
                observed = patient_rollup(row['actions'], grades, patients)
                check_image_counts(observed, old['metrics'])
                observed['review_excess_over_patient_cap'] = max(
                    0, observed['patient_actions'][REVIEW] - caps['review_capacity'])
                observed['recapture_excess_over_patient_cap'] = max(
                    0, observed['patient_actions'][RECAPTURE] - caps['recapture_capacity'])
            else:
                raise ValueError('Prior image status and availability of action outcomes disagree')
            scenarios.append({
                'review_budget': review, 'recapture_budget': recapture, 'image_status': old['status'],
                'image_allocation_patient_rollup': observed, 'patient_capacity': caps,
            })
        fits.append({'alpha': alpha, 'method': method, 'pointwise': pointwise, 'scenarios': scenarios})
    return fits


def summarize_fits(fits):
    pointwise = [fit['pointwise'] for fit in fits]
    scenarios = [scenario for fit in fits for scenario in fit['scenarios']]
    rollups = [row['image_allocation_patient_rollup'] for row in scenarios if row['image_status'] == 'evaluated']
    return {
        'pointwise_fits': len(pointwise),
        'image_allocations_evaluated': len(rollups), 'image_allocations_overloaded': len(scenarios) - len(rollups),
        'patient_capacity_feasible': sum(row['patient_capacity']['status'] == 'feasible' for row in scenarios),
        'patient_capacity_overloaded': sum(row['patient_capacity']['status'] == 'overloaded' for row in scenarios),
        'image_evaluated_patient_overloaded': sum(row['image_status'] == 'evaluated'
                                                  and row['patient_capacity']['status'] == 'overloaded'
                                                  for row in scenarios),
        'image_rollups_exceeding_patient_caps': sum(bool(row['review_excess_over_patient_cap']
                                                        or row['recapture_excess_over_patient_cap'])
                                                    for row in rollups),
        'pointwise_max_automatic_images': max(row['image_actions'][AUTO] for row in pointwise),
        'pointwise_max_patients_with_any_automatic_image': max(row['patients_with_any_automatic_image']
                                                              for row in pointwise),
        'pointwise_max_fully_automatic_patients': max(row['patient_actions'][AUTO] for row in pointwise),
        'image_allocations_max_automatic_images': max(row['image_actions'][AUTO] for row in rollups),
        'image_allocations_max_patients_with_any_automatic_image': max(row['patients_with_any_automatic_image']
                                                                      for row in rollups),
        'image_allocations_max_fully_automatic_patients': max(row['patient_actions'][AUTO] for row in rollups),
    }


def run(commit, output):
    output = (ROOT / output).resolve()
    if not output.is_relative_to(ROOT / 'reports') or output == ROOT / 'reports' or output.exists():
        raise ValueError('Use a new experiment directory under reports')
    source_hashes = bind_source(commit)
    manifest, source_report, audit, actions, ordinal, input_hashes = load_inputs()
    grades, patient_indices = selection_patients(manifest, ordinal)
    fits = evaluate_fits(source_report, actions, grades, patient_indices)
    summary = summarize_fits(fits)
    if (summary['image_allocations_evaluated'], summary['image_allocations_overloaded']) != (
        audit['evaluated_allocations'], audit['overloaded_allocations'],
    ):
        raise ValueError('Prior audit and reconstructed scenario counts differ')
    report = {
        'schema_version': 1, 'experiment_id': EXPERIMENT, 'dataset': 'DeepDRiD',
        'partition': 'validation', 'scope': 'development_patient_rollup_and_capacity_feasibility',
        'source_commit': commit, 'source_sha256': source_hashes, 'protocol_sha256': source_hashes[PROTOCOL],
        'input_sha256': input_hashes, 'model_version': source_report['model_version'],
        'population': {'images': len(grades), 'patients': len(patient_indices), 'views_per_eye': 2,
                       'referable_patients': sum(max(grades[index] for index in group) >= REFERABLE
                                                for group in patient_indices),
                       'referable_grade': REFERABLE},
        'fits': fits, 'summary': summary,
        'planning': {'one_sided_confidence': 1. - TAIL, 'counts': [
            {'target_upper_risk': target, 'zero_misses': planning_count(0, target),
             'one_miss': planning_count(1, target)} for target in PLANNING_RISKS]},
        'limitations': [
            'The development selection patients previously informed model choice.',
            'Patient actions summarize frozen recommendations across four photographs.',
            'The patient-capacity calculation uses one slot per patient and assigns no new care actions.',
            'Prospective sample sizes assume independent fully automatic patients under one fixed rule.',
        ],
    }
    content = json.dumps(report, indent=2, allow_nan=False) + '\n'
    output.mkdir(parents=True)
    with (output / 'evaluation.json').open('xb') as stream:
        stream.write(content.encode('utf-8'))
    print(json.dumps(summary))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--output', type=Path, default=Path('reports') / EXPERIMENT)
    args = parser.parse_args()
    run(args.source_commit, args.output)


if __name__ == '__main__':
    main()
