"""Evaluate locked Set-C patients with frozen model and routing policies."""

import argparse
from dataclasses import asdict
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
import platform
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qsart.evaluation_policy import BUDGET_PAIRS, MODES
from qsart.set_c_inputs import load_and_score
from qsart.set_c_metrics import paired_comparison, summarize_actions, summarize_predictions
from qsart.triage_v2 import METHODS, BudgetOverloaded, PolicyConfig, allocate, pointwise_action
from scripts.run_patient_triage import patient_capacity


ROOT = Path(__file__).resolve().parents[1]
STUDY = 'set_c_validation_v1'
PRIVATE = ROOT / 'artifacts' / STUDY / 'scores_and_actions.json'
REPORT = ROOT / 'reports' / STUDY / 'evaluation.json'
SOURCE_PATHS = (
    'docs/research/SET_C_VALIDATION_V1_PROTOCOL.md', 'scripts/run_set_c_validation.py',
    'scripts/audit_set_c_validation.py', 'qsart/set_c_inputs.py', 'qsart/set_c_metrics.py',
    'qsart/evaluation_policy.py', 'qsart/triage_v2.py', 'qsart/triage.py', 'scripts/run_patient_triage.py',
    'qsart/inference/engine.py', 'qsart/inference/conformal_sidecar.py',
    'qsart/models/checkpoint.py', 'qsart/training/data.py', 'qsart/preprocessing/fundus.py',
    'qsart/uncertainty/conformal.py', 'requirements.txt',
)


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT)


def bind_source(commit):
    if git('rev-parse', 'HEAD').decode().strip() != commit or git('status', '--porcelain'):
        raise ValueError('Execution requires the declared clean source commit')
    hashes = {}
    for path in SOURCE_PATHS:
        value = (ROOT / path).read_bytes()
        if value != git('show', f'{commit}:{path}'):
            raise ValueError(f'Execution file differs from the declared Git blob: {path}')
        hashes[path] = sha256(value).hexdigest()
    return hashes


def write_once(path, content):
    payload = (json.dumps(content, indent=2, allow_nan=False) + '\n').encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as output:
        output.write(payload)
    return sha256(payload).hexdigest()


def allocations(images):
    """Form all routing inputs and decisions without using outcome labels."""
    p = [row['p'] for row in images]
    quality = [row['good_quality'] for row in images]
    entropy = [row['entropy'] for row in images]
    all_pointwise, results = {}, []
    for mode in MODES:
        sets = [row[mode] if mode != 'no_cp' else None for row in images]
        all_pointwise[mode] = [pointwise_action(probabilities, good, uncertainty, conformal_set=labels)[0].value
                               for probabilities, good, uncertainty, labels in zip(p, quality, entropy, sets,
                                                                                   strict=True)]
        for index, budget in enumerate(BUDGET_PAIRS):
            config = PolicyConfig(review_budget=budget[0], recapture_budget=budget[1])
            for method in METHODS:
                row = {'id': f'{mode}:{method}:{index}', 'mode': mode, 'method': method, 'budget': list(budget)}
                try:
                    actions = allocate(p, quality, entropy, config, method, conformal_sets=sets)
                except BudgetOverloaded as exc:
                    row.update(status='overload', actions=None, diagnostics=exc.diagnostics)
                else:
                    row.update(status='evaluated', actions=[action.value for action in actions], diagnostics=None)
                results.append(row)
    return all_pointwise, results


def evaluate(images, all_pointwise, scenarios):
    grades = [row['grade'] for row in images]
    groups = [row['patient'] for row in images]
    pointwise = {mode: summarize_actions(grades, groups, actions) for mode, actions in all_pointwise.items()}
    results = [{**{key: value for key, value in row.items() if key != 'actions'},
                'outcomes': summarize_actions(grades, groups, row['actions']) if row['actions'] is not None else None,
                'patient_capacity': patient_capacity(pointwise[row['mode']]['patient_actions'], *row['budget'])}
               for row in scenarios]
    indexed = {row['id']: row for row in scenarios}
    comparisons = []
    for row in scenarios:
        if row['method'] == 'dual_budget':
            continue
        mode, _, index = row['id'].split(':')
        candidate = indexed[f'{mode}:dual_budget:{index}']
        comparison = {'mode': mode, 'budget': row['budget'], 'candidate': candidate['id'], 'reference': row['id']}
        if row['actions'] is None or candidate['actions'] is None:
            comparison.update(status='unavailable', matched_image_workload=None, matched_patient_workload=None,
                              identical_image_actions=None, intervals=None)
        else:
            comparison.update(status='comparison', **paired_comparison(grades, groups, candidate['actions'],
                                                                      row['actions']))
        comparisons.append(comparison)
    return pointwise, results, comparisons


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True, help='Original DeepDRiD author package directory')
    parser.add_argument('--source-commit', required=True)
    args = parser.parse_args()
    if PRIVATE.exists() or REPORT.exists():
        parser.error('Frozen output already exists')
    hashes = bind_source(args.source_commit)
    print('Verified the locked source. Starting the frozen Set-C CPU evaluation.', flush=True)
    images, provenance = load_and_score(args.root.resolve())
    pointwise, scenarios = allocations(images)
    recorded = write_once(PRIVATE, {
        'experiment_id': STUDY, 'source_commit': args.source_commit, 'images': images, 'pointwise': pointwise,
        'scenarios': [{k: v for k, v in row.items() if k != 'diagnostics'} for row in scenarios],
    })
    pointwise, scenarios, comparisons = evaluate(images, pointwise, scenarios)
    evaluated = sum(row['status'] == 'evaluated' for row in scenarios)
    matched = sum(row['intervals'] is not None for row in comparisons)
    report = {
        'experiment_id': STUDY, 'source_commit': args.source_commit, 'source_sha256': hashes,
        'input_provenance': provenance, 'private_witness_sha256': recorded,
        'model_inference_performed': True, 'partition': 'same_release_held_out',
        'images': len(images), 'patients': len({row['patient'] for row in images}),
        'prediction_metrics': summarize_predictions(images), 'policy': asdict(PolicyConfig()),
        'budget_pairs': BUDGET_PAIRS, 'pointwise': pointwise, 'scenarios': scenarios, 'comparisons': comparisons,
        'runtime': {'python': platform.python_version(), 'system': platform.system(),
                    **{package: version(package) for package in ('numpy', 'scipy', 'torch', 'torchvision', 'pillow')}},
    }
    write_once(REPORT, report)
    print(f'Scored {report["images"]} photos from {report["patients"]} patients. '
          f'{evaluated}/{len(scenarios)} scenarios fit both caps. {matched}/{len(comparisons)} exactly matched pairs.')


if __name__ == '__main__':
    main()
