"""Fit and evaluate locked DeepDRiD paired-view acquisition policies."""

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

import numpy as np

from qsart.paired_view_inputs import load_and_score
from qsart.paired_view_metrics import evaluate_policy, paired_interval, summarize_policy
from qsart.paired_view_policy import (ACQUISITION_COST, BUDGETS, COMPARATORS, FEATURE_NAMES, budget_routes,
                                      first_view_features, fit_value)


ROOT = Path(__file__).resolve().parents[1]
STUDY = 'paired_view_v1'
PRIVATE = ROOT / 'artifacts' / STUDY / 'scores_and_routes.json'
REPORT = ROOT / 'reports' / STUDY / 'evaluation.json'
SOURCE_PATHS = (
    'docs/research/PAIRED_VIEW_V1_PROTOCOL.md', 'scripts/run_paired_view_v1.py', 'scripts/audit_paired_view_v1.py',
    'qsart/paired_view_inputs.py', 'qsart/paired_view_metrics.py', 'qsart/paired_view_policy.py', 'qsart/triage_v2.py',
    'qsart/inference/engine.py', 'qsart/models/checkpoint.py', 'qsart/training/data.py',
    'qsart/preprocessing/fundus.py', 'requirements.txt',
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
            raise ValueError(f'Execution source differs from the locked commit: {path}')
        hashes[path] = sha256(value).hexdigest()
    return hashes


def write_once(path, record):
    payload = (json.dumps(record, indent=2, allow_nan=False) + '\n').encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(payload)
    return sha256(payload).hexdigest()


def initial_features(rows):
    return first_view_features([row['first'] for row in rows], [row['first_quality'] for row in rows])


def build_routes(fit_rows, selection_features):
    # Outcomes for the two recorded views are used in fitting on development patients only.
    no_extra = np.zeros(len(fit_rows), dtype=bool)
    first = evaluate_policy(fit_rows, no_extra)['ledger']
    second = evaluate_policy(fit_rows, ~no_extra)['ledger']
    gross_gain = first - second + ACQUISITION_COST
    model = fit_value(initial_features(fit_rows), gross_gain)
    return model, budget_routes(selection_features, model), {
        'mean_gross_gain': float(gross_gain.mean()),
        'positive_after_cost': int((gross_gain > ACQUISITION_COST).sum()),
    }


def compare_routes(rows, routes):
    scenarios = [{**{key: value for key, value in route.items() if key != 'selected'},
                  'outcomes': summarize_policy(rows, route['selected'])} for route in routes]
    indexed = {route['id']: route for route in routes}
    pairs = []
    for index in range(1, len(BUDGETS) - 1):
        candidate = indexed[f'voi:{index}']
        for name in COMPARATORS:
            reference = indexed[f'{name}:{index}']
            pairs.append({'candidate': candidate['id'], 'reference': reference['id'], 'budget': candidate['budget'],
                          'extra_views': candidate['slots'],
                          'intervals': paired_interval(rows, candidate['selected'], reference['selected'])})
    return scenarios, pairs


def population(rows, split):
    return {'split': split, 'photos': 2 * len(rows), 'eyes': len(rows),
            'patients': len({row['patient'] for row in rows})}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-commit', required=True)
    args = parser.parse_args()
    if PRIVATE.exists() or REPORT.exists():
        parser.error('Frozen output already exists')
    hashes = bind_source(args.source_commit)
    print('Verified the locked source. Scoring frozen Set-A pairs on CPU.', flush=True)
    fit_rows, selection_rows, provenance = load_and_score()
    if {row['patient'] for row in fit_rows} & {row['patient'] for row in selection_rows}:
        raise ValueError('Acquisition fitting and evaluation must be patient disjoint')
    model, routes, training_summary = build_routes(fit_rows, initial_features(selection_rows))
    core_model = asdict(model)
    recorded = write_once(PRIVATE, {
        'experiment_id': STUDY, 'source_commit': args.source_commit, 'fit_rows': fit_rows,
        'selection_rows': selection_rows, 'model': core_model, 'routes': routes,
    })
    scenarios, pairs = compare_routes(selection_rows, routes)
    report = {
        'experiment_id': STUDY, 'source_commit': args.source_commit, 'source_sha256': hashes,
        'input_provenance': provenance, 'private_witness_sha256': recorded,
        'model_inference_performed': True, 'partition': 'validation',
        'fit_population': population(fit_rows, 'train'), 'selection_population': population(selection_rows, 'validation'),
        'model': {**core_model, 'feature_names': FEATURE_NAMES, 'fit_gain': training_summary},
        'budgets': BUDGETS, 'scenarios': scenarios, 'comparisons': pairs,
        'runtime': {'python': platform.python_version(), 'system': platform.system(),
                    **{package: version(package) for package in ('numpy', 'scipy', 'torch', 'torchvision', 'pillow')}},
    }
    write_once(REPORT, report)
    print(f'Evaluated {len(selection_rows)} eye pairs, {len(scenarios)} policy scenarios and {len(pairs)} comparisons.')


if __name__ == '__main__':
    main()
