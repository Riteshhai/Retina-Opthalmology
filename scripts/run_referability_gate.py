"""Compare conformal gates for referability on frozen DeepDRiD development scores."""

import argparse
from dataclasses import asdict, replace
from importlib import metadata
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qsart.inference.conformal_sidecar import ConformalSidecar, temperature_probabilities
from qsart.models.checkpoint import sha256_file
from qsart.training.data import read_manifest, split_fingerprint
from qsart.triage_v2 import (
    Action, BudgetOverloaded, PolicyConfig, _diagnostics, allocate, pointwise_action, screening_metrics,
)
from qsart.uncertainty.conformal import coverage_metrics, fit_conformal, predict_sets


ROOT = Path(__file__).resolve().parents[1]
SLUG = 'referability_gate_v1'
PROTOCOL = 'docs/research/REFERABILITY_GATE_V1_PROTOCOL.md'
ALPHAS = (.10, .05)
BUDGET_PAIRS = ((.30, .20), (.60, .40), (.90, .40), (1., .40))
METHODS = ('ordinal_projected', 'lac_projected', 'binary_lac')
ORIGINAL_METHODS = {'ordinal_projected': 'ordinal_greedy', 'lac_projected': 'lac'}
ARTIFACTS = {
    'ordinal_report': ('reports/ordinal_study_v1/evaluation.json',
                       '657359119b33a6639328052c898c761c3330be577cd01e14c9b5f3eeb4f1e9a2'),
    'ordinal_cache': ('artifacts/ordinal_study_v1/scores_and_sets.json',
                      '0ef95338994b3a587dd38b3034f5c059f063a77901b5af3f17f39292d5f40b4c'),
    'robustness_report': ('reports/robustness_v1/evaluation.json',
                          'f63848e4c68f18a5b4c1197e48fff6c7acf61412f6ffd0ec85a640228493e8cb'),
    'robustness_cache': ('artifacts/robustness_v1/scores.json',
                         '3a5a7b3bee6092223d564666173652f9092a3437bbc48ceda6c9660817f759d6'),
    'manifest': ('data/processed/deepdrid/development_v1.csv',
                 '45e9f2d34ed54a12056b57a808d03cfaf29ccc970d148e9eb107b79cf5ac3e0d'),
    'dr': ('checkpoints/development_v1/selected/dr.pt',
           'f530056c27f83dbff8bba4ebb481cf7a9d82289947c4eda655a69ed22685a527'),
    'quality': ('checkpoints/development_v1/selected/quality.pt',
                'a600872ce8e2b256205350a5ef6c99b77d71f679d217c20c26810a3ed76eff43'),
    'calibration': ('checkpoints/development_v1/selected/calibration.json',
                    '674821e445490c209fab99e56b0e9915ac406ca4ec8694e5efdbed63e456a737'),
    'conformal': ('checkpoints/spec_v1/conformal_r1.json',
                  '15a98f6bfe93f81156c5592701995d08796dd889e3a5e43bcbde7ab5aa61c221'),
}
SOURCE_PATHS = (
    PROTOCOL, 'scripts/run_referability_gate.py', 'qsart/inference/conformal_sidecar.py',
    'qsart/models/checkpoint.py', 'qsart/training/data.py', 'qsart/triage.py', 'qsart/triage_v2.py',
    'qsart/uncertainty/conformal.py',
)


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def execution_sources(commit):
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    status = subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip()
    if head != commit or status:
        raise ValueError('Execution requires the exact declared clean source commit')
    hashes = {}
    for name in SOURCE_PATHS:
        actual = (ROOT / name).read_bytes()
        if actual != subprocess.check_output(['git', 'show', f'{commit}:{name}'], cwd=ROOT):
            raise ValueError(f'Changed execution source: {name}')
        hashes[name] = hashlib.sha256(actual).hexdigest()
    return hashes


def load_bound_inputs():
    paths = {key: ROOT / value[0] for key, value in ARTIFACTS.items()}
    hashes = {key: sha256_file(path) for key, path in paths.items()}
    if hashes != {key: value[1] for key, value in ARTIFACTS.items()}:
        raise ValueError('A frozen manifest, checkpoint, sidecar, or score artifact differs')
    ordinal_report, ordinal, robustness_report, robustness = [
        read_json(paths[key]) for key in ('ordinal_report', 'ordinal_cache', 'robustness_report', 'robustness_cache')
    ]
    model_hashes = {key: hashes[key] for key in ('dr', 'quality', 'calibration', 'conformal')}
    if (
        ordinal_report['private_output_sha256'] != hashes['ordinal_cache']
        or ordinal_report['selection_cache_sha256'] != hashes['robustness_cache']
        or ordinal_report['selection_report_sha256'] != hashes['robustness_report']
        or robustness_report['private_scores_sha256'] != hashes['robustness_cache']
        or any(report['artifact_sha256'] != model_hashes or report['manifest_sha256'] != hashes['manifest']
               for report in (ordinal_report, robustness_report))
    ):
        raise ValueError('Frozen score cache provenance differs')
    bundle = ConformalSidecar(
        paths['conformal'], dr_checkpoint=paths['dr'], quality_checkpoint=paths['quality'],
        calibration=paths['calibration'],
    ).bundle
    frame = read_manifest(paths['manifest'])
    if set(frame.split) != {'train', 'validation', 'calibration'}:
        raise ValueError('Expected the development-only manifest')
    selected = frame.loc[frame.split.eq('validation')].reset_index(drop=True)
    calibration = frame.loc[frame.split.eq('calibration')]
    patients = sorted(calibration.subject_id.unique(), key=lambda value: (
        hashlib.sha256(f'spec-v1-conformal-20261020:{value}'.encode()).hexdigest(), str(value)))
    temperature_frame = calibration.loc[calibration.subject_id.isin(patients[:21])]
    conformal_frame = calibration.loc[calibration.subject_id.isin(patients[21:])]
    if (
        len(patients) != 42 or len(selected) != 168 or selected.subject_id.nunique() != 42
        or len(temperature_frame) != 84 or len(conformal_frame) != 84
        or set(selected.subject_id) & set(patients)
        or bundle['manifest_sha256'] != hashes['manifest']
        or bundle['temperature_fit_split_sha256'] != split_fingerprint(temperature_frame)
        or bundle['split_sha256'] != split_fingerprint(conformal_frame)
    ):
        raise ValueError('Frozen selection or independent calibration roles differ')
    for row in (*conformal_frame.itertuples(), *selected.itertuples()):
        if sha256_file(row.image_path) != row.sha256:
            raise ValueError('Source image bytes differ from the frozen manifest')
    calibration_labels = conformal_frame.dr_grade.to_numpy(dtype=int)
    calibration_groups = conformal_frame.subject_id.to_numpy()
    truth = selected.dr_grade.to_numpy(dtype=int)
    groups = selected.subject_id.to_numpy()
    if (
        ordinal['calibration_grades'] != calibration_labels.tolist()
        or ordinal['calibration_subject_ids'] != calibration_groups.tolist()
        or ordinal['selection_grades'] != truth.tolist()
        or ordinal['selection_subject_ids'] != groups.tolist()
        or robustness['original_grades'] != truth.tolist()
        or robustness['original_subject_ids'] != groups.tolist()
    ):
        raise ValueError('Saved labels or patient order differs from the frozen manifest')
    rows = robustness['conditions']['clean']['rows']
    temperature = bundle['independent_temperature']['first_pass']
    if (
        ordinal_report['independent_temperature'] != temperature
        or len(ordinal['calibration_raw_probabilities']) != len(conformal_frame)
        or len(rows) != len(selected)
        or any(row['second_pass_activated'] or row['data_purpose'] != 'real_research'
               or row['conformal']['path'] != 'first_pass' or row['model_version'] != bundle['model_version']
               for row in rows)
    ):
        raise ValueError('Expected only genuine, clean, first-pass scores and the frozen temperature')
    return hashes, bundle, ordinal, rows, calibration_labels, calibration_groups, truth, groups


def binary_probabilities(probabilities, referable_grade):
    return np.column_stack((probabilities[:, :referable_grade].sum(1),
                            probabilities[:, referable_grade:].sum(1)))


def project_sets(grade_sets, referable_grade):
    return [sorted({int(grade >= referable_grade) for grade in values}) for values in grade_sets]


def expand_sets(binary_sets, referable_grade):
    """Encode binary membership solely for the unchanged five-grade guard."""
    return [[grade for grade in range(5) if int(grade >= referable_grade) in values] for values in binary_sets]


def fit_sets(method, alpha, p_cal, p_eval, labels, groups, bundle, ordinal, rows, config):
    grade_cut = config.referable_grade
    if method == 'binary_lac':
        artifact = fit_conformal(
            binary_probabilities(p_cal, grade_cut), (labels >= grade_cut).astype(int),
            method='lac', alpha=alpha, groups=groups, empty_policy='allow',
        )
        return artifact, predict_sets(binary_probabilities(p_eval, grade_cut), artifact)
    historical_method = ORIGINAL_METHODS[method]
    artifact = fit_conformal(p_cal, labels, method=historical_method, alpha=alpha, groups=groups,
                            empty_policy='allow')
    raw_sets = predict_sets(p_eval, artifact)
    sidecar = bundle['models']['first_pass'][str(alpha)][historical_method]
    (prior,) = [fit for fit in ordinal['fits_and_sets'] if fit['method'] == historical_method
                and fit['unit'] == 'patient_max' and fit['alpha'] == alpha and fit['seed'] is None]
    if (
        any(artifact[key] != sidecar[key] or artifact[key] != prior['artifact'][key] for key in ('qhat', 'rank'))
        or raw_sets != prior['raw_sets']
        or raw_sets != predict_sets(p_eval, {**sidecar, 'empty_policy': 'allow'})
    ):
        raise ValueError('Frozen five-grade thresholds or raw sets did not replay')
    if alpha == .10 and method == 'ordinal_projected' and (
        raw_sets != [row['conformal_set'] for row in rows]
        or raw_sets != [row['conformal']['prediction_set'] for row in rows]
    ):
        raise ValueError('Default deployed ordinal sets did not replay exactly')
    return artifact, project_sets(raw_sets, grade_cut)


def summarize_sets(binary_sets, truth, groups, referable_grade):
    binary_truth = (truth >= referable_grade).astype(int)
    metrics = coverage_metrics(binary_sets, binary_truth, num_classes=2, groups=groups)
    covered = np.asarray([label in values for label, values in zip(binary_truth, binary_sets, strict=True)])
    metrics['nonreferable_singleton_count'] = binary_sets.count([0])
    metrics['referable_singleton_count'] = binary_sets.count([1])
    metrics['by_true_grade'] = {}
    for grade in range(5):
        mask = truth == grade
        metrics['by_true_grade'][str(grade)] = {
            'n': int(mask.sum()), 'coverage': float(covered[mask].mean()) if mask.any() else None,
        }
    return metrics


def summarize_actions(truth, p, actions, config, *, pointwise=False):
    metrics = screening_metrics(truth, p, actions, config)
    if pointwise:
        for key in ('review_capacity', 'recapture_capacity', 'review_budget_violation',
                    'recapture_budget_violation'):
            del metrics[key]
    action_values = np.array([action.value for action in actions])
    metrics['by_true_grade'] = {
        str(grade): {action.value: int(np.sum((truth == grade) & (action_values == action.value)))
                     for action in Action}
        for grade in range(5)
    }
    return metrics


def evaluate_allocations(p, quality, uncertainty, truth, binary_sets, config):
    guard_sets = expand_sets(binary_sets, config.referable_grade)
    pointwise = [pointwise_action(probs, good, entropy, config, grade_set)[0]
                 for probs, good, entropy, grade_set in zip(p, quality, uncertainty, guard_sets, strict=True)]
    mandatory = np.asarray([action is Action.SPECIALIST_REVIEW for action in pointwise])
    poor = quality < config.minimum_quality
    public, private = [], []
    for review, recapture in BUDGET_PAIRS:
        active = replace(config, review_budget=review, recapture_budget=recapture)
        diagnostics = _diagnostics(len(p), mandatory, poor, active)
        budget = {'review_budget': review, 'recapture_budget': recapture}
        try:
            actions = allocate(p, quality, uncertainty, active, conformal_sets=guard_sets)
        except BudgetOverloaded as exc:
            if exc.diagnostics != diagnostics:
                raise ValueError('Pointwise and cohort allocator guard counts differ') from exc
            public.append({**budget, 'status': 'overloaded', 'diagnostics': diagnostics, 'metrics': None})
            private.append({**budget, 'actions': None})
        else:
            public.append({**budget, 'status': 'evaluated', 'diagnostics': diagnostics,
                           'metrics': summarize_actions(truth, p, actions, active)})
            private.append({**budget, 'actions': [action.value for action in actions]})
    return summarize_actions(truth, p, pointwise, config, pointwise=True), pointwise, public, private


def encode_json(value):
    return (json.dumps(value, indent=2, allow_nan=False) + '\n').encode('utf-8')


def output_folder(value, parent):
    path = (ROOT / value).resolve()
    base = (ROOT / parent).resolve()
    if path == base or not path.is_relative_to(base) or path.exists():
        raise ValueError(f'Expected a new experiment directory inside {parent}')
    return path


def run(args):
    output = output_folder(args.output, 'reports')
    private_folder = output_folder(args.private, 'artifacts')
    sources = execution_sources(args.source_commit)
    hashes, bundle, ordinal, rows, labels, cal_groups, truth, groups = load_bound_inputs()
    temperature = bundle['independent_temperature']['first_pass']
    p_cal = temperature_probabilities(ordinal['calibration_raw_probabilities'], temperature)
    p_eval = temperature_probabilities([row['uncalibrated']['grade_probabilities'] for row in rows], temperature)
    p = np.asarray([row['grade_probabilities'] for row in rows])
    quality = np.asarray([1 - row['quality']['poor_probability'] for row in rows])
    uncertainty = np.asarray([row['uncertainty']['entropy'] for row in rows])
    config = PolicyConfig()
    public_fits, private_fits = [], []
    for alpha in ALPHAS:
        for method in METHODS:
            artifact, binary_sets = fit_sets(method, alpha, p_cal, p_eval, labels, cal_groups,
                                             bundle, ordinal, rows, config)
            point_metrics, pointwise, public_allocations, private_allocations = evaluate_allocations(
                p, quality, uncertainty, truth, binary_sets, config,
            )
            public_fits.append({
                'alpha': alpha, 'method': method, 'rank': artifact['rank'], 'qhat': artifact['qhat'],
                'set_metrics': summarize_sets(binary_sets, truth, groups, config.referable_grade),
                'pointwise_metrics': point_metrics, 'allocations': public_allocations,
            })
            private_fits.append({
                'alpha': alpha, 'method': method, 'raw_binary_sets': binary_sets,
                'pointwise_actions': [action.value for action in pointwise], 'allocations': private_allocations,
            })
    private = encode_json({'experiment_id': SLUG, 'fits': private_fits})
    report = {
        'schema_version': 1, 'experiment_id': SLUG, 'spec_version': '1.0',
        'dataset': 'DeepDRiD', 'partition': 'validation', 'scope': 'development_referability_gate_diagnostic',
        'images': len(truth), 'patients': len(set(groups)), 'model_version': bundle['model_version'],
        'source_commit': args.source_commit, 'source_sha256': sources, 'protocol_sha256': sources[PROTOCOL],
        'input_sha256': hashes, 'private_output_sha256': hashlib.sha256(private).hexdigest(),
        'package_versions': {name: metadata.version(name) for name in ('numpy', 'pandas', 'scipy')},
        'calibration': {
            'temperature_images': bundle['temperature_images'], 'temperature_patients': bundle['temperature_patients'],
            'conformal_images': bundle['conformal_images'], 'conformal_patients': bundle['conformal_patients'],
            'independent_first_pass_temperature': temperature,
            'temperature_fit_split_sha256': bundle['temperature_fit_split_sha256'],
            'conformal_split_sha256': bundle['split_sha256'],
        },
        'policy': asdict(config), 'budget_pairs': BUDGET_PAIRS, 'fits': public_fits,
        'default_ordinal_sets_replayed': True,
        'limitations': [
            'Exploratory development comparison on selection patients already used for model choice.',
            'Coverage is empirical on a small reused cohort and establishes no grade-conditional, shifted-domain, '
            'or clinical guarantee.',
            'Binary sets describe referability rather than inclusion of the exact disease grade.',
            'Pointwise results have no cohort budget. Cohort allocations use one frozen selection batch.',
            'No new model forwards, probability-temperature fitting, policy tuning, or held-out evaluation.',
            'Decisions are research recommendations and proxy costs are unvalidated.',
        ],
    }
    public = encode_json(report)
    if any(token in public for token in (
        b'"subject_id"', b'"patient_id"', b'"image_id"', b'"image_path"', b'"raw_binary_sets"',
        b'"pointwise_actions"', b'"actions"',
    )) or re.search(rb'"(?:[A-Za-z]:[\\/]|/|\\\\)', public):
        raise ValueError('Public report contains a private field or path')
    private_folder.mkdir(parents=True)
    with (private_folder / 'sets_and_actions.json').open('xb') as stream:
        stream.write(private)
    output.mkdir(parents=True)
    with (output / 'evaluation.json').open('xb') as stream:
        stream.write(public)
    print(json.dumps({'experiment_id': SLUG, 'fits': len(public_fits),
                      'allocations': sum(len(fit['allocations']) for fit in public_fits)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--output', type=Path, default=Path('reports') / SLUG)
    parser.add_argument('--private', type=Path, default=Path('artifacts') / SLUG)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
