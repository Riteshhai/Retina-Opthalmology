"""Run the locked primary-seed APTOS schedule extension; default is readiness only."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
from importlib import metadata as package_metadata
import json
from pathlib import Path
import re
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qsart.calibration.temperature import calibration_metrics, fit_temperature
from qsart.models.checkpoint import load_checkpoint, sha256_file
from qsart.training.data import split_fingerprint
from qsart.training.finetune import FinetuneConfig, source_provenance, train_finetune
from qsart.training.transfer import _input_content_hash
from scripts.inspect_datasets import default_aptos_root
from scripts.run_aptos_development import CALIBRATION_UNIT, _bools, _grading, conformal_comparisons, read_json
from scripts.run_aptos_finetune import (BASELINE_SHA256, DEVELOPMENT_SHA256, INITIAL_SHA256,
    SOURCE_PATHS, cached_logits, candidate_record, paired_fingerprint_bootstrap,
    posthoc_logits, scoring_cache, validate_ready)

EXPERIMENT = 'aptos_optimization_v2'
SCOPE = 'source_development_primary_seed_schedule_extension'
PRIOR_SHA256 = '8c6f50e93db412fc845600a4e368094c46fce041109806e37819dcd6bb70c5f4'
PREFIX = 'aptos-optimization-v2'


def fixed_config(path):
    expected = replace(FinetuneConfig(), epochs=15)
    actual = yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    if actual != asdict(expected):
        raise ValueError('Configuration differs from the declared 15-epoch primary-seed extension')
    return expected


def execution_sources(root, args):
    if (not args.source_commit or subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip() != args.source_commit
            or subprocess.check_output(['git', 'status', '--porcelain'], cwd=root, text=True).strip()):
        raise ValueError('Execution requires the exact declared clean source commit')
    names = list(SOURCE_PATHS)+['scripts/run_aptos_optimization.py', 'api/research.py', 'dashboard/app.py',
        args.config.resolve().relative_to(root).as_posix(), args.protocol.resolve().relative_to(root).as_posix()]
    result = {}
    for name in dict.fromkeys(names):
        value = (root/name).read_bytes()
        if value != subprocess.check_output(['git', 'show', f'{args.source_commit}:{name}'], cwd=root):
            raise ValueError(f'Changed execution source: {name}')
        result[name] = hashlib.sha256(value).hexdigest()
    return result


def prior_primary(path, validation):
    if sha256_file(path) != PRIOR_SHA256:
        raise ValueError('The preserved four-epoch study changed')
    previous = read_json(path)
    if previous['summary']['primary_training_seed'] != 2026:
        raise ValueError('The original primary seed differs')
    row = next(row for row in previous['runs'] if row['training_seed'] == 2026)
    folder = Path('checkpoints/aptos_finetune_v1/seed2026')
    scores = folder/'selection_predictions.csv'
    if sha256_file(scores) != row['artifact_sha256']['selection_predictions.csv']:
        raise ValueError('The preserved primary comparator scores changed')
    records = pd.read_csv(scores, float_precision='round_trip')
    if (not np.array_equal(records.manifest_row, validation.index) or
            not np.array_equal(records.true_dr_grade, validation.dr_grade)):
        raise ValueError('The primary comparator is not paired with the original selection cohort')
    p = records[[f'dr_grade_p{i}' for i in range(5)]].to_numpy(float)
    p /= p.sum(1, keepdims=True)
    return row, p


def preserved_files(root):
    names = subprocess.check_output(['git', 'ls-files', 'reports/**/*.json', 'reports/**/*.png'], cwd=root, text=True).splitlines()
    names += ['MASTER_SPEC.md', 'docs/specifications/MASTER_SPEC_v1.0.md',
        'checkpoints/development_v1/selected/dr.pt', 'checkpoints/development_v1/selected/quality.pt',
        'checkpoints/development_v1/selected/calibration.json', 'checkpoints/development_v1/selected/model_card.json',
        'checkpoints/spec_v1/conformal_r1.json', 'checkpoints/aptos_v1/dr_efficientnet_b0/best.pt',
        'data/manifests/aptos_v1/development.csv', 'data/manifests/aptos_v1/calibration_roles.csv']
    for seed in (2026, 2027, 2028):
        prior = read_json(root/'reports/aptos_finetune_v1/evaluation.json')
        run = next(row for row in prior['runs'] if row['training_seed'] == seed)
        for name, expected in run['artifact_sha256'].items():
            path = root/f'checkpoints/aptos_finetune_v1/seed{seed}'/name
            if sha256_file(path) != expected:
                raise ValueError('An original fine-tuning artifact changed')
            names.append(path.relative_to(root).as_posix())
    return {name: sha256_file(root/name) for name in sorted(set(names))}


def write_or_verify(path, value, resume):
    if path.exists():
        if not resume or path.read_bytes() != value:
            raise ValueError('Existing execution inputs differ or require explicit --resume')
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as handle:
            handle.write(value)


def ensure_aggregate(report):
    json.dumps(report, allow_nan=False)

    def check(value):
        if isinstance(value, dict):
            if {'image_id', 'image_path', 'subject_id', 'patient_id'} & value.keys():
                raise ValueError('Public report contains private identities or absolute paths')
            for item in value.values():
                check(item)
        elif isinstance(value, list):
            for item in value:
                check(item)
        elif isinstance(value, str) and (re.search(r'[A-Za-z]:[\\/]', value) or value.startswith(('/', '\\\\'))):
            raise ValueError('Public report contains private identities or absolute paths')

    check(report)


def calibration_masks(lookup):
    """Own boolean arrays so Torch never wraps a read-only pandas view."""
    temperature = lookup.calibration_role.eq('temperature')
    conformal = lookup.calibration_role.eq('conformal') & _bools(lookup.is_group_representative)
    return (temperature.to_numpy(dtype=bool, copy=True),
            conformal.to_numpy(dtype=bool, copy=True))


def run(args):
    root = Path(__file__).resolve().parents[1]
    original, roles, runtime, runtime_roles, reps, bindings, baseline = validate_ready(args)
    config = fixed_config(args.config)
    validation = runtime.loc[runtime.split.eq('validation')]
    old_primary, old_p = prior_primary(args.prior_report, validation)
    if not args.execute:
        print(json.dumps({'status': 'ready_inputs_only', 'training_or_inference_performed': False,
            'experiment_id': EXPERIMENT, 'training_seed': config.seed, 'maximum_epochs': config.epochs,
            'patience': config.patience, 'population': runtime.split.value_counts().to_dict(),
            'reserved_test_or_external_inference_supported': False}, indent=2))
        return
    for path, expected in ((args.private, root/'artifacts'/EXPERIMENT),
                           (args.checkpoints, root/'checkpoints'/EXPERIMENT),
                           (args.output, root/'reports'/EXPERIMENT)):
        if path.resolve() != expected:
            raise ValueError('The versioned study artifact destinations are fixed')
    if (args.output/'evaluation.json').exists():
        raise ValueError('Preserve the completed schedule-extension report')
    sources = execution_sources(root, args)
    snapshot = preserved_files(root)
    content_hash = _input_content_hash(runtime)
    runtime_manifest = args.private/'runtime_development.csv'
    for path, frame in ((runtime_manifest, runtime), (args.private/'runtime_calibration_roles.csv', runtime_roles)):
        write_or_verify(path, frame.to_csv(index=False, lineterminator='\n').encode(), args.resume)
    lock = {'source_commit': args.source_commit, 'execution_sources': sources,
        'configuration': asdict(config), 'original_bindings': bindings,
        'runtime_manifest_sha256': sha256_file(runtime_manifest),
        'runtime_calibration_roles_sha256': sha256_file(args.private/'runtime_calibration_roles.csv'),
        'development_content_sha256': content_hash, 'initial_checkpoint_sha256': INITIAL_SHA256,
        'prior_study_sha256': PRIOR_SHA256, 'preserved_artifact_sha256': snapshot,
        'selection_only_design_basis': 'All three source-development v1 runs selected epoch four at their cap.',
        'prior_idrid_outcome_inspection_disclosed': True, 'external_outcomes_used_for_tuning': False}
    write_or_verify(args.private/'execution_lock.json', (json.dumps(lock, indent=2)+'\n').encode(), args.resume)
    torch.set_num_threads(config.cpu_threads)
    source = source_provenance()
    fitting = runtime.loc[runtime.split.isin(['train', 'validation'])]
    cache_dir = args.private/'inputs'
    start = time.perf_counter()
    warm_model, _ = load_checkpoint(args.initial_checkpoint)
    warm_logits, warm_cache = cached_logits(warm_model, fitting, runtime_manifest, cache_dir, config)
    warm_p = warm_logits.double().softmax(1).numpy()
    saved = pd.read_csv(args.initial_checkpoint.parent/'selection_predictions.csv', float_precision='round_trip')
    baseline_row = next(row for row in baseline['candidates'] if row['checkpoint_sha256'] == INITIAL_SHA256)
    if sha256_file(args.initial_checkpoint.parent/'selection_predictions.csv') != baseline_row['artifact_sha256']['selection_predictions.csv']:
        raise ValueError('Original baseline scores changed')
    if not np.array_equal(saved.manifest_row, validation.index) or not np.array_equal(saved.true_dr_grade, validation.dr_grade):
        raise ValueError('Baseline selection identities differ')
    baseline_p = saved[[f'dr_grade_p{i}' for i in range(5)]].to_numpy(float)
    baseline_p /= baseline_p.sum(1, keepdims=True)
    delta = float(np.max(np.abs(warm_p-baseline_p)))
    if delta > 1e-5 or not np.array_equal(warm_p.argmax(1), baseline_p.argmax(1)):
        raise ValueError('Warm-start canonical preprocessing changed')
    preflight = {'images': len(validation), 'max_abs_probability_delta': delta,
        'all_argmax_predictions_match': True, 'tolerance': 1e-5,
        'canonical_cache_sha256': warm_cache['cache_file_sha256'],
        'canonical_pixel_tensor_sha256': warm_cache['tensor_sha256']}
    print(json.dumps({'event': 'source_preflight_passed', **preflight}), flush=True)
    del warm_model, warm_logits, warm_p
    folder = args.checkpoints/'seed2026'
    if not folder.exists():
        train_finetune(runtime_manifest, args.initial_checkpoint, folder, cache_dir,
                       config=config, model_version_prefix=PREFIX)
    elif not args.resume:
        raise ValueError('Existing training output requires explicit --resume')
    elif not (folder/'training_result.json').exists():
        train_finetune(runtime_manifest, args.initial_checkpoint, folder, cache_dir,
                       config=config, resume_last=True, model_version_prefix=PREFIX)
    candidate, model, checkpoint, logits = candidate_record(folder, config, runtime, runtime_manifest, cache_dir, source)
    if checkpoint['metadata'].get('model_version_prefix') != PREFIX:
        raise ValueError('The completed model belongs to another experiment namespace')
    calibration = runtime.loc[runtime.split.eq('calibration')]
    pixels, calibration_cache = scoring_cache(calibration, runtime_manifest,
        args.private/'posthoc_inputs/calibration_uint8.pt', config)
    calibration_logits = posthoc_logits(model, pixels, config)
    lookup = runtime_roles.set_index('image_id').loc[calibration.image_id]
    t_mask, cp_mask = calibration_masks(lookup)
    truth = validation.dr_grade.to_numpy(int)
    calibration_truth = torch.tensor(calibration.dr_grade.to_numpy(), dtype=torch.long)
    temperature = fit_temperature(calibration_logits[t_mask], calibration_truth[t_mask])
    p = (logits.double()/temperature).softmax(1).numpy()
    cp_p = (calibration_logits[cp_mask].double()/temperature).softmax(1).numpy()
    binding = {'checkpoint_sha256': candidate['checkpoint_sha256'], 'model_training_seed': 2026,
        'original_development_manifest_sha256': DEVELOPMENT_SHA256,
        'runtime_manifest_sha256': sha256_file(runtime_manifest), 'independent_temperature': temperature,
        'temperature_split_fingerprint': split_fingerprint(calibration.loc[t_mask]),
        'conformal_representatives_fingerprint': split_fingerprint(calibration.loc[cp_mask])}
    comparisons, private_sets = conformal_comparisons(cp_p, calibration_truth[cp_mask].numpy(), p,
        truth, reps.loc[validation.index].to_numpy(bool), binding)
    if len(comparisons) != 14:
        raise ValueError('All fourteen fixed conformal comparisons must be retained')
    candidate.update({'grading_before_temperature': _grading(truth, logits.double().softmax(1).numpy()),
        'grading': _grading(truth, p), 'calibration': {'temperature': temperature,
            'temperature_fit_before': calibration_metrics(calibration_logits[t_mask], calibration_truth[t_mask]),
            'temperature_fit_after': calibration_metrics(calibration_logits[t_mask], calibration_truth[t_mask], temperature),
            'selection_before': calibration_metrics(logits, truth), 'selection_after': calibration_metrics(logits, truth, temperature)}})
    paired = {}
    for name, probabilities in (('frozen_feature_baseline', baseline_p), ('four_epoch_primary', old_p)):
        paired[name] = paired_fingerprint_bootstrap(truth, probabilities, p,
            validation.split_group_id.to_numpy(), repeats=2000, seed=20261106)
        paired[name]['direction'] = 'extended_schedule_minus_'+name
    private_hashes = {}
    for name, part, values in (('validation', validation, logits), ('calibration', calibration, calibration_logits)):
        records = part[['image_id', 'split_group_id', 'dr_grade', 'is_group_representative']].copy()
        if name == 'calibration':
            records['calibration_role'] = lookup.calibration_role.to_numpy()
        for grade in range(5):
            records[f'logit_{grade}'] = values[:, grade].double().numpy()
        path = args.private/'scores'/f'{name}_scores.csv'
        write_or_verify(path, records.to_csv(index=False, lineterminator='\n', float_format='%.17g').encode(), args.resume)
        private_hashes[path.name] = sha256_file(path)
    sets_path = args.private/'scores/conformal_comparisons.json'
    write_or_verify(sets_path, (json.dumps(private_sets, indent=2, allow_nan=False)+'\n').encode(), args.resume)
    private_hashes[sets_path.name] = sha256_file(sets_path)
    if any(sha256_file(root/name) != expected for name, expected in snapshot.items()):
        raise ValueError('A preserved artifact changed during the experiment')
    execution_sources(root, args)
    selected_at_cap = candidate['selected_epoch'] == config.epochs
    report = {'schema_version': 1, 'spec_version': 'MASTER_SPEC_v1.0', 'experiment_id': EXPERIMENT,
        'dataset': 'APTOS_2019', 'partition': 'validation', 'scope': SCOPE,
        'model_version': candidate['model_version'], 'images': len(validation),
        'summary': {'primary_training_seed': 2026, 'completed_runs': 1, 'maximum_epochs': 15,
            'patience': 4, 'selected_epoch': candidate['selected_epoch'], 'epochs_executed': candidate['epochs_executed'],
            'stop_reason': 'maximum_epochs' if candidate['epochs_executed'] == 15 else 'four_nonimproving_selection_epochs',
            'selected_at_maximum_epoch': selected_at_cap, 'convergence_established': False,
            'conformal_comparisons': 14, 'deployment_changed': False, 'seed_variance_estimated': False},
        'population': {'partition_images': {str(k): len(v) for k, v in runtime.groupby('split')},
            'temperature_images': int(t_mask.sum()), 'conformal_representative_images': int(cp_mask.sum()),
            'selection_representative_images': int(reps.loc[validation.index].sum()),
            'patient_identity_available': False, 'quality_labels_available': False,
            'reserved_test_inference_performed': False, 'external_inference_performed': False},
        'baseline': {'report_sha256': BASELINE_SHA256, 'checkpoint_sha256': INITIAL_SHA256,
            'selection_metrics': _grading(truth, baseline_p)},
        'prior_primary': {'report_sha256': PRIOR_SHA256, 'checkpoint_sha256': old_primary['checkpoint_sha256'],
            'training_seed': 2026, 'selected_epoch': old_primary['selected_epoch'], 'selection_metrics': _grading(truth, old_p)},
        'runs': [candidate], 'grading': candidate['grading'], 'grading_before_temperature': candidate['grading_before_temperature'],
        'calibration': candidate['calibration'], 'paired_comparisons': paired,
        'comparisons': [{**row, 'model_training_seed': 2026} for row in comparisons],
        'calibration_unit_definition': CALIBRATION_UNIT,
        'source_commit': args.source_commit, 'execution_source_sha256': sources, 'input_bindings': lock,
        'pretraining_warm_start_equivalence': preflight, 'posthoc_calibration_cache': calibration_cache,
        'private_score_sha256': private_hashes, 'elapsed_seconds': time.perf_counter()-start,
        'package_versions': {name: package_metadata.version(name) for name in ('torch', 'torchvision', 'numpy', 'pandas', 'scikit-learn', 'Pillow')},
        'limitations': [
            'Exploratory source-development extension after inspecting prior development and consumed IDRiD results; design basis is source epoch-cap truncation only.',
            'Only the predeclared primary seed is extended for this CPU study; no new across-seed uncertainty is estimated.',
            'Same previously selected warm start and reused selection/calibration populations; not fresh test evidence or formal coverage validation.',
            '224px cropped RGB, frozen BatchNorm buffers and horizontal flips differ from the paper; stronger optimization is not exact reproduction or proof of convergence.',
            'Fingerprint components are not patients; label-dependent quarantine and unknown related views limit exchangeability.',
            'All five randomized seeds, raw empties, fallback and severe-grade outcomes are retained without selecting a favorable result.',
            'IDRiD is consumed and never read for this run; reserved APTOS test and production checkpoints remain unchanged.',
            'No image-quality supervision, useful validated triage, external transport or clinical safety is established.']}
    ensure_aggregate(report)
    staging = args.private/'publication/evaluation.json'
    write_or_verify(staging, (json.dumps(report, indent=2, allow_nan=False)+'\n').encode(), False)
    args.output.mkdir(parents=True, exist_ok=True)
    staging.replace(args.output/'evaluation.json')
    print(json.dumps({'status': 'completed', 'summary': report['summary'], 'grading': report['grading'],
        'elapsed_seconds': report['elapsed_seconds']}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name, default in [('manifest', 'data/manifests/aptos_v1/development.csv'),
        ('full-manifest', 'data/manifests/aptos_v1/dataset_manifest.csv'),
        ('calibration-roles', 'data/manifests/aptos_v1/calibration_roles.csv'),
        ('data-audit', 'reports/aptos_v1/data_audit.json'), ('baseline-report', 'reports/aptos_v1/evaluation.json'),
        ('prior-report', 'reports/aptos_finetune_v1/evaluation.json'),
        ('initial-checkpoint', 'checkpoints/aptos_v1/dr_efficientnet_b0/best.pt'),
        ('config', 'configs/aptos_optimization_v2.yaml'), ('protocol', 'docs/research/APTOS_OPTIMIZATION_V2_PROTOCOL.md'),
        ('private', 'artifacts/aptos_optimization_v2'), ('checkpoints', 'checkpoints/aptos_optimization_v2'),
        ('output', 'reports/aptos_optimization_v2')]:
        parser.add_argument('--'+name, type=Path, default=Path(default))
    parser.add_argument('--aptos-root', type=Path, default=default_aptos_root())
    parser.add_argument('--source-commit')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--resume', action='store_true', help='Restore only provenance-identical completed epochs/exports')
    run(parser.parse_args())


if __name__ == '__main__':
    main()
