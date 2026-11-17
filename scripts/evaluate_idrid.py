"""Locked four-model IDRiD transfer evaluation; no fitting or deployment change."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib import metadata as package_metadata
import json
from pathlib import Path
import subprocess
import sys
import time
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import torch

from qsart.data.external_evaluation import (POPULATIONS, external_pixels, external_uniforms, frozen_comparisons,
                                           grading_views, infer_logits, uncertainty_views, validate_external)
from qsart.data.idrid_provenance import idrid_source_confirmation
from qsart.models.checkpoint import load_checkpoint, sha256_file
from qsart.training.data import MEAN, STD
from qsart.training.finetune import _named_tensor_hash
from scripts.prepare_idrid import default_idrid_root
from scripts.run_aptos_finetune import paired_fingerprint_bootstrap

ROOT = Path(__file__).resolve().parents[1]
PRIVATE = ROOT/'artifacts/idrid_external_v1'
REPORT = ROOT/'reports/idrid_external_v1/evaluation.json'
CONFIG = ROOT/'configs/idrid_external_v1.json'
SOURCES = ('scripts/evaluate_idrid.py', 'qsart/data/external_evaluation.py', 'qsart/data/idrid_provenance.py',
    'qsart/data/idrid.py', 'qsart/data/manifests.py', 'qsart/data/aptos.py', 'qsart/preprocessing/fundus.py',
    'qsart/training/finetune.py', 'qsart/training/data.py', 'qsart/evaluation.py',
    'qsart/models/networks.py', 'qsart/models/checkpoint.py', 'qsart/uncertainty/conformal.py',
    'qsart/uncertainty/randomized_ordinal.py', 'scripts/prepare_idrid.py', 'scripts/run_aptos_finetune.py',
    'configs/idrid_external_v1.json', 'docs/research/IDRID_EXTERNAL_V1_PROTOCOL.md')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save_once(path, value):
    path = Path(path)
    data = json.dumps(value, indent=2, allow_nan=False)+'\n'
    if path.exists():
        if path.read_text(encoding='utf-8') != data:
            raise ValueError('Existing output differs from the exact bound evaluation')
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        staging = PRIVATE/'json_staging'/f'{uuid4().hex}.json'
        staging.parent.mkdir(parents=True, exist_ok=True)
        with staging.open('x', encoding='utf-8', newline='\n') as stream:
            stream.write(data)
        staging.rename(path)


def validate_config(config):
    fixed = {'experiment_id': 'idrid_external_v1', 'primary_model': 'finetune_seed2026',
        'primary_population': POPULATIONS[2], 'population_counts': dict(zip(POPULATIONS, (516, 513, 505))),
        'cpu_threads': 4, 'cache_workers': 2, 'batch_size': 16, 'image_size': 224,
        'bootstrap_repeats': 2000, 'bootstrap_seed': 20261105,
        'uniform_seeds': list(range(2026, 2031)), 'alphas': [.1, .05]}
    if any(config.get(key) != value for key, value in fixed.items()):
        raise ValueError('External analysis configuration differs from its fixed declaration')
    if [(row['model_id'], row['training_seed']) for row in config['models']] != [
            ('frozen_baseline', None)]+[(f'finetune_seed{s}', s) for s in range(2026, 2029)]:
        raise ValueError('All four frozen classifiers in their fixed roles are required')


def ready(config):
    validate_config(config)
    for name, digest in config['input_sha256'].items():
        if sha256_file(ROOT/name) != digest:
            raise ValueError('A frozen input report/manifest/cache changed')
    audit_path = ROOT/'reports/idrid_data_v1/data_audit.json'
    if idrid_source_confirmation(audit_path).get('source_provenance_status') != 'official_download_user_confirmed':
        raise ValueError('Valid bound user source confirmation required')
    frame = pd.read_csv(ROOT/'artifacts/idrid_review_v1/external_population.csv', dtype={'sha256': str, 'phash': str})
    validate_external(frame)
    if {name: int(frame['in_'+name].sum()) for name in POPULATIONS} != config['population_counts']:
        raise ValueError('The original frozen sensitivity population counts changed')
    source_root = default_idrid_root()
    for row in frame.itertuples():
        if not Path(row.image_path).resolve().is_relative_to(source_root) or sha256_file(row.image_path) != row.sha256:
            raise ValueError('External source location/content differs from the frozen IDRiD audit')
    comparisons = {}
    for model in config['models']:
        if (sha256_file(ROOT/model['checkpoint']) != model['checkpoint_sha256'] or
                sha256_file(ROOT/model['selection_scores']) != model['selection_scores_sha256']):
            raise ValueError('Frozen classifier or development-score bytes changed')
        source = read(ROOT/model['source_report'])
        rows = source['comparisons']
        if model['training_seed'] is not None:
            rows = [row for row in rows if row['model_training_seed'] == model['training_seed']]
            source_model = next(row for row in source['runs'] if row['training_seed'] == model['training_seed'])
        else:
            source_model = source
        if source_model['calibration']['temperature'] != model['temperature']:
            raise ValueError('The frozen source temperature changed')
        comparisons[model['model_id']] = frozen_comparisons(rows,
            checkpoint_sha256=model['checkpoint_sha256'], temperature=model['temperature'])
    return frame, comparisons


def bind_source(commit):
    if (subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip() != commit or
            subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip()):
        raise ValueError('External execution requires the declared clean source commit')
    result = {}
    for name in SOURCES:
        current = (ROOT/name).read_bytes()
        if subprocess.check_output(['git', 'show', f'{commit}:{name}'], cwd=ROOT) != current:
            raise ValueError('Execution source differs from its committed bytes')
        result[name] = hashlib.sha256(current).hexdigest()
    return result


def preflight(config, specification, model, cache, runtime):
    saved = pd.read_csv(ROOT/specification['selection_scores'])
    validation = runtime.loc[runtime.split.eq('validation')]
    if saved.image_id.tolist() != validation.image_id.tolist():
        raise ValueError('Bound development selection identities/order differ')
    chosen = saved.groupby('dr_grade', sort=True).head(2).index.to_numpy()
    if len(chosen) != 10:
        raise ValueError('Ten predetermined development preflight observations required')
    positions = {row: index for index, row in enumerate(cache['metadata']['manifest_rows'])}
    indices = [positions[int(validation.index[index])] for index in chosen]
    logits = infer_logits(model, cache['images'][indices], batch_size=config['batch_size'])
    columns = [f'logit_{grade}' for grade in range(5)]
    expected = torch.tensor(saved.iloc[chosen][columns].to_numpy(float)).softmax(1).numpy()
    actual = logits.double().softmax(1).numpy()
    delta = float(np.abs(actual-expected).max())
    if delta > 1e-5 or not np.array_equal(actual.argmax(1), expected.argmax(1)):
        raise ValueError('Predictor preprocessing regression failed before external inference')
    return {'images': 10, 'raw_probability_max_abs_delta': delta, 'argmax_predictions_match': True, 'tolerance': 1e-5}


def export_scores(path, logits):
    if Path(path).exists():
        raise ValueError('Preserve existing external score exports')
    scores = pd.DataFrame({'manifest_row': range(len(logits))})
    for grade in range(5):
        # Preserve the exact float32 values in decimal, rather than float32's short CSV rendering.
        scores[f'logit_{grade}'] = logits[:, grade].double().numpy()
    scores.to_csv(path, index=False, lineterminator='\n')


def import_scores(path, n):
    saved = pd.read_csv(path, float_precision='round_trip')
    if saved.columns.tolist() != ['manifest_row']+[f'logit_{grade}' for grade in range(5)]:
        raise ValueError('External score export columns differ')
    values = saved[[f'logit_{grade}' for grade in range(5)]].to_numpy(float)
    if (saved.manifest_row.tolist() != list(range(n)) or values.shape != (n, 5) or not np.isfinite(values).all()):
        raise ValueError('External saved logits must be finite, complete and in original order')
    return torch.tensor(values)


def candidate(config, specification, pixels, cache_info, preflight_cache, runtime, resume):
    folder = PRIVATE/'models'/specification['model_id']
    binding = {'source_commit': config['execution_source_commit'], 'model': specification,
        'external_cache_sha256': cache_info['cache_file_sha256'],
        'population_sha256': config['input_sha256']['artifacts/idrid_review_v1/external_population.csv']}
    if folder.exists():
        if not resume:
            raise ValueError('Completed external predictor requires explicit provenance-checked reuse')
        proof = read(folder/'verification.json')
        if proof['binding'] != binding or sha256_file(folder/'scores.csv') != proof['scores_sha256']:
            raise ValueError('Existing external predictor export binding mismatch')
        return import_scores(folder/'scores.csv', len(pixels)), proof
    model, checkpoint = load_checkpoint(ROOT/specification['checkpoint'])
    metadata = checkpoint['metadata']
    if (metadata['backbone'] != 'efficientnet_b0' or metadata['heads'] != {'dr_grade': 5} or
            metadata['image_size'] != 224 or metadata['normalization_mean'] != MEAN or metadata['normalization_std'] != STD):
        raise ValueError('Frozen external predictor architecture/preprocessing mismatch')
    state_before = _named_tensor_hash(model.state_dict().items())
    regression = preflight(config, specification, model, preflight_cache, runtime)
    consumption = PRIVATE/'consumption.json'
    if not consumption.exists():
        save_once(consumption, {'external_outcomes_may_be_consumed': True, 'source_commit': config['execution_source_commit'],
            'started_at_utc': datetime.now(timezone.utc).isoformat(), 'scope': 'locked external inference only'})
    logits = infer_logits(model, pixels, batch_size=config['batch_size'])
    state_after = _named_tensor_hash(model.state_dict().items())
    if state_before != state_after:
        raise ValueError('Predictor state changed during external evaluation')
    staging = PRIVATE/'models'/('pending_'+specification['model_id']+'_'+uuid4().hex)
    staging.mkdir(parents=True)
    export_scores(staging/'scores.csv', logits)
    proof = {'binding': binding, 'preflight': regression, 'scores_sha256': sha256_file(staging/'scores.csv'),
        'predictor_state_before_sha256': state_before, 'predictor_state_after_sha256': state_after,
        'model_version': metadata['model_version'], 'data_purpose': metadata['data_purpose']}
    save_once(staging/'verification.json', proof)
    staging.rename(folder)
    print(json.dumps({'event': 'external_predictor_complete', 'model_id': specification['model_id'],
                      'observations': len(pixels), 'preflight': regression}), flush=True)
    return logits, proof


def run(args):
    config = read(CONFIG)
    frame, artifacts = ready(config)
    if not args.execute:
        print(json.dumps({'status': 'ready_inputs_only', 'external_inference_performed': False,
            'models': [row['model_id'] for row in config['models']], 'populations': config['population_counts']}, indent=2))
        return
    if not args.source_commit:
        raise ValueError('--execute requires a frozen source commit')
    if REPORT.exists():
        raise ValueError('Preserve the completed external evaluation; outcomes are consumed')
    sources = bind_source(args.source_commit)
    lock = {'source_commit': args.source_commit, 'sources': sources, 'config': config}
    if PRIVATE.exists() and (not args.resume or not (PRIVATE/'execution_lock.json').is_file()):
        raise ValueError('Partial external evaluation requires explicit identical-source resume')
    PRIVATE.mkdir(parents=True, exist_ok=True)
    save_once(PRIVATE/'execution_lock.json', lock)
    torch.set_num_threads(config['cpu_threads'])
    started = time.perf_counter()
    pixels, cache_info = external_pixels(frame, PRIVATE/'external_uint8.pt', workers=config['cache_workers'])
    save_once(PRIVATE/'cache_binding.json', cache_info)
    preflight_cache = torch.load(ROOT/config['preflight_cache'], map_location='cpu', weights_only=True)
    runtime = pd.read_csv(ROOT/config['preflight_runtime_manifest'])
    uniforms = external_uniforms(len(frame))
    save_once(PRIVATE/'external_uniforms.json', {str(seed): values.tolist() for seed, values in uniforms.items()})
    models, comparisons, probabilities, private_hashes = [], [], {}, {}
    for specification in config['models']:
        model_id = specification['model_id']
        logits, proof = candidate({**config, 'execution_source_commit': args.source_commit}, specification,
                                 pixels, cache_info, preflight_cache, runtime, args.resume)
        raw_p = logits.double().softmax(1).numpy()
        p = (logits.double()/specification['temperature']).softmax(1).numpy()
        probabilities[model_id] = p
        cp, individual = uncertainty_views(frame, p, artifacts[model_id], uniforms)
        comparisons.extend({'model_id': model_id, **row} for row in cp)
        save_once(PRIVATE/'models'/model_id/'sets.json', individual)
        private_hashes[model_id] = {name: sha256_file(PRIVATE/'models'/model_id/name)
                                   for name in ('scores.csv', 'verification.json', 'sets.json')}
        models.append({'model_id': model_id, 'training_seed': specification['training_seed'],
            'checkpoint_sha256': specification['checkpoint_sha256'], 'temperature': specification['temperature'],
            'model_version': proof['model_version'], 'grading_before_temperature': grading_views(frame, raw_p),
            'grading': grading_views(frame, p), 'preflight': proof['preflight'],
            'predictor_state_unchanged': proof['predictor_state_before_sha256'] == proof['predictor_state_after_sha256']})
    paired = []
    for model in config['models'][1:]:
        for name in POPULATIONS:
            mask = frame['in_'+name].to_numpy(bool)
            estimate = paired_fingerprint_bootstrap(frame.loc[mask, 'dr_grade'].to_numpy(int),
                probabilities['frozen_baseline'][mask], probabilities[model['model_id']][mask],
                frame.loc[mask, 'split_group_id'].to_numpy(), repeats=config['bootstrap_repeats'], seed=config['bootstrap_seed'])
            paired.append({'model_id': model['model_id'], 'population': name, **estimate})
    across = {}
    for name in POPULATIONS:
        across[name] = {metric: {'mean': float(np.mean([row['grading'][name][metric] for row in models[1:]])),
            'sample_standard_deviation': float(np.std([row['grading'][name][metric] for row in models[1:]], ddof=1))}
            for metric in ('accuracy', 'macro_f1', 'quadratic_weighted_kappa', 'nll', 'ece', 'brier')}
    for name, expected in config['input_sha256'].items():
        if sha256_file(ROOT/name) != expected:
            raise ValueError('A frozen source/input artifact changed during evaluation')
    for specification in config['models']:
        if sha256_file(ROOT/specification['checkpoint']) != specification['checkpoint_sha256']:
            raise ValueError('A frozen checkpoint changed during evaluation')
    primary = next(row for row in models if row['model_id'] == config['primary_model'])
    report = {'schema_version': 1, 'spec_version': 'MASTER_SPEC_v1.0', 'experiment_id': 'idrid_external_v1',
        'dataset': 'IDRiD', 'partition': 'external_holdout', 'scope': 'locked_external_domain_diagnostics_consumed',
        'images': len(frame), 'model_version': primary['model_version'],
        'summary': {'primary_model': config['primary_model'], 'primary_population': config['primary_population'],
            'population_counts': config['population_counts'], 'models_evaluated': len(models),
            'conformal_comparisons': len(comparisons), 'raw_fallback_sets': 2*len(frame)*len(comparisons),
            'external_model_observations': len(frame)*len(models), 'paired_comparisons': len(paired),
            'grading_across_finetuning_seeds': across, 'deployment_changed': False},
        'models': models, 'grading': primary['grading'][config['primary_population']],
        'comparisons': comparisons, 'paired_baseline_changes': paired,
        'population': {'original_images': 516, 'views': read(ROOT/'reports/idrid_review_v1/data_review.json')['summary']['populations'],
            'patient_identity_available': False, 'quality_labels_available': False},
        'input_bindings': lock, 'external_cache': cache_info, 'private_output_sha256': private_hashes,
        'uniforms_sha256': sha256_file(PRIVATE/'external_uniforms.json'),
        'uniform_generation': 'SeedSequence(seed).spawn(3)[2]; common external uniforms across models and alpha; source children0/1 remain separate.',
        'source_commit': args.source_commit, 'execution_source_sha256': sources,
        'external_outcomes_consumed': True, 'consumption_started_at_utc': read(PRIVATE/'consumption.json')['started_at_utc'],
        'classifier_fitting_performed': False, 'temperature_fitting_performed': False,
        'conformal_threshold_fitting_performed': False, 'aptos_reserved_test_inference_performed': False,
        'elapsed_seconds': time.perf_counter()-started,
        'package_versions': {name: package_metadata.version(name) for name in ('torch', 'torchvision', 'numpy', 'pandas', 'scikit-learn', 'Pillow')},
        'limitations': ['First locked external model outcomes; now consumed, with future reuse disclosed and no outcome-driven tuning.',
            'Filtered fingerprint representatives are not patients; unknown related views, undetected overlap and source-label ambiguity remain.',
            'Full516 contains unresolved cross-source candidates; all full/filtered populations are reported without a favorable-view selection.',
            'Four grade-conflicting components remain; annotation envelopes assume an observed component label is appropriate, not clinical ground truth.',
            'All fine-tuning seeds share a selected source initialization; SD is conditional variation, not independent cohorts or full-pipeline uncertainty.',
            'Source-only temperatures/thresholds are frozen; no marginal, grade-conditional or patient coverage guarantee transfers under shift.',
            'Paired fixed-component bootstrap intervals are descriptive, unadjusted and do not establish patient-level or clinical superiority.',
            'No quality, DME prediction, triage utility, clinical validation, exact-paper reproduction or deployment improvement is established.']}
    if len(models) != 4 or len(comparisons) != 56 or len(paired) != 9:
        raise ValueError('Declared external evaluation grid is incomplete')
    serialized = json.dumps(report, allow_nan=False)
    if any(token in serialized for token in ('"image_id"', '"image_path"', '"subject_id"', str(ROOT), 'C:\\Users\\')):
        raise ValueError('Private identities or paths in public external aggregates')
    save_once(REPORT, report)
    print(json.dumps({'status': 'completed_external_outcomes_consumed', 'summary': report['summary'],
                      'primary_grading': {key: report['grading'][key] for key in ('accuracy', 'macro_f1', 'nll')}}, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--source-commit')
    parser.add_argument('--resume', action='store_true')
    arguments = parser.parse_args()
    run(arguments)
