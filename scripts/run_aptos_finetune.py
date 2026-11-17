"""Execute the declared three-seed APTOS fine-tuning development study.

Default mode checks frozen inputs only. No reserved-test prediction or automatic
deployment is supported; actual fitting requires committed source and --execute.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from concurrent.futures import ThreadPoolExecutor
import hashlib
from importlib import metadata as package_metadata
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd
from PIL import Image
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qsart.calibration.temperature import calibration_metrics, fit_temperature
from qsart.models.checkpoint import load_checkpoint, sha256_file
from qsart.preprocessing import preprocess_image
from qsart.training.data import split_fingerprint
from qsart.training.finetune import (FinetuneConfig, batchnorm_buffer_hash, normalize_cached,
                                     parameter_hash, prepare_input_cache, source_provenance, train_finetune)
from qsart.training.transfer import _input_content_hash
from scripts.inspect_datasets import default_aptos_root
from scripts.run_aptos_development import (CALIBRATION_UNIT, PRESERVED, _bools, _grading,
    conformal_comparisons, read_json, validate_development,
    validate_inputs as validate_frozen_inputs, write_json)

TRAINING_SEEDS = (2026, 2027, 2028)
PRIMARY_SEED = 2026
INITIAL_SHA256 = '52574e61db4b68308348aab3794bcc64c55f6d47b0d02932955286c35d190407'
BASELINE_SHA256 = '994a8e78148d6c16d601d61d8774e0d36070b653ebf5549dec093ecbad4924ec'
DEVELOPMENT_SHA256 = '681799c7ee5f04e988dc607f6cf296fd882c5f55f2603dcd0a2ccd8148951fab'
SOURCE_PATHS = ('qsart/training/finetune.py', 'qsart/training/data.py', 'qsart/training/trainer.py',
    'qsart/models/networks.py', 'qsart/models/checkpoint.py', 'qsart/preprocessing/fundus.py', 'qsart/training/transfer.py',
    'qsart/calibration/temperature.py', 'qsart/uncertainty/conformal.py',
    'qsart/uncertainty/randomized_ordinal.py', 'scripts/run_aptos_development.py',
    'scripts/inspect_datasets.py', 'scripts/run_aptos_finetune.py', 'scripts/plot_aptos_finetune.py')


def source_bindings(root, args, commit):
    """Require the declared source bytes, not merely a matching commit string."""
    actual = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    if actual != commit or subprocess.check_output(['git', 'status', '--porcelain'], cwd=root, text=True).strip():
        raise ValueError('Fine-tuning requires the declared clean committed source')
    paths = list(SOURCE_PATHS)+[Path(args.protocol).resolve().relative_to(root).as_posix(),
                                Path(args.config).resolve().relative_to(root).as_posix()]
    result = {}
    for name in dict.fromkeys(paths):
        current = (root/name).read_bytes()
        committed = subprocess.check_output(['git', 'show', f'{commit}:{name}'], cwd=root)
        if current != committed:
            raise ValueError(f'Execution source differs from its frozen commit: {name}')
        result[name] = hashlib.sha256(current).hexdigest()
    return result


def runtime_mapping(frame, roles, source_root):
    """Map the same identities to the active source without changing split/labels."""
    source_root = Path(source_root).resolve()
    mapped, mapped_roles = frame.copy(), roles.copy()
    for part in (mapped, mapped_roles):
        if not part.image_id.astype(str).str.fullmatch(r'[A-Za-z0-9_-]+').all():
            raise ValueError('Safe source image IDs are required before runtime mapping')
        paths = [(source_root/'train_images'/f'{identity}.png').resolve() for identity in part.image_id]
        if any(not path.is_relative_to(source_root) for path in paths):
            raise ValueError('Runtime source mapping escapes the configured source root')
        part['image_path'] = [str(path) for path in paths]
    validate_development(mapped, mapped_roles)
    return mapped, mapped_roles


def fixed_configs(path):
    actual = yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    expected = asdict(FinetuneConfig())
    if actual != expected:
        raise ValueError('Fine-tuning configuration differs from the declared fixed settings')
    return {seed: replace(FinetuneConfig(**actual), seed=seed) for seed in TRAINING_SEEDS}


def _metric_pair(truth, predictions):
    matrix = np.bincount(5*np.asarray(truth, int)+np.asarray(predictions, int), minlength=25).reshape(5, 5)
    diagonal = np.diag(matrix)
    denominator = matrix.sum(0)+matrix.sum(1)
    f1 = np.divide(2*diagonal, denominator, out=np.zeros(5, dtype=float), where=denominator > 0)
    return float(diagonal.sum()/matrix.sum()), float(f1.mean())


def paired_fingerprint_bootstrap(truth, baseline_p, candidate_p, groups, *, repeats=1000, seed=20261104):
    """Descriptive paired uncertainty conditional on the fixed selected classifiers."""
    truth, groups = np.asarray(truth, int), np.asarray(groups)
    baseline, candidate = np.argmax(baseline_p, 1), np.argmax(candidate_p, 1)
    if not (truth.shape == groups.shape == baseline.shape == candidate.shape) or not len(truth):
        raise ValueError('Paired observations must have matching nonempty identities and outcomes')
    unique = np.unique(groups)
    members = [np.flatnonzero(groups == group) for group in unique]
    if len(unique) < 2 or repeats < 1:
        raise ValueError('At least two fingerprint groups and one resample are required')
    rng = np.random.default_rng(seed)
    deltas = []
    for draw in rng.integers(0, len(unique), size=(repeats, len(unique))):
        indices = np.concatenate([members[index] for index in draw])
        first = _metric_pair(truth[indices], baseline[indices])
        second = _metric_pair(truth[indices], candidate[indices])
        deltas.append(np.subtract(second, first))
    observed = np.subtract(_metric_pair(truth, candidate), _metric_pair(truth, baseline))
    lower, upper = np.quantile(deltas, [.025, .975], axis=0)
    return {'images': len(truth), 'fingerprint_groups': len(unique), 'repeats': repeats, 'seed': seed,
        'method': 'paired_fingerprint_component_percentile_bootstrap_fixed_predictions',
        'direction': 'fine_tuned_minus_frozen_feature_baseline', 'confidence_level': .95,
        'accuracy': {'delta': float(observed[0]), 'lower': float(lower[0]), 'upper': float(upper[0])},
        'macro_f1': {'delta': float(observed[1]), 'lower': float(lower[1]), 'upper': float(upper[1])},
        'limitations': 'Fixed selected predictions; no refitting, multiplicity adjustment, patient independence or full training uncertainty.'}


def seed_summary(records, metric_field='grading'):
    """Training-seed variability is separate from conformal uniform variability."""
    if [row['training_seed'] for row in records] != list(TRAINING_SEEDS):
        raise ValueError('All declared training seeds must remain in the fixed order')
    summary = {}
    for key in ('accuracy', 'macro_f1', 'quadratic_weighted_kappa', 'nll', 'ece', 'brier'):
        values = [row[metric_field][key] for row in records]
        summary[key] = {'mean': float(np.mean(values)), 'sample_standard_deviation': float(np.std(values, ddof=1))}
    for grade in range(5):
        values = [row[metric_field]['per_class'][grade]['recall'] for row in records]
        summary[f'grade_{grade}_recall'] = {'mean': float(np.mean(values)),
                                         'sample_standard_deviation': float(np.std(values, ddof=1))}
    return summary


def cached_logits(model, frame, manifest, cache_dir, config):
    if not frame.split.isin(['train', 'validation']).all():
        raise ValueError('Classifier cache inference accepts fitting/selection records only')
    cache, info = prepare_input_cache(frame, manifest, cache_dir, image_size=config.image_size,
                                      workers=config.cache_workers)
    model.eval()
    if config.channels_last:
        model.to(memory_format=torch.channels_last)
    values = []
    selected_images = cache['images'][torch.tensor(frame.split.eq('validation').to_numpy())]
    with torch.inference_mode():
        for images in selected_images.split(config.batch_size):
            values.append(model(normalize_cached(images))['dr_grade'].detach().cpu())
    return torch.cat(values), info


def scoring_cache(frame, manifest, path, config):
    """Separate deterministic calibration pixels, never accepted by the trainer."""
    if frame.empty or not frame.split.eq('calibration').all():
        raise ValueError('Post-hoc scoring cache accepts calibration observations only')
    content = _input_content_hash(frame)
    metadata = {'scope': 'posthoc_calibration_scoring_only', 'manifest_sha256': sha256_file(manifest),
        'input_content_sha256': content, 'image_size': config.image_size,
        'source_sha256': source_provenance()['source_sha256'], 'manifest_rows': frame.index.tolist()}
    path = Path(path)
    if path.exists():
        cached = torch.load(path, map_location='cpu', weights_only=True)
        if cached.get('metadata') != metadata:
            raise ValueError('Calibration scoring cache provenance differs')
    else:
        pixels = torch.empty((len(frame), config.image_size, config.image_size, 3), dtype=torch.uint8)

        def canonical(path):
            with Image.open(path) as handle:
                return np.asarray(preprocess_image(handle, image_size=config.image_size, crop=True).convert('RGB')).copy()

        with ThreadPoolExecutor(max_workers=config.cache_workers) as pool:
            for index, values in enumerate(pool.map(canonical, frame.image_path)):
                pixels[index] = torch.from_numpy(values)
        cached = {'metadata': metadata, 'images': pixels,
                  'tensor_sha256': hashlib.sha256(pixels.numpy().tobytes()).hexdigest()}
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(cached, path)
    pixels = cached['images']
    if (pixels.dtype != torch.uint8 or tuple(pixels.shape) != (len(frame), config.image_size, config.image_size, 3)
            or hashlib.sha256(pixels.numpy().tobytes()).hexdigest() != cached['tensor_sha256']):
        raise ValueError('Calibration scoring pixels changed')
    return pixels, {'scope': metadata['scope'], 'images': len(frame), 'file_sha256': sha256_file(path),
                    'tensor_sha256': cached['tensor_sha256'], 'input_content_sha256': content}


def posthoc_logits(model, pixels, config):
    model.eval()
    if config.channels_last:
        model.to(memory_format=torch.channels_last)
    output = []
    with torch.inference_mode():
        for batch in pixels.split(config.batch_size):
            values = normalize_cached(batch)
            if config.channels_last:
                values = values.contiguous(memory_format=torch.channels_last)
            output.append(model(values)['dr_grade'].detach().cpu())
    return torch.cat(output)


def candidate_record(folder, config, runtime, runtime_manifest, cache_dir, expected_source):
    required = ('best.pt', 'last.pt', 'training_result.json', 'training_metadata.json',
                'selection_predictions.csv', 'training_history.csv', 'selection_metrics.json',
                'selection_epoch_predictions.csv')
    if not all((folder/name).is_file() for name in required):
        raise ValueError('Only completed, provenance-identical training runs may be reused')
    model, checkpoint = load_checkpoint(folder/'best.pt')
    metadata = checkpoint['metadata']
    if metadata != read_json(folder/'training_metadata.json'):
        raise ValueError('Checkpoint and exported fine-tuning metadata differ')
    for key, value in {'seed': config.seed, 'training_config': asdict(config),
                       'initial_checkpoint_sha256': INITIAL_SHA256,
                       'manifest_sha256': sha256_file(runtime_manifest),
                       'code_provenance': expected_source}.items():
        if metadata.get(key) != value:
            raise ValueError(f'Fine-tuned candidate metadata mismatch: {key}')
    if (metadata.get('backbone_parameters_changed') is not True or metadata.get('batchnorm_buffers_unchanged') is not True
            or metadata['initial_backbone_parameters_sha256'] == metadata['backbone_parameters_sha256']
            or metadata['initial_batchnorm_buffers_sha256'] != metadata['batchnorm_buffers_sha256']
            or parameter_hash(model.backbone) != metadata['backbone_parameters_sha256']
            or batchnorm_buffer_hash(model) != metadata['batchnorm_buffers_sha256']):
        raise ValueError('Saved fine-tuned backbone/BatchNorm changes do not match the declared strategy')
    validation = runtime.loc[runtime.split.eq('validation')]
    predictions = pd.read_csv(folder/'selection_predictions.csv')
    if (not np.array_equal(predictions.manifest_row, validation.index) or
            not np.array_equal(predictions.true_dr_grade, validation.dr_grade) or
            'epoch' not in predictions or not predictions.epoch.eq(metadata['epoch']).all()):
        raise ValueError('Fine-tuned selection predictions differ from frozen cohort order/labels')
    probabilities = predictions[[f'dr_grade_p{i}' for i in range(5)]].to_numpy(float)
    if (not np.isfinite(probabilities).all() or (probabilities < 0).any() or
            not np.allclose(probabilities.sum(1), 1, atol=1e-5, rtol=0)):
        raise ValueError('Fine-tuned selection probabilities are invalid')
    probabilities /= probabilities.sum(1, keepdims=True)
    selection_metrics = _grading(validation.dr_grade.to_numpy(int), probabilities)
    exported = read_json(folder/'selection_metrics.json')
    if 'dr_grade' in exported:
        exported = exported['dr_grade']
    for key in ('accuracy', 'macro_f1', 'nll'):
        if not np.isclose(selection_metrics[key], exported[key], atol=1e-5, rtol=0):
            raise ValueError('Fine-tuned grading differs from selected checkpoint metrics')
    history = pd.read_csv(folder/'training_history.csv')
    result = read_json(folder/'training_result.json')
    if len(history) != result['epochs_executed'] or result['selected_epoch'] != metadata['epoch']:
        raise ValueError('Fine-tuning history, selected epoch and training result disagree')
    # Reconstruct predictions from source-bound canonical inputs, with no extra crop.
    selected = runtime.loc[runtime.split.isin(['train', 'validation'])]
    logits, cache_info = cached_logits(model, selected, runtime_manifest, cache_dir, config)
    recorded_cache = metadata['canonical_cache_provenance']
    if (metadata['canonical_cache_key'] != cache_info['cache_key'] or
            recorded_cache['cache_file_sha256'] != cache_info['cache_file_sha256'] or
            recorded_cache['tensor_sha256'] != cache_info['tensor_sha256'] or
            recorded_cache['metadata'] != cache_info['metadata']):
        raise ValueError('Fine-tuned checkpoint and canonical input cache bindings differ')
    difference = float(np.max(np.abs(logits.softmax(1).numpy()-probabilities)))
    if difference > 2e-6:
        raise ValueError('Selected model differs from its saved genuine selection probabilities')
    public = {'training_seed': config.seed, 'checkpoint_sha256': sha256_file(folder/'best.pt'),
        'last_checkpoint_sha256': sha256_file(folder/'last.pt'), 'checkpoint_bytes': (folder/'best.pt').stat().st_size,
        'model_version': metadata['model_version'], 'selected_epoch': result['selected_epoch'],
        'epochs_executed': result['epochs_executed'], 'optimizer_steps': result['total_optimizer_steps'],
        'training_config': asdict(config), 'selection_metrics': selection_metrics, 'history': history.to_dict('records'),
        'training_strategy': metadata['training_strategy'], 'initial_checkpoint_sha256': INITIAL_SHA256,
        'backbone_parameters_changed': metadata['backbone_parameters_changed'],
        'batchnorm_buffers_unchanged': metadata['batchnorm_buffers_unchanged'],
        'canonical_cache_key': metadata['canonical_cache_key'], 'canonical_cache_sha256': cache_info['cache_file_sha256'],
        'canonical_pixel_tensor_sha256': cache_info['tensor_sha256'],
        'initial_backbone_parameters_sha256': metadata['initial_backbone_parameters_sha256'],
        'backbone_parameters_sha256': metadata['backbone_parameters_sha256'],
        'initial_batchnorm_buffers_sha256': metadata['initial_batchnorm_buffers_sha256'],
        'batchnorm_buffers_sha256': metadata['batchnorm_buffers_sha256'],
        'selection_probability_equivalence_max_abs_delta': difference,
        'artifact_sha256': {name: sha256_file(folder/name) for name in required}}
    return public, model, checkpoint, logits


def validate_ready(args):
    if sha256_file(args.manifest) != DEVELOPMENT_SHA256 or sha256_file(args.baseline_report) != BASELINE_SHA256:
        raise ValueError('Original development manifest or completed baseline report changed')
    if sha256_file(args.initial_checkpoint) != INITIAL_SHA256:
        raise ValueError('The declared common warm-start checkpoint changed')
    frame, roles, reps, bindings = validate_frozen_inputs(args)
    baseline = read_json(args.baseline_report)
    if baseline['summary']['selected_checkpoint_sha256'] != INITIAL_SHA256:
        raise ValueError('Baseline report no longer binds the declared warm start')
    source_root = Path(args.aptos_root).resolve()
    if sha256_file(source_root/'train.csv') != read_json(args.data_audit)['source_annotation_sha256']:
        raise ValueError('Configured active source annotations differ from validated source')
    runtime, runtime_roles = runtime_mapping(frame, roles, source_root)
    if not runtime.image_path.map(lambda path: Path(path).is_file()).all():
        raise ValueError('Configured source is missing required development images')
    return frame, roles, runtime, runtime_roles, reps, bindings, baseline


def run(args):
    root = Path(__file__).resolve().parents[1]
    original, roles, runtime, runtime_roles, reps, bindings, baseline = validate_ready(args)
    configs = fixed_configs(args.config)
    if not args.execute:
        print(json.dumps({'status': 'ready_inputs_only', 'training_or_inference_performed': False,
            'population': {str(name): len(part) for name, part in runtime.groupby('split')},
            'training_seeds': list(TRAINING_SEEDS), 'primary_training_seed': PRIMARY_SEED,
            'original_bindings': bindings}, indent=2))
        return
    if not args.source_commit:
        raise ValueError('--execute requires the committed --source-commit')
    if (args.output/'evaluation.json').exists():
        raise ValueError('Preserve the completed fine-tuning study')
    for path, allowed in ((args.private, root/'artifacts'), (args.checkpoints, root/'checkpoints')):
        if not path.resolve().is_relative_to(allowed):
            raise ValueError('Private data/checkpoints must remain under ignored artifact trees')
    execution_sources = source_bindings(root, args, args.source_commit)
    preservation_paths = {name: root/name for name in PRESERVED}
    preservation_paths['aptos_baseline_report'] = args.baseline_report
    for previous in baseline['candidates']:
        folder = args.initial_checkpoint.parent.parent/f"dr_{previous['backbone']}"
        for name, expected in previous['artifact_sha256'].items():
            path = folder/name
            if sha256_file(path) != expected:
                raise ValueError('An original APTOS candidate artifact changed')
            preservation_paths[f"aptos_{previous['backbone']}_{name}"] = path
    preservation = {name: sha256_file(path) for name, path in preservation_paths.items()}
    source = source_provenance()
    content_hash = _input_content_hash(runtime)
    args.private.mkdir(parents=True, exist_ok=True)
    runtime_manifest = args.private/'runtime_development.csv'
    runtime_roles_path = args.private/'runtime_calibration_roles.csv'
    for path, records in ((runtime_manifest, runtime), (runtime_roles_path, runtime_roles)):
        if path.exists():
            if not (args.resume_completed or args.resume_last):
                raise ValueError('Existing runtime mappings require explicit verified completed-run reuse')
            comparison = pd.read_csv(path)
            if not comparison.astype(str).equals(records.reset_index(drop=True).astype(str)):
                raise ValueError('Existing runtime mapping differs from the current active source')
        else:
            records.to_csv(path, index=False, lineterminator='\n')
    lock = {'source_commit': args.source_commit, 'execution_sources': execution_sources,
        'original_bindings': bindings, 'runtime_manifest_sha256': sha256_file(runtime_manifest),
        'runtime_roles_sha256': sha256_file(runtime_roles_path), 'development_content_sha256': content_hash,
        'initial_checkpoint_sha256': INITIAL_SHA256, 'baseline_report_sha256': BASELINE_SHA256,
        'preserved_artifact_sha256': preservation, 'configurations': {str(seed): asdict(c) for seed, c in configs.items()}}
    lock_path = args.private/'execution_lock.json'
    if lock_path.exists():
        if not (args.resume_completed or args.resume_last) or read_json(lock_path) != lock:
            raise ValueError('Existing execution lock differs from the declared source/input state')
    else:
        write_json(lock_path, lock)
    validation = runtime.loc[runtime.split.eq('validation')]
    calibration = runtime.loc[runtime.split.eq('calibration')]
    baseline_predictions = pd.read_csv(args.initial_checkpoint.parent/'selection_predictions.csv')
    baseline_candidate = next(row for row in baseline['candidates'] if row['checkpoint_sha256'] == INITIAL_SHA256)
    if sha256_file(args.initial_checkpoint.parent/'selection_predictions.csv') != baseline_candidate['artifact_sha256']['selection_predictions.csv']:
        raise ValueError('Saved baseline predictions changed')
    if (not np.array_equal(baseline_predictions.manifest_row, validation.index) or
            not np.array_equal(baseline_predictions.true_dr_grade, validation.dr_grade)):
        raise ValueError('Baseline and fine-tuning selection cohorts are not paired')
    baseline_p = baseline_predictions[[f'dr_grade_p{i}' for i in range(5)]].to_numpy(float)
    baseline_p /= baseline_p.sum(1, keepdims=True)
    cache_dir = args.private/'inputs'
    torch.set_num_threads(configs[PRIMARY_SEED].cpu_threads)
    warm_started = time.perf_counter()
    warm_model, _ = load_checkpoint(args.initial_checkpoint)
    warm_logits, warm_cache_info = cached_logits(warm_model,
        runtime.loc[runtime.split.isin(['train', 'validation'])], runtime_manifest, cache_dir, configs[PRIMARY_SEED])
    warm_p = warm_logits.softmax(1).numpy()
    warm_difference = float(np.max(np.abs(warm_p-baseline_p)))
    warm_argmax_agreement = bool(np.array_equal(warm_p.argmax(1), baseline_p.argmax(1)))
    if warm_difference > 1e-5 or not warm_argmax_agreement:
        raise ValueError('Canonical preprocessing differs from the original bound warm-start predictions')
    preflight = {'images': len(validation), 'max_abs_probability_delta': warm_difference,
        'argmax_predictions_match': warm_argmax_agreement, 'tolerance': 1e-5,
        'canonical_cache_sha256': warm_cache_info['cache_file_sha256'],
        'canonical_pixel_tensor_sha256': warm_cache_info['tensor_sha256'],
        'elapsed_seconds': time.perf_counter()-warm_started,
        'cache_preparation_or_verification_seconds': warm_cache_info['preparation_or_verification_seconds'],
        'scope': 'Pre-optimizer development equivalence check, not fresh test evaluation'}
    print(json.dumps({'event': 'warm_start_preprocessing_verified', **preflight}), flush=True)
    del warm_model, warm_logits, warm_p
    models, runs = {}, []
    started = time.perf_counter()
    for seed, config in configs.items():
        source_bindings(root, args, args.source_commit)
        folder = args.checkpoints/f'seed{seed}'
        begin = time.perf_counter()
        if folder.exists():
            if not (args.resume_completed or args.resume_last):
                raise ValueError('Existing completed candidates require explicit provenance verification')
            if not (folder/'training_result.json').is_file():
                if not args.resume_last:
                    raise ValueError('Partial candidate requires --resume-last with exact provenance')
                train_finetune(runtime_manifest, args.initial_checkpoint, folder, cache_dir,
                               config=config, resume_last=True)
        else:
            train_finetune(runtime_manifest, args.initial_checkpoint, folder, cache_dir, config=config)
        row, model, checkpoint, logits = candidate_record(folder, config, runtime, runtime_manifest, cache_dir, source)
        row['training_and_verification_seconds'] = time.perf_counter()-begin
        runs.append(row)
        models[seed] = model, checkpoint, logits
    comparisons, private_hashes = [], {}
    calibration_pixels, calibration_cache_info = scoring_cache(calibration, runtime_manifest,
        args.private/'posthoc_inputs/calibration_uint8.pt', configs[PRIMARY_SEED])
    truth = validation.dr_grade.to_numpy(int)
    role_lookup = runtime_roles.set_index('image_id').loc[calibration.image_id]
    temperature_mask = role_lookup.calibration_role.eq('temperature').to_numpy()
    cp_mask = (role_lookup.calibration_role.eq('conformal') & _bools(role_lookup.is_group_representative)).to_numpy()
    calibration_truth = torch.tensor(calibration.dr_grade.to_numpy(), dtype=torch.long)
    for row in runs:
        seed = row['training_seed']
        config = configs[seed]
        model, checkpoint, validation_logits = models[seed]
        calibration_logits = posthoc_logits(model, calibration_pixels, config)
        temperature = fit_temperature(calibration_logits[temperature_mask], calibration_truth[temperature_mask])
        selection_p = (validation_logits.double()/temperature).softmax(1).numpy()
        cp_p = (calibration_logits[cp_mask].double()/temperature).softmax(1).numpy()
        row['grading_before_temperature'] = _grading(truth, validation_logits.double().softmax(1).numpy())
        row['grading'] = _grading(truth, selection_p)
        row['calibration'] = {'temperature': temperature, 'ece_bins': 15,
            'temperature_fit_before': calibration_metrics(calibration_logits[temperature_mask], calibration_truth[temperature_mask]),
            'temperature_fit_after': calibration_metrics(calibration_logits[temperature_mask], calibration_truth[temperature_mask], temperature),
            'selection_before': calibration_metrics(validation_logits, truth),
            'selection_after': calibration_metrics(validation_logits, truth, temperature)}
        row['paired_baseline_changes'] = paired_fingerprint_bootstrap(truth, baseline_p, selection_p,
                                                                      validation.split_group_id.to_numpy())
        binding = {'checkpoint_sha256': row['checkpoint_sha256'], 'model_training_seed': seed,
            'original_development_manifest_sha256': DEVELOPMENT_SHA256,
            'runtime_manifest_sha256': sha256_file(runtime_manifest), 'independent_temperature': temperature,
            'temperature_split_fingerprint': split_fingerprint(calibration.loc[temperature_mask]),
            'conformal_representatives_fingerprint': split_fingerprint(calibration.loc[cp_mask])}
        public_cp, private_cp = conformal_comparisons(cp_p, calibration_truth[cp_mask].numpy(), selection_p,
            truth, reps.loc[validation.index].to_numpy(bool), binding)
        comparisons.extend({**record, 'model_training_seed': seed} for record in public_cp)
        scores = args.private/'scores'/f'seed{seed}'
        scores.mkdir(parents=True, exist_ok=True)
        for name, part, logits in (('validation', validation, validation_logits), ('calibration', calibration, calibration_logits)):
            output = scores/f'{name}_scores.csv'
            if output.exists():
                raise ValueError('Preserve existing per-image score outputs')
            records = part[['image_id', 'split_group_id', 'dr_grade', 'is_group_representative']].copy()
            if name == 'calibration':
                records['calibration_role'] = role_lookup.calibration_role.to_numpy()
            for grade in range(5):
                records[f'logit_{grade}'] = logits[:, grade].numpy()
            records.to_csv(output, index=False, lineterminator='\n')
        write_json(scores/'conformal_comparisons.json', private_cp)
        private_hashes[str(seed)] = {name: sha256_file(scores/name) for name in
            ('validation_scores.csv', 'calibration_scores.csv', 'conformal_comparisons.json')}
    if len(comparisons) != 42 or any(sha256_file(preservation_paths[name]) != value for name, value in preservation.items()):
        raise ValueError('Declared comparisons are incomplete or original production artifacts changed')
    primary = next(row for row in runs if row['training_seed'] == PRIMARY_SEED)
    report = {'schema_version': 1, 'spec_version': 'MASTER_SPEC_v1.0', 'experiment_id': 'aptos_finetune_v1',
        'dataset': 'APTOS_2019', 'partition': 'validation', 'scope': 'development_finetuning_and_training_seed_diagnostics',
        'model_version': primary['model_version'], 'images': len(validation),
        'population': {'development_images': len(runtime), 'partition_images': {str(name): len(part) for name, part in runtime.groupby('split')},
            'temperature_images': int(temperature_mask.sum()), 'conformal_representative_images': int(cp_mask.sum()),
            'selection_representative_images': int(reps.loc[validation.index].sum()),
            'patient_identity_available': False, 'quality_labels_available': False, 'reserved_test_inference_performed': False},
        'summary': {'training_seeds': list(TRAINING_SEEDS), 'primary_training_seed': PRIMARY_SEED,
            'completed_runs': len(runs), 'conformal_comparisons': len(comparisons), 'deployment_changed': False,
            'grading_across_training_seeds': seed_summary(runs),
            'grading_before_temperature_across_training_seeds': seed_summary(runs, 'grading_before_temperature')},
        'baseline': {'report_sha256': BASELINE_SHA256, 'checkpoint_sha256': INITIAL_SHA256,
            'selection_metrics': baseline_candidate['selection_metrics'],
            'grading_before_temperature': baseline['grading_before_temperature'], 'grading': baseline['grading'],
            'same_selection_cohort': True},
        'runs': runs, 'grading': primary['grading'], 'grading_before_temperature': primary['grading_before_temperature'],
        'calibration': primary['calibration'], 'comparisons': comparisons, 'calibration_unit_definition': CALIBRATION_UNIT,
        'uniform_generation': 'Per model: SeedSequence(uniform_seed).spawn(2) independent fitting/selection streams; common draws across alpha and classifier seeds.',
        'source_commit': args.source_commit, 'execution_source_sha256': execution_sources,
        'training_source_provenance': source, 'input_bindings': lock,
        'posthoc_calibration_cache': calibration_cache_info,
        'pretraining_warm_start_equivalence': preflight,
        'private_score_sha256': private_hashes, 'elapsed_seconds': time.perf_counter()-started,
        'elapsed_seconds_scope': 'Three training/verification runs plus post-hoc diagnostics; warm-start canonical cache preflight recorded separately.',
        'package_versions': {name: package_metadata.version(name) for name in ('torch', 'torchvision', 'numpy', 'pandas', 'scikit-learn', 'Pillow')},
        'limitations': [
            'Development fine-tuning after inspecting the same selection population; no independent test-performance claim.',
            'All three classifier seeds share a previously validation-selected warm start; SD is conditional fine-tuning variation, not full pipeline uncertainty.',
            'Primary seed2026 is fixed before outcomes; no favorable seed or uncertainty replicate is selected.',
            'All learnable weights adapt while BatchNorm running buffers remain fixed; this differs from exact base-paper full fine-tuning.',
            'Fingerprint components are not patients; unknown related views, label-dependent quarantine and grade stratification limit exchangeability.',
            'Conformal score fitting uses one representative per calibration component, with image-unit schema and no patient/conditional guarantee.',
            'Repeated use of previously inspected calibration data is disclosed; all thresholds/methods/alphas are fixed, without new-coverage tuning.',
            'Five conformal uniform seeds per classifier are not additional classifier-training seeds.',
            'Bootstrap resamples fixed paired fingerprint groups without refitting; descriptive conditional intervals, not patient or training uncertainty.',
            'No APTOS quality labels, triage utility, external performance or deployment improvement is established.',
            'Grading ECE uses10bins; separate temperature diagnostics use15bins. Shared-machine durations are not architecture speed comparisons.']}
    serialized = json.dumps(report, allow_nan=False)
    if any(token in serialized for token in ('"image_id"', '"image_path"', '"subject_id"', str(root), 'C:\\Users\\')):
        raise ValueError('Public fine-tuning report includes private identities or paths')
    write_json(args.output/'evaluation.json', report)
    print(json.dumps({'status': 'completed', 'summary': report['summary'], 'elapsed_seconds': report['elapsed_seconds']}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=Path('data/manifests/aptos_v1/development.csv'))
    parser.add_argument('--full-manifest', type=Path, default=Path('data/manifests/aptos_v1/dataset_manifest.csv'))
    parser.add_argument('--calibration-roles', type=Path, default=Path('data/manifests/aptos_v1/calibration_roles.csv'))
    parser.add_argument('--data-audit', type=Path, default=Path('reports/aptos_v1/data_audit.json'))
    parser.add_argument('--baseline-report', type=Path, default=Path('reports/aptos_v1/evaluation.json'))
    parser.add_argument('--initial-checkpoint', type=Path, default=Path('checkpoints/aptos_v1/dr_efficientnet_b0/best.pt'))
    parser.add_argument('--aptos-root', type=Path, default=default_aptos_root())
    parser.add_argument('--protocol', type=Path, default=Path('docs/research/APTOS_FINETUNE_V1_PROTOCOL.md'))
    parser.add_argument('--config', type=Path, default=Path('configs/aptos_finetune_v1.yaml'))
    parser.add_argument('--private', type=Path, default=Path('artifacts/aptos_finetune_v1'))
    parser.add_argument('--checkpoints', type=Path, default=Path('checkpoints/aptos_finetune_v1'))
    parser.add_argument('--output', type=Path, default=Path('reports/aptos_finetune_v1'))
    parser.add_argument('--source-commit')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--resume-completed', action='store_true')
    parser.add_argument('--resume-last', action='store_true',
                        help='Resume provenance-identical completed training epochs; preserve partial score outputs')
    run(parser.parse_args())


if __name__ == '__main__':
    main()
