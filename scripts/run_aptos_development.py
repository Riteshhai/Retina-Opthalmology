"""Run the declared APTOS frozen-feature baseline; never infer on reserved test images.

Without --execute, only input readiness is checked. Actual execution requires the
protocol and this source to have been committed at --source-commit beforehand.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
from importlib import metadata as package_metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qsart.calibration.temperature import calibration_metrics, fit_temperature
from qsart.evaluation import grade_metrics
from qsart.models.checkpoint import load_checkpoint, sha256_file
from qsart.training.data import ManifestDataset, read_manifest, split_fingerprint
from qsart.training.transfer import (HEAD_RNG_VERSION, TransferConfig, _input_content_hash,
                                    pretrained_spec, source_provenance, train_transfer)
from qsart.uncertainty.conformal import coverage_metrics, fit_conformal, predict_sets
from qsart.uncertainty.randomized_ordinal import fit_randomized, predict_randomized

BACKBONES = ('efficientnet_b0', 'mobilenet_v3_small')
ALPHAS = (.10, .05)
RANDOMIZATION_SEEDS = (2026, 2027, 2028, 2029, 2030)
ROLE_PREFIX = 'aptos-v1-temperature-20261103:'
REPRESENTATIVE_PREFIX = 'aptos-v1-representative-20261103:'
CALIBRATION_UNIT = 'One deterministic representative image per observed SHA/pHash fingerprint component.'
PRESERVED = ('checkpoints/development_v1/selected/dr.pt', 'checkpoints/development_v1/selected/quality.pt',
             'checkpoints/development_v1/selected/calibration.json',
             'checkpoints/development_v1/selected/model_card.json', 'checkpoints/spec_v1/conformal_r1.json')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    path = Path(path)
    if path.exists():
        raise ValueError(f'Preserve existing output: {path.name}')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8', newline='\n')


def _order(value, prefix):
    return hashlib.sha256((prefix+str(value)).encode()).hexdigest(), str(value)


def representative_mask(frame):
    """Reconstruct the declared representative without using any outcome label."""
    mask = pd.Series(False, index=frame.index)
    for _, component in frame.groupby('split_group_id', sort=True):
        chosen = min(component.index, key=lambda i: _order(frame.at[i, 'image_id'], REPRESENTATIVE_PREFIX))
        mask.loc[chosen] = True
    return mask


def _bools(series):
    if not series.map(lambda value: str(value).lower() in {'true', 'false'}).all():
        raise ValueError('Representative flags must be explicit booleans')
    return series.astype(str).str.lower().eq('true')


def validate_development(frame, roles):
    """Fail closed on test rows, role leakage and label-dependent representative changes."""
    required = {'image_id', 'image_path', 'split', 'split_group_id', 'sha256', 'dr_grade',
                'dataset_source', 'official_split', 'is_group_representative'}
    if not required.issubset(frame) or frame.empty or not required.issubset(roles):
        raise ValueError('Complete APTOS development and calibration-role records required')
    if set(frame.split) != {'train', 'validation', 'calibration'}:
        raise ValueError('Development must contain train/validation/calibration and no test or holdout rows')
    if not frame.dataset_source.eq('APTOS_2019').all() or not frame.official_split.eq('official_train').all():
        raise ValueError('Only source-verified APTOS official-training records are accepted')
    if frame.image_id.duplicated().any() or frame.image_path.str.casefold().duplicated().any():
        raise ValueError('Development identities must be unique')
    if not pd.to_numeric(frame.dr_grade, errors='coerce').isin(range(5)).all():
        raise ValueError('Genuine integer DR grades 0 through 4 required')
    for column in ('split_group_id', 'sha256'):
        if frame[column].isna().any() or frame.groupby(column).split.nunique().gt(1).any():
            raise ValueError('Fingerprint components must remain in one partition')
    if frame.groupby('split_group_id').dr_grade.nunique().gt(1).any():
        raise ValueError('Conflicting components must be quarantined before development')
    expected_reps = representative_mask(frame)
    if not np.array_equal(_bools(frame.is_group_representative), expected_reps):
        raise ValueError('Representative flags differ from the declared outcome-independent hash rule')
    calibration = frame.loc[frame.split.eq('calibration')]
    if ('calibration_role' not in roles or roles.image_id.duplicated().any() or
            set(roles.calibration_role) != {'temperature', 'conformal'} or
            not roles.split.eq('calibration').all() or set(roles.image_id) != set(calibration.image_id)):
        raise ValueError('Calibration roles must cover exactly calibration rows and both disjoint roles')
    by_id = calibration.set_index('image_id')
    compared = roles.set_index('image_id').loc[by_id.index]
    for field in ('split_group_id', 'image_path', 'sha256', 'dr_grade'):
        if not np.array_equal(compared[field].to_numpy(), by_id[field].to_numpy()):
            raise ValueError(f'Calibration roles differ from the frozen development records: {field}')
    if not np.array_equal(_bools(compared.is_group_representative), _bools(by_id.is_group_representative)):
        raise ValueError('Calibration role representatives differ from the development manifest')
    groups = sorted(calibration.split_group_id.unique(), key=lambda value: _order(value, ROLE_PREFIX))
    temperature_groups = set(groups[:len(groups)//2])
    expected = roles.split_group_id.map(lambda group: 'temperature' if group in temperature_groups else 'conformal')
    if not roles.calibration_role.eq(expected).all():
        raise ValueError('Temperature/conformal roles differ from the declared group hash partition')
    if roles.groupby('split_group_id').calibration_role.nunique().gt(1).any():
        raise ValueError('A fingerprint component crosses temperature/conformal roles')
    temperature = roles.loc[roles.calibration_role.eq('temperature')]
    conformal = roles.loc[roles.calibration_role.eq('conformal') & _bools(roles.is_group_representative)]
    if len(temperature) < 2 or len(conformal) < 2:
        raise ValueError('Independent temperature and conformal observations are insufficient')
    return expected_reps


def validate_inputs(args):
    frame, roles = read_manifest(args.manifest), read_manifest(args.calibration_roles)
    representative_mask_actual = validate_development(frame, roles)
    # Do not read the reserved test DR-label column. Full-manifest identities are
    # used only to check that its non-test rows exactly match development.
    columns = ['image_id', 'image_path', 'split', 'split_group_id', 'sha256']
    full = pd.read_csv(args.full_manifest, usecols=columns)
    if set(full.split) != {'train', 'validation', 'calibration', 'test'}:
        raise ValueError('Complete source manifest needs an explicitly reserved internal test partition')
    if full.image_id.duplicated().any() or full.groupby('split_group_id').split.nunique().gt(1).any():
        raise ValueError('Full source manifest identities/groups are inconsistent')
    development = full.loc[~full.split.eq('test')].set_index('image_id').sort_index()
    compared = frame[columns].set_index('image_id').sort_index()
    if not development.equals(compared):
        raise ValueError('Development is not the exact non-test subset of the frozen full manifest')
    audit = read_json(args.data_audit)
    bindings = {'manifest_sha256': args.full_manifest, 'development_manifest_sha256': args.manifest,
                'calibration_roles_sha256': args.calibration_roles}
    if any(audit.get(key) != sha256_file(path) for key, path in bindings.items()):
        raise ValueError('Dataset audit and frozen manifest hash bindings differ')
    hashes = {key: sha256_file(path) for key, path in bindings.items()}
    hashes['data_audit_sha256'] = sha256_file(args.data_audit)
    hashes['archive_sha256'] = audit['archive_sha256']
    return frame, roles, representative_mask_actual, hashes


def validate_source(root, protocol, config_path, commit, require_clean=True):
    if subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip() != commit:
        raise ValueError('Current HEAD must equal the declared execution source commit')
    if require_clean and subprocess.check_output(['git', 'status', '--porcelain'], cwd=root, text=True).strip():
        raise ValueError('Commit protocol and execution sources before real APTOS training')
    paths = [protocol, config_path, Path(__file__).resolve()]
    hashes = {}
    for path in paths:
        relative = Path(path).resolve().relative_to(root).as_posix()
        committed = subprocess.check_output(['git', 'show', f'{commit}:{relative}'], cwd=root)
        if committed != Path(path).read_bytes():
            raise ValueError(f'Committed protocol/source bytes differ: {relative}')
        hashes[relative] = sha256_file(path)
    return hashes


def fixed_configs(path):
    actual = yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    expected = asdict(TransferConfig(backbone='efficientnet_b0'))
    if actual != expected:
        raise ValueError('APTOS config differs from the fixed declared settings')
    return {backbone: replace(TransferConfig(**actual), backbone=backbone) for backbone in BACKBONES}


def copy_local_pretrained(root, cache):
    destination = cache/'torchhub/checkpoints'
    destination.mkdir(parents=True, exist_ok=True)
    for backbone in BACKBONES:
        spec = pretrained_spec(backbone)
        target = destination/spec['filename']
        if not target.is_file():
            found = sorted(path for path in (root/'artifacts').rglob(spec['filename']) if path != target)
            source = next((path for path in found if sha256_file(path).startswith(spec['published_sha256_prefix'])), None)
            if source is None:
                raise ValueError(f'Locally preserved official pretrained weights missing for {backbone}; no download attempted')
            shutil.copy2(source, target)
        if not sha256_file(target).startswith(spec['published_sha256_prefix']):
            raise ValueError('Local pretrained checksum does not match its official URL prefix')


def _grading(labels, probabilities):
    result = grade_metrics(labels, probabilities)
    result['nll'] = float(-np.log(np.clip(probabilities[np.arange(len(labels)), labels], 1e-15, 1)).mean())
    result['ece_bins'] = 10
    return result


def verify_candidate(folder, config, frame, manifest_hash, provenance, cache):
    required = ('best.pt', 'last.pt', 'training_result.json', 'training_metadata.json',
                'selection_predictions.csv', 'training_history.csv', 'selection_metrics.json')
    if not all((folder/name).is_file() for name in required):
        raise ValueError('Candidate folder is incomplete; completed-run reuse refuses partial training')
    model, checkpoint = load_checkpoint(folder/'best.pt')
    metadata = checkpoint['metadata']
    expected = {'training_config': asdict(config), 'backbone': config.backbone, 'task': 'dr',
        'heads': {'dr_grade': 5}, 'manifest_sha256': manifest_hash, 'confirmation_used': False,
        'backbone_frozen': True, 'data_purpose': 'real_research', 'head_rng_version': HEAD_RNG_VERSION,
        'head_initialization_seed': config.seed, 'code_provenance': provenance}
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError('Candidate does not match declared config/source/manifest/provenance')
    if metadata != read_json(folder/'training_metadata.json'):
        raise ValueError('Checkpoint and exported training metadata differ')
    selected = frame.loc[frame.split.eq('validation')]
    predictions = pd.read_csv(folder/'selection_predictions.csv')
    if (predictions.manifest_row.duplicated().any() or
            not np.array_equal(predictions.manifest_row, selected.index) or
            not np.array_equal(predictions.true_dr_grade, selected.dr_grade)):
        raise ValueError('Candidate predictions differ from the genuine frozen selection records')
    probabilities = predictions[[f'dr_grade_p{i}' for i in range(5)]].to_numpy(float)
    if (not np.isfinite(probabilities).all() or (probabilities < 0).any() or
            not np.allclose(probabilities.sum(1), 1, atol=1e-5, rtol=0)):
        raise ValueError('Candidate prediction probabilities are invalid')
    probabilities /= probabilities.sum(1, keepdims=True)
    metrics = _grading(selected.dr_grade.to_numpy(int), probabilities)
    for key in ('accuracy', 'macro_f1', 'nll'):
        if not np.isclose(metrics[key], metadata['selection_metrics']['dr_grade'][key], atol=1e-5, rtol=0):
            raise ValueError('Recomputed candidate metrics differ from checkpoint selection metrics')
    cache_info = metadata['feature_cache_provenance']
    cache_path = cache/f"features_{config.backbone}_{metadata['feature_cache_key']}.pt"
    if sha256_file(cache_path) != cache_info['cache_file_sha256']:
        raise ValueError('Feature cache changed since training')
    cached = torch.load(cache_path, map_location='cpu', weights_only=True)
    if cached['metadata'] != cache_info['metadata']:
        raise ValueError('Feature cache metadata differs from selected checkpoint')
    expected_rows = frame.loc[frame.split.isin(['train', 'validation'])].index.tolist()
    if cached['metadata']['manifest_rows'] != expected_rows:
        raise ValueError('Feature extraction included records outside fitting/selection')
    result = read_json(folder/'training_result.json')
    history = pd.read_csv(folder/'training_history.csv')
    if len(history) != result['epochs_executed'] or result['selected_epoch'] != checkpoint['epoch']+1:
        raise ValueError('Selected epoch/history/result records disagree')
    public = {'backbone': config.backbone, 'legacy_checkpoint_model_version': metadata['model_version'],
        'checkpoint_sha256': sha256_file(folder/'best.pt'), 'last_checkpoint_sha256': sha256_file(folder/'last.pt'),
        'checkpoint_bytes': (folder/'best.pt').stat().st_size, 'selected_epoch': result['selected_epoch'],
        'epochs_executed': result['epochs_executed'], 'optimizer_steps': result['total_optimizer_steps'],
        'training_config': asdict(config), 'selection_metrics': metrics, 'history': history.to_dict('records'),
        'training_strategy': metadata['training_strategy'], 'head_rng_version': metadata['head_rng_version'],
        'backbone_unchanged': result['backbone_unchanged'], 'pretrained_provenance': metadata['pretrained_provenance'],
        'feature_cache_key': metadata['feature_cache_key'], 'feature_cache_sha256': cache_info['cache_file_sha256'],
        'feature_extraction_seconds': cache_info['extraction_seconds'],
        'artifact_sha256': {name: sha256_file(folder/name) for name in required}}
    return public, model, checkpoint, probabilities


def choose_candidate(candidates):
    """The declared classifier rule cannot inspect conformal or calibration outcomes."""
    return min(candidates, key=lambda row: (-row['selection_metrics']['macro_f1'],
                                           row['selection_metrics']['nll'], BACKBONES.index(row['backbone'])))


def infer_logits(model, frame, metadata):
    if not frame.split.isin(['calibration', 'validation']).all():
        raise ValueError('Score inference accepts calibration and development selection rows only')
    dataset = ManifestDataset(frame, {'dr_grade': 5}, metadata['image_size'],
                              mean=metadata['normalization_mean'], std=metadata['normalization_std'])
    loader = DataLoader(dataset, batch_size=16, num_workers=2, shuffle=False)
    logits, labels = [], []
    with torch.inference_mode():
        for images, batch_labels in loader:
            logits.append(model(images)['dr_grade'].detach().cpu())
            labels.append(batch_labels['dr_grade'])
    output, truth = torch.cat(logits), torch.cat(labels)
    if not torch.equal(truth, torch.tensor(frame.dr_grade.to_numpy(), dtype=torch.long)):
        raise ValueError('Score inference order/labels changed')
    return output


def set_metrics(sets, truth, reps):
    def values(selected_sets, labels):
        result = coverage_metrics(selected_sets, labels)
        result['full_five_grade_set_count'] = result['set_size_counts']['5']
        return result
    return {'all_selection_images': values(sets, truth),
            'selection_group_representatives': values([s for s, keep in zip(sets, reps, strict=True) if keep], truth[reps])}


def conformal_comparisons(calibration_p, calibration_truth, selection_p, selection_truth, reps, binding):
    """Keep all declared seeds/alphas, raw empties and the common modal enlargement."""
    comparisons, private = [], []
    streams = {}
    for seed in RANDOMIZATION_SEEDS:
        fitting_seed, selection_seed = np.random.SeedSequence(seed).spawn(2)
        streams[seed] = (np.random.default_rng(fitting_seed).random(len(calibration_truth)),
                         np.random.default_rng(selection_seed).random(len(selection_truth)))
    for alpha in ALPHAS:
        for method, seed in [('lac', None), ('ordinal_greedy', None)]+[
                ('ordinal_greedy_randomized', seed) for seed in RANDOMIZATION_SEEDS]:
            metadata = {**binding, 'calibration_unit_definition': CALIBRATION_UNIT,
                'scope': 'empirical development diagnostics; unknown patient overlap and grade-stratified partitions',
                'randomization_seed': seed}
            if seed is None:
                artifact = fit_conformal(calibration_p, calibration_truth, method=method, alpha=alpha,
                                         groups=None, metadata=metadata, empty_policy='allow')
                raw = predict_sets(selection_p, artifact)
                enlarged = predict_sets(selection_p, {**artifact, 'empty_policy': 'argmax'})
            else:
                fitting_uniforms, selection_uniforms = streams[seed]
                artifact = fit_randomized(calibration_p, calibration_truth, fitting_uniforms,
                                          alpha=alpha, groups=None, metadata=metadata)
                raw = predict_randomized(selection_p, artifact, selection_uniforms, empty_policy='allow')
                enlarged = predict_randomized(selection_p, artifact, selection_uniforms, empty_policy='argmax')
            key = {'method': method, 'alpha': alpha, 'randomization_seed': seed}
            comparisons.append({**key, 'artifact': artifact, 'raw': set_metrics(raw, selection_truth, reps),
                                'argmax_fallback': set_metrics(enlarged, selection_truth, reps)})
            private.append({**key, 'raw_sets': raw, 'argmax_fallback_sets': enlarged,
                            'calibration_uniforms': streams[seed][0].tolist() if seed is not None else None,
                            'selection_uniforms': streams[seed][1].tolist() if seed is not None else None})
    if len(comparisons) != 14:
        raise ValueError('All fourteen declared conformal comparisons must be retained')
    return comparisons, private


def run(args):
    root = Path(__file__).resolve().parents[1]
    frame, roles, reps, bindings = validate_inputs(args)
    configs = fixed_configs(args.config)
    counts = {str(name): len(part) for name, part in frame.groupby('split')}
    if not args.execute:
        print(json.dumps({'status': 'ready_inputs_only', 'population': counts, 'bindings': bindings,
                          'planned_candidates': list(BACKBONES), 'training_or_inference_performed': False}, indent=2))
        return
    if not args.source_commit:
        raise ValueError('--execute requires a frozen --source-commit')
    if (args.output/'evaluation.json').exists():
        raise ValueError('Completed APTOS experiment already exists; preserve it')
    for path, allowed in [(args.checkpoints, root/'checkpoints'), (args.private, root/'artifacts')]:
        if not path.resolve().is_relative_to(allowed):
            raise ValueError('Checkpoints and per-image outputs must remain in ignored local artifact trees')
    protocol_sources = validate_source(root, args.protocol, args.config, args.source_commit)
    provenance = source_provenance()
    preserved = {path: sha256_file(root/path) for path in PRESERVED}
    # This verifies bytes only; the internal test manifest's images are never opened.
    all_development_bytes = _input_content_hash(frame)
    lock = {'bindings': bindings, 'source_commit': args.source_commit, 'source_sha256': provenance['source_sha256'],
        'protocol_sources': protocol_sources, 'configurations': {name: asdict(value) for name, value in configs.items()},
        'development_input_content_sha256': all_development_bytes, 'preserved_artifact_sha256': preserved}
    lock_path = args.private/'execution_lock.json'
    if lock_path.exists():
        if not args.resume_completed or read_json(lock_path) != lock:
            raise ValueError('Existing execution lock requires matching provenance and --resume-completed')
    else:
        write_json(lock_path, lock)
    feature_cache = args.private/'features'
    copy_local_pretrained(root, feature_cache)
    candidates, candidate_models = [], {}
    started = time.perf_counter()
    for backbone, config in configs.items():
        validate_source(root, args.protocol, args.config, args.source_commit)
        folder = args.checkpoints/f'dr_{backbone}'
        begin = time.perf_counter()
        if folder.exists():
            if not args.resume_completed:
                raise ValueError('Existing candidate outputs require explicit verified completed-run reuse')
        else:
            train_transfer(args.manifest, folder, feature_cache, task='dr', config=config,
                           allow_download=False, data_purpose='real_research')
        candidate, model, checkpoint, probabilities = verify_candidate(
            folder, config, frame, bindings['development_manifest_sha256'], provenance, feature_cache)
        candidate['run_or_verification_seconds'] = time.perf_counter()-begin
        candidates.append(candidate)
        candidate_models[backbone] = model, checkpoint, probabilities
    selected = choose_candidate(candidates)
    model, checkpoint, old_probabilities = candidate_models[selected['backbone']]
    torch.set_num_threads(2)
    validation = frame.loc[frame.split.eq('validation')]
    calibration = frame.loc[frame.split.eq('calibration')]
    validation_logits = infer_logits(model, validation, checkpoint['metadata'])
    calibration_logits = infer_logits(model, calibration, checkpoint['metadata'])
    difference = float(np.max(np.abs(validation_logits.softmax(1).numpy()-old_probabilities)))
    if difference > 2e-6:
        raise ValueError('Ordinary selected-model inference differs from saved genuine selection probabilities')
    role_by_id = roles.set_index('image_id').loc[calibration.image_id]
    temperature_mask = role_by_id.calibration_role.eq('temperature').to_numpy()
    cp_mask = (role_by_id.calibration_role.eq('conformal') & _bools(role_by_id.is_group_representative)).to_numpy()
    calibration_truth = torch.tensor(calibration.dr_grade.to_numpy(), dtype=torch.long)
    validation_truth = validation.dr_grade.to_numpy(int)
    temperature = fit_temperature(calibration_logits[temperature_mask], calibration_truth[temperature_mask])
    validation_p = (validation_logits.double()/temperature).softmax(1).numpy()
    cp_p = (calibration_logits[cp_mask].double()/temperature).softmax(1).numpy()
    selected_reps = reps.loc[validation.index].to_numpy(bool)
    cp_binding = {'checkpoint_sha256': selected['checkpoint_sha256'],
        'development_manifest_sha256': bindings['development_manifest_sha256'],
        'calibration_roles_sha256': bindings['calibration_roles_sha256'], 'independent_temperature': temperature,
        'temperature_split_fingerprint': split_fingerprint(calibration.loc[temperature_mask]),
        'conformal_representatives_fingerprint': split_fingerprint(calibration.loc[cp_mask])}
    comparisons, private_comparisons = conformal_comparisons(cp_p, calibration_truth[cp_mask].numpy(),
        validation_p, validation_truth, selected_reps, cp_binding)
    scores = args.private/'scores'
    scores.mkdir(parents=True, exist_ok=True)
    for name, part, logits in [('validation', validation, validation_logits),
                               ('calibration', calibration, calibration_logits)]:
        output = scores/f'{name}_scores.csv'
        if output.exists():
            raise ValueError('Preserve existing private score outputs')
        records = part[['image_id', 'split_group_id', 'dr_grade', 'is_group_representative']].copy()
        if name == 'calibration':
            records['calibration_role'] = role_by_id.calibration_role.to_numpy()
        for grade in range(5):
            records[f'logit_{grade}'] = logits[:, grade].numpy()
        records.to_csv(output, index=False, lineterminator='\n')
    write_json(scores/'conformal_comparisons.json', private_comparisons)
    write_json(args.checkpoints/'study_calibration.json', {**cp_binding, 'temperature': temperature,
        'scope': 'research-only DR temperature; not a production DR/quality calibration bundle'})
    if any(sha256_file(root/path) != value for path, value in preserved.items()):
        raise ValueError('Existing selected deployment artifacts changed during the APTOS study')
    artifact_identity = f"aptos-v1-{selected['backbone']}-{selected['checkpoint_sha256'][:12]}"
    report = {'schema_version': 1, 'spec_version': 'MASTER_SPEC_v1.0', 'experiment_id': 'aptos_development_v1',
        'dataset': 'APTOS_2019', 'partition': 'validation', 'scope': 'development_selection_and_independent_calibration',
        'model_version': artifact_identity, 'artifact_identity': artifact_identity,
        'legacy_checkpoint_model_version': checkpoint['metadata']['model_version'], 'images': len(validation),
        'population': {'development_images': len(frame), 'partition_images': counts,
            'partition_fingerprint_groups': {str(name): int(part.split_group_id.nunique())
                                             for name, part in frame.groupby('split')},
            'temperature_images': int(temperature_mask.sum()), 'conformal_representative_images': int(cp_mask.sum()),
            'selection_representative_images': int(selected_reps.sum()),
            'patient_identity_available': False, 'patient_disjointness_established': False,
            'quality_labels_available': False, 'reserved_test_inference_performed': False},
        'summary': {'candidates': len(candidates), 'selected_backbone': selected['backbone'],
            'selection_rule': 'macro_f1_then_lower_unweighted_nll_then_declared_backbone_order',
            'conformal_comparisons': len(comparisons), 'training_seed': 2026,
            'randomization_seeds': list(RANDOMIZATION_SEEDS), 'alphas': list(ALPHAS),
            'selected_checkpoint_sha256': selected['checkpoint_sha256'], 'deployment_changed': False},
        'candidates': candidates, 'grading': _grading(validation_truth, validation_p),
        'grading_before_temperature': _grading(validation_truth, validation_logits.double().softmax(1).numpy()),
        'calibration': {'temperature': temperature, 'ece_bins': 15,
            'temperature_fit_before': calibration_metrics(calibration_logits[temperature_mask], calibration_truth[temperature_mask]),
            'temperature_fit_after': calibration_metrics(calibration_logits[temperature_mask], calibration_truth[temperature_mask], temperature),
            'selection_before': calibration_metrics(validation_logits, validation_truth),
            'selection_after': calibration_metrics(validation_logits, validation_truth, temperature),
            'fit_scope': 'Temperature-role fitting diagnostics; not independent test results.'},
        'comparisons': comparisons, 'calibration_unit_definition': CALIBRATION_UNIT,
        'uniform_generation': 'NumPy SeedSequence(seed).spawn(2): independent calibration/selection default_rng streams; same draws across alpha.',
        'source_commit': args.source_commit, 'execution_source_sha256': protocol_sources,
        'training_source_provenance': provenance, 'input_bindings': bindings,
        'development_input_content_sha256': all_development_bytes,
        'preserved_deployment_artifact_sha256': preserved,
        'selection_probability_equivalence_max_abs_delta': difference,
        'private_score_sha256': {name: sha256_file(scores/name) for name in
                               ('validation_scores.csv', 'calibration_scores.csv', 'conformal_comparisons.json')},
        'package_versions': {name: package_metadata.version(name) for name in
                            ('torch', 'torchvision', 'numpy', 'pandas', 'scikit-learn', 'Pillow')},
        'elapsed_seconds': time.perf_counter()-started,
        'limitations': [
            'CPU frozen-ImageNet-feature linear-head baseline, not full fine-tuning or exact base-paper reproduction.',
            'Selection labels choose epochs and architecture; grading and conformal outcomes here are development diagnostics, not a fresh test.',
            'Fingerprint components are not patients; unknown same-person/eye overlap, grade stratification and quarantine preclude blanket exchangeability claims.',
            'Conformal thresholds use one fingerprint-component representative with the image-unit schema; no patient-max or patient-coverage claim.',
            'The reserved internal test and inaccessible Kaggle competition test have no inference in this study; initial label-stratified preparation necessarily read source grades.',
            'No APTOS quality/eye/patient labels are available; no quality-aware triage or clinical utility is measured.',
            'Five randomization replicates are not model-training seeds; no seed, alpha, uncertainty method or checkpoint is promoted to deployment.',
            'ECE in grading uses 10 bins; separate temperature diagnostics use 15 bins.',
            'Legacy checkpoint model-version strings may collide across datasets; checkpoint/manifest hashes and study identity are authoritative.',
            'Shared-machine elapsed durations do not establish an architecture speed advantage.',
        ]}
    serialized = json.dumps(report, allow_nan=False)
    if any(token in serialized for token in ('"image_id"', '"image_path"', '"subject_id"', str(root), 'C:\\Users\\')):
        raise ValueError('Public report contains private per-image fields or local paths')
    write_json(args.output/'evaluation.json', report)
    print(json.dumps({'status': 'completed', 'summary': report['summary'], 'grading': report['grading'],
                      'elapsed_seconds': report['elapsed_seconds']}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=Path('data/manifests/aptos_v1/development.csv'))
    parser.add_argument('--full-manifest', type=Path, default=Path('data/manifests/aptos_v1/dataset_manifest.csv'))
    parser.add_argument('--calibration-roles', type=Path, default=Path('data/manifests/aptos_v1/calibration_roles.csv'))
    parser.add_argument('--data-audit', type=Path, default=Path('reports/aptos_v1/data_audit.json'))
    parser.add_argument('--protocol', type=Path, default=Path('docs/research/APTOS_V1_PROTOCOL.md'))
    parser.add_argument('--config', type=Path, default=Path('configs/aptos_v1.yaml'))
    parser.add_argument('--checkpoints', type=Path, default=Path('checkpoints/aptos_v1'))
    parser.add_argument('--private', type=Path, default=Path('artifacts/aptos_v1'))
    parser.add_argument('--output', type=Path, default=Path('reports/aptos_v1'))
    parser.add_argument('--source-commit')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--resume-completed', action='store_true')
    run(parser.parse_args())


if __name__ == '__main__':
    main()
