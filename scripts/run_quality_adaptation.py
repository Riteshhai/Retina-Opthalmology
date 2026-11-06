"""Predeclared six-run DeepDRiD quality study. Default: input readiness only."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import re
import subprocess
import sys

import numpy as np
import pandas as pd
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qsart.data.development import defect_mapping_report
from qsart.models.checkpoint import load_checkpoint, sha256_file
from qsart.training.data import read_manifest, split_fingerprint
from qsart.training.finetune import prepare_input_cache
from qsart.training.quality_adaptation import (ARMS, HEADS, MANIFEST_SHA256, SELECTOR, SEEDS,
    WARM_SHA256, QualityConfig, canonical_hash, evaluate, paired_component_changes, selection_score,
    source_provenance, train_quality, validate_frame, verify_seal, write_json)
from qsart.training.transfer import _input_content_hash

STUDY = 'quality_adaptation_v1'
PRIOR_SHA256 = '1ddb1d230a53dab0c75019e73654f2ab8d17313cc586a9c70efeff29fae4bb9e'
PROTOCOL = 'docs/research/QUALITY_ADAPTATION_V1_PROTOCOL.md'
CONFIG = 'configs/quality_adaptation_v1.yaml'
LIMITATIONS = [
    'Reused DeepDRiD development selection; this is not fresh confirmation or clinical validation.',
    'Three paired seeds condition on one development split, legacy warm starts and a fixed 15-epoch budget.',
    'Both arms use a new four-head selector and online flips; only their matched difference isolates backbone adaptation.',
    'Field-definition class 0 has only three fitting images and one selection image; all class supports and recalls are retained.',
    'Whole-component bootstrap intervals condition on selected checkpoints; no refitting, reselection or multiplicity correction.',
    'Raw probabilities only. No calibration pixels, temperature/conformal fitting, triage tuning or production model switch.',
    'Consumed internal confirmation and IDRiD, and reserved APTOS test are excluded from this study.',
]


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def population(root):
    manifest = root/'data/processed/deepdrid/development_v1.csv'
    if sha256_file(manifest) != MANIFEST_SHA256:
        raise ValueError('Frozen quality development manifest changed')
    frame = read_manifest(manifest)
    validate_frame(frame)
    counts = {name: len(part) for name, part in frame.groupby('split')}
    patients = {name: int(part.subject_id.nunique()) for name, part in frame.groupby('split')}
    groups = {name: int(part.split_group_id.nunique()) for name, part in frame.groupby('split')}
    if (counts != {'train': 848, 'validation': 168, 'calibration': 168} or
            patients != {'train': 212, 'validation': 42, 'calibration': 42} or groups != patients):
        raise ValueError('Frozen population counts/patient/components changed')
    selected = frame.loc[frame.split.isin(['train', 'validation'])]
    supports = {head: {role: [int((part[head] == c).sum()) for c in range(classes)]
                      for role, part in selected.groupby('split')} for head, classes in HEADS.items()}
    return frame, {'manifest_sha256': MANIFEST_SHA256, 'split_counts': counts,
        'split_patient_counts': patients, 'split_group_counts': groups,
        'split_fingerprints': {name: split_fingerprint(part) for name, part in frame.groupby('split')},
        'input_content_sha256': _input_content_hash(selected), 'class_supports': supports,
        'calibration_images_read_or_predicted': False, 'confirmation_images_read_or_predicted': False,
        'aptos_images_read_or_predicted': False, 'idrid_images_read_or_predicted': False}


def warm_bindings(root):
    prior = root/'reports/repeated_seeds_v1/summary.json'
    if sha256_file(prior) != PRIOR_SHA256:
        raise ValueError('Original repeated-seeds report changed')
    result = {}
    for seed in SEEDS:
        run = next(row for row in read_json(prior)['runs'] if row['task'] == 'quality' and row['seed'] == seed)
        if run['artifact_sha256']['best.pt'] != WARM_SHA256[seed]:
            raise ValueError('Warm-start binding differs from protocol')
        folder = root/f'checkpoints/repeated_seeds_v1/quality_seed{seed}'
        for name, expected in run['artifact_sha256'].items():
            if sha256_file(folder/name) != expected:
                raise ValueError('Legacy quality artifact changed: '+name)
        result[str(seed)] = run['artifact_sha256']
    return result


def preserved_files(root):
    names = subprocess.check_output(['git', 'ls-files', 'reports/**/*.json', 'reports/**/*.png'], cwd=root, text=True).splitlines()
    names = [name for name in names if not name.startswith(f'reports/{STUDY}/')]
    names += ['MASTER_SPEC.md', 'docs/specifications/MASTER_SPEC_v1.0.md',
        'data/processed/deepdrid/development_v1.csv',
        'checkpoints/development_v1/selected/dr.pt', 'checkpoints/development_v1/selected/quality.pt',
        'checkpoints/development_v1/selected/calibration.json', 'checkpoints/development_v1/selected/model_card.json',
        'checkpoints/spec_v1/conformal_r1.json']
    for study in ('aptos_finetune_v1', 'aptos_optimization_v2'):
        prior = read_json(root/f'reports/{study}/evaluation.json')
        for run in prior['runs']:
            seed = run['training_seed']
            for name, expected in run['artifact_sha256'].items():
                path = root/f'checkpoints/{study}/seed{seed}'/name
                if sha256_file(path) != expected:
                    raise ValueError('Prior APTOS run artifact changed')
                names.append(path.relative_to(root).as_posix())
    for seed in SEEDS:
        for name in warm_bindings(root)[str(seed)]:
            names.append(f'checkpoints/repeated_seeds_v1/quality_seed{seed}/{name}')
    return {name: sha256_file(root/name) for name in sorted(set(names))}


def preflight(root, frame, config):
    selected = frame.loc[frame.split.isin(['train', 'validation'])]
    cache, cache_info = prepare_input_cache(selected, root/'data/processed/deepdrid/development_v1.csv',
        root/'artifacts/quality_adaptation_v1/input_cache', image_size=config.image_size, workers=config.cache_workers)
    val = selected.loc[selected.split.eq('validation')]
    mask = torch.from_numpy(selected.split.eq('validation').to_numpy(dtype=bool, copy=True))
    targets = {name: torch.tensor(val[name].to_numpy(dtype=np.int64, copy=True)) for name in HEADS}
    results = {}
    torch.set_num_threads(config.cpu_threads)
    for seed in SEEDS:
        folder = root/f'checkpoints/repeated_seeds_v1/quality_seed{seed}'
        model, _ = load_checkpoint(folder/'best.pt')
        if config.channels_last:
            model.to(memory_format=torch.channels_last)
        logits, metrics = evaluate(model, cache['images'][mask], targets, config)
        old = pd.read_csv(folder/'selection_predictions.csv', float_precision='round_trip').sort_values('manifest_row')
        if not np.array_equal(old.manifest_row, val.index):
            raise ValueError('Legacy warm prediction population differs')
        delta = {}
        for name, classes in HEADS.items():
            probs = torch.softmax(logits[name].double(), 1).numpy()
            previous = old[[f'{name}_p{i}' for i in range(classes)]].to_numpy()
            delta[name] = float(np.max(np.abs(previous-probs)))
            if delta[name] > 2e-5 or not np.array_equal(previous.argmax(1), probs.argmax(1)):
                raise ValueError('Canonical warm-start reproduction failed')
        results[str(seed)] = {'all_head_argmax_identical': True, 'maximum_probability_delta': delta,
                              'warm_metrics': metrics}
    return {'cache_key': cache_info['cache_key'], 'tensor_sha256': cache_info['tensor_sha256'], 'seeds': results}


def verify_run(root, seed, arm, runtime, config):
    folder = root/f'checkpoints/{STUDY}/seed{seed}_{arm}'
    result = verify_seal(folder)
    metadata = read_json(folder/'training_metadata.json')
    state = torch.load(folder/'state.pt', weights_only=True, map_location='cpu')
    provenance = metadata['source_provenance']
    if (provenance != runtime['source_provenance'] or metadata['training_config'] != asdict(config) or
            result['epochs_executed'] != config.epochs or result['seed'] != seed or result['arm'] != arm or
            metadata['initial_checkpoint_sha256'] != WARM_SHA256[seed] or
            metadata['manifest_sha256'] != MANIFEST_SHA256 or result['data_purpose'] != 'real_research'):
        raise ValueError('Completed run differs from frozen study')
    selected = max(state['history'], key=lambda row: (row['mean_head_macro_f1'], -row['mean_head_nll']))
    if selected['epoch'] != result['selected_epoch'] or list(selection_score(selected['selection_metrics'])) != state['best']['score']:
        raise ValueError('Selected epoch disagrees with predeclared four-head selector')
    return {**result, 'artifact_sha256': read_json(folder/'completion.json')['artifact_sha256'],
            'completion_sha256': sha256_file(folder/'completion.json'), 'training_config': asdict(config),
            'optimizer_steps_at_selected_epoch': metadata['optimizer_steps']}


def summarize(root, runtime, frame, config):
    runs = [verify_run(root, seed, arm, runtime, replace(config, seed=seed)) for seed in SEEDS for arm in ARMS]
    validation = frame.loc[frame.split.eq('validation')]
    truth = {name: validation[name].to_numpy(dtype=int) for name in HEADS}
    changes = []
    for seed in SEEDS:
        predictions = {}
        histories = []
        for arm in ARMS:
            folder = root/f'checkpoints/{STUDY}/seed{seed}_{arm}'
            saved = pd.read_csv(folder/'selection_predictions.csv', float_precision='round_trip')
            if not np.array_equal(saved.manifest_row, validation.index):
                raise ValueError('Paired prediction identities differ')
            predictions[arm] = {name: saved[f'predicted_{name}'].to_numpy(dtype=int) for name in HEADS}
            histories.append(read_json(folder/'training_history.json'))
        for first, second in zip(*histories, strict=True):
            if any(first[name] != second[name] for name in ('epoch', 'optimizer_steps', 'order_sha256', 'flip_sha256', 'flip_count')):
                raise ValueError('Paired arms have different exposure/order/flips')
        changes.append({'seed': seed, **paired_component_changes(truth, predictions['frozen'], predictions['adapted'],
                                                               validation.split_group_id.to_numpy())})
    summary = {}
    for arm in ARMS:
        selected = [row for row in runs if row['arm'] == arm]
        values = {}
        keys = ('accuracy', 'macro_f1', 'nll', 'ece_10', 'brier')
        for head in HEADS:
            values[head] = {key: {'mean': float(np.mean([row['selected_metrics'][head][key] for row in selected])),
                                 'sample_sd': float(np.std([row['selected_metrics'][head][key] for row in selected], ddof=1))}
                            for key in keys}
        mean_scores = [row['mean_head_macro_f1'] for row in selected]
        summary[arm] = {'seeds': list(SEEDS), 'mean_head_macro_f1': {'mean': float(np.mean(mean_scores)),
                         'sample_sd': float(np.std(mean_scores, ddof=1))}, 'heads': values}
    majority = {}
    for name, classes in HEADS.items():
        support = runtime['population']['class_supports'][name]['train']
        label = int(np.argmax(support))
        y = truth[name]
        cm = np.bincount(y*classes+label, minlength=classes*classes).reshape(classes, classes)
        d = cm.sum(0)+cm.sum(1)
        majority[name] = {'class': label, 'accuracy': float(np.mean(y==label)),
            'macro_f1': float(np.divide(2*cm.diagonal(), d, out=np.zeros(classes), where=d>0).mean()),
            'support': runtime['population']['class_supports'][name]['validation']}
    report = {'schema_version': 'qsart.quality_adaptation.v1', 'experiment_id': STUDY, 'dataset': 'DeepDRiD',
        'partition': 'validation', 'scope': 'development_matched_quality_backbone_adaptation',
        'source_commit': runtime['source_provenance']['git_revision'],
        'source_file_sha256': runtime['source_provenance']['source_file_sha256'],
        'protocol_sha256': runtime['protocol_sha256'], 'config_sha256': runtime['config_sha256'],
        'runtime_binding_sha256': canonical_hash(runtime), 'population': runtime['population'],
        'primary_seed': 2026, 'selection_rule': SELECTOR, 'fixed_epochs_per_run': config.epochs,
        'runs': runs, 'summary': summary, 'paired_changes': changes, 'majority_references': majority,
        'warm_start_reproduction': runtime['warm_start_reproduction'], 'defect_mapping': defect_mapping_report(),
        'prior_repeated_seed_report_sha256': PRIOR_SHA256, 'preserved_artifact_sha256': runtime['preserved'],
        'production_model_changed': False, 'classifier_fitting_performed': True,
        'temperature_fitting_performed': False, 'conformal_threshold_fitting_performed': False,
        'limitations': LIMITATIONS}
    ensure_aggregate(report)
    return report


def ensure_aggregate(report):
    json.dumps(report, allow_nan=False)
    def check(value):
        if isinstance(value, dict):
            if {'image_id', 'image_path', 'subject_id', 'patient_id'} & value.keys():
                raise ValueError('Public report contains private identities')
            for item in value.values():
                check(item)
        elif isinstance(value, list):
            for item in value:
                check(item)
        elif isinstance(value, str) and (re.search(r'[A-Za-z]:[\\/]', value) or value.startswith(('/', '\\\\'))):
            raise ValueError('Public report contains absolute paths')
    check(report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--summarize', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config = QualityConfig(**yaml.safe_load((root/CONFIG).read_text(encoding='utf-8')))
    if asdict(config) != asdict(QualityConfig()):
        raise ValueError('Study config differs from predeclared fixed defaults')
    config.validate()
    frame, pop = population(root)
    warm = warm_bindings(root)
    runtime_path = root/f'artifacts/{STUDY}/runtime.json'
    if not args.execute and not args.summarize:
        print(json.dumps({'status': 'input_readiness_only', 'population': pop, 'warm_artifact_bindings': warm,
                          'training_or_model_inference_performed': False, 'config': asdict(config)}))
        return
    provenance = source_provenance()
    if args.execute:
        if provenance['working_tree_dirty']:
            raise ValueError('Commit tested source/protocol/config and require a clean tree before learning')
        if runtime_path.exists():
            if not args.resume:
                raise ValueError('Existing execution requires explicit --resume')
            runtime = read_json(runtime_path)
            if runtime['source_provenance'] != provenance or runtime['population'] != pop or runtime['warm_artifacts'] != warm:
                raise ValueError('Resume inputs/source differ from original execution')
        else:
            runtime = {'source_provenance': provenance, 'population': pop, 'warm_artifacts': warm,
                'config_sha256': sha256_file(root/CONFIG), 'protocol_sha256': sha256_file(root/PROTOCOL),
                'preserved': preserved_files(root), 'warm_start_reproduction': preflight(root, frame, config)}
            write_json(runtime_path, runtime)
        for seed in SEEDS:
            for arm in ARMS:
                train_quality(root/'data/processed/deepdrid/development_v1.csv',
                    root/f'checkpoints/repeated_seeds_v1/quality_seed{seed}/best.pt',
                    root/f'checkpoints/{STUDY}/seed{seed}_{arm}', root/f'artifacts/{STUDY}/input_cache',
                    config=replace(config, seed=seed), arm=arm, resume=args.resume)
    else:
        runtime = read_json(runtime_path)
    for name, expected in runtime['source_provenance']['source_file_sha256'].items():
        if sha256_file(root/name) != expected:
            raise ValueError('Scientific execution source changed before aggregation')
    for name, expected in runtime['preserved'].items():
        if sha256_file(root/name) != expected:
            raise ValueError('Preserved artifact changed: '+name)
    report = summarize(root, runtime, frame, config)
    target = root/f'reports/{STUDY}/evaluation.json'
    if target.exists() and read_json(target) != report:
        raise ValueError('Existing measured quality report differs; do not overwrite')
    if not target.exists():
        write_json(target, report)
    print(json.dumps({'event': 'quality_study_completed', 'source_commit': report['source_commit'],
                      'evaluation_sha256': sha256_file(target), 'summary': report['summary']}), flush=True)


if __name__ == '__main__':
    main()
