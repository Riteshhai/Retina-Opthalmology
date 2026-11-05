"""Matched quality-head/backbone study with an atomic, resumable epoch state.

Real fitting is confined to the frozen DeepDRiD development manifest. Generated
software fixtures use a separate marker and cannot be loaded in production.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
import torch
from torch import nn

from qsart.data.development import add_defect_targets
from qsart.models.checkpoint import FORMAT_VERSION, load_checkpoint, sha256_file
from qsart.training.data import MEAN, STD, read_manifest, split_fingerprint
from qsart.training.finetune import (batchnorm_buffer_hash, normalize_cached, parameter_hash,
                                     prepare_input_cache, train_with_fixed_batchnorm)

HEADS = {'quality_grade': 2, 'clarity_class': 5, 'field_definition_class': 5, 'artifact_class': 6}
ARMS = ('frozen', 'adapted')
SEEDS = (2026, 2027, 2028)
MANIFEST_SHA256 = '45e9f2d34ed54a12056b57a808d03cfaf29ccc970d148e9eb107b79cf5ac3e0d'
WARM_SHA256 = {
    2026: '67368a38922423e38d541f96bfa7a49ce740d7b172eef7389073367cee6ffbc1',
    2027: '545c367d226951d63e9e0170407289ffa0474686f3355680a7c32510fd9aa9ca',
    2028: '09e19a910c0c6d9c80bb743aa47c2476d1b6091802421ea7210dd458ee052285',
}
STRATEGY = 'matched_four_head_quality_adaptation_v1'
SELECTOR = 'mean_four_head_macro_f1_then_mean_unweighted_nll'
RNG_VERSION = 'independent_order_flip_generator_after_cache_v1'


@dataclass
class QualityConfig:
    image_size: int = 224
    epochs: int = 15
    batch_size: int = 16
    backbone_lr: float = 1e-4
    head_lr: float = 3e-4
    weight_decay: float = 1e-4
    seed: int = 2026
    cpu_threads: int = 4
    cache_workers: int = 2
    horizontal_flip: bool = True
    channels_last: bool = True

    def validate(self):
        for name in ('image_size', 'epochs', 'batch_size', 'cpu_threads', 'cache_workers'):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError('Positive integer size/epoch/batch/thread values required')
        if self.image_size < 32 or self.cache_workers > 2:
            raise ValueError('Image size >=32 and at most two cache workers required')
        if type(self.seed) is not int or not 0 <= self.seed < 2**63:
            raise ValueError('Valid nonnegative integer seed required')
        if type(self.horizontal_flip) is not bool or type(self.channels_last) is not bool:
            raise ValueError('Boolean augmentation/layout settings required')
        for name in ('backbone_lr', 'head_lr', 'weight_decay'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError('Finite optimizer values required')
        if min(self.backbone_lr, self.head_lr) < 1e-6 or self.weight_decay < 0:
            raise ValueError('Learning rates >= cosine minimum 1e-6 and nonnegative decay required')


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def source_provenance():
    root = Path(__file__).resolve().parents[2]
    names = ('qsart/training/quality_adaptation.py', 'qsart/training/finetune.py',
             'qsart/training/data.py', 'qsart/training/transfer.py', 'qsart/data/development.py',
             'qsart/models/networks.py', 'qsart/models/checkpoint.py', 'qsart/preprocessing/fundus.py',
             'scripts/run_quality_adaptation.py', 'configs/quality_adaptation_v1.yaml',
             'docs/research/QUALITY_ADAPTATION_V1_PROTOCOL.md')
    hashes = {name: sha256_file(root/name) for name in names}
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    dirty = bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=root, text=True).strip())
    return {'git_revision': revision, 'working_tree_dirty': dirty,
            'source_file_sha256': hashes, 'source_sha256': canonical_hash(hashes)}


def validate_frame(frame, *, selected_only=False, data_purpose='real_research'):
    allowed = {'train', 'validation'} if selected_only else {'train', 'validation', 'calibration'}
    if (frame.empty or not {'image_path', 'split', *HEADS}.issubset(frame) or
            not frame.split.isin(allowed).all() or not {'train', 'validation'}.issubset(set(frame.split))):
        raise ValueError('Quality fitting accepts nonempty development train/validation rows only')
    if data_purpose not in {'real_research', 'software_test'}:
        raise ValueError('Explicit real_research or software_test purpose required')
    if data_purpose == 'real_research':
        required = {'official_split', 'dataset_source', 'subject_id', 'split_group_id', 'sha256',
                    'annotation_status', 'eligible_for_training', 'readable', 'exists',
                    'clarity', 'field_definition', 'artifact'}
        if not required.issubset(frame):
            raise ValueError('Verified source/status/group/defect columns required')
        if (not frame.dataset_source.eq('DeepDRiD_regular_fundus').all() or
                not frame.official_split.eq('official_train').all() or
                not frame.annotation_status.eq('verified').all()):
            raise ValueError('Only verified official-training DeepDRiD records accepted')
        for name in ('eligible_for_training', 'readable', 'exists'):
            if not frame[name].astype(str).str.lower().isin(['true', '1']).all():
                raise ValueError('Ineligible/unreadable/missing images rejected')
        for name in ('subject_id', 'split_group_id', 'sha256'):
            if frame[name].isna().any():
                raise ValueError('Complete source fingerprints and known grouping required')
        mapped = add_defect_targets(frame)
        if any(not np.array_equal(mapped[name].to_numpy(), frame[name].to_numpy()) for name in HEADS if name != 'quality_grade'):
            raise ValueError('Mapped natural defect labels disagree with source codes')
    for head, classes in HEADS.items():
        numbers = pd.to_numeric(frame[head], errors='coerce')
        if not numbers.isin(range(classes)).all():
            raise ValueError(f'Complete integer labels required for {head}')
    if frame.image_path.astype(str).str.casefold().duplicated().any():
        raise ValueError('Duplicate source image paths')
    for name in ('subject_id', 'eye_id', 'sha256', 'split_group_id'):
        if name in frame and frame.dropna(subset=[name]).groupby(name).split.nunique().gt(1).any():
            raise ValueError('Source fingerprint/group crosses development partitions')


def metrics_from_logits(logits, truth, classes):
    values = np.asarray(logits, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.int64)
    if (values.shape != (len(truth), classes) or not len(truth) or not np.isfinite(values).all()
            or not np.isin(truth, range(classes)).all()):
        raise ValueError('Finite complete logits and in-range labels required')
    shifted = values-values.max(1, keepdims=True)
    logp = shifted-np.log(np.exp(shifted).sum(1, keepdims=True))
    probability = np.exp(logp)
    predicted = probability.argmax(1)
    matrix = np.bincount(truth*classes+predicted, minlength=classes*classes).reshape(classes, classes)
    support = matrix.sum(1)
    diagonal = matrix.diagonal()
    precision = np.divide(diagonal, matrix.sum(0), out=np.zeros(classes, dtype=float), where=matrix.sum(0)>0)
    recall = np.divide(diagonal, support, out=np.zeros(classes, dtype=float), where=support>0)
    f1 = np.divide(2*diagonal, support+matrix.sum(0), out=np.zeros(classes, dtype=float), where=support+matrix.sum(0)>0)
    confidence = probability.max(1)
    correct = predicted == truth
    bins = np.minimum((confidence*10).astype(int), 9)
    ece = sum(float(np.abs(correct[bins==b].mean()-confidence[bins==b].mean())*(bins==b).mean())
              for b in range(10) if (bins==b).any())
    result = {'n': len(truth), 'accuracy': float(correct.mean()), 'macro_f1': float(f1.mean()),
              'nll': float(-logp[np.arange(len(truth)), truth].mean()), 'ece_10': ece,
              'brier': float(np.square(probability-np.eye(classes)[truth]).sum(1).mean()),
              'confusion_matrix': matrix.tolist(), 'precision': precision.tolist(), 'recall': recall.tolist(),
              'class_f1': f1.tolist(), 'support': support.tolist()}
    if classes == 2:
        positive = truth == 0
        result['poor_class_index'] = 0
        result['poor_auroc'] = float(roc_auc_score(positive, probability[:, 0])) if np.unique(positive).size == 2 else None
        result['poor_auprc'] = float(average_precision_score(positive, probability[:, 0])) if positive.any() else None
        result['poor_sensitivity'] = float(recall[0])
        result['poor_specificity'] = float(recall[1])
    return result


def selection_score(metrics):
    return (float(np.mean([metrics[name]['macro_f1'] for name in HEADS])),
            -float(np.mean([metrics[name]['nll'] for name in HEADS])))


def evaluate(model, images, targets, config):
    model.eval()
    chunks = {name: [] for name in HEADS}
    with torch.inference_mode():
        for indices in torch.arange(len(images)).split(config.batch_size):
            batch = normalize_cached(images[indices]).contiguous(
                memory_format=torch.channels_last if config.channels_last else torch.contiguous_format)
            output = model(batch)
            for name in HEADS:
                chunks[name].append(output[name].detach().cpu())
    logits = {name: torch.cat(values) for name, values in chunks.items()}
    metrics = {name: metrics_from_logits(logits[name].numpy(), targets[name].numpy(), size)
               for name, size in HEADS.items()}
    return logits, metrics


def _atomic_torch(value, path):
    temporary = path.with_suffix('.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    temporary.replace(path)


def verify_seal(folder, expected_binding=None):
    folder = Path(folder)
    seal = json.loads((folder/'completion.json').read_text(encoding='utf-8'))
    if expected_binding is not None and seal['binding_sha256'] != expected_binding:
        raise ValueError('Completed run provenance differs from requested configuration/source/arm')
    for name, expected in seal['artifact_sha256'].items():
        if sha256_file(folder/name) != expected:
            raise ValueError('Completed run artifact changed: '+name)
    return json.loads((folder/'training_result.json').read_text(encoding='utf-8'))


def _predictions(frame, logits, epoch):
    result = {'manifest_row': frame.index.to_numpy(), 'epoch': epoch}
    for name, classes in HEADS.items():
        values = logits[name].double().numpy()
        shifted = values-values.max(1, keepdims=True)
        probability = np.exp(shifted)/np.exp(shifted).sum(1, keepdims=True)
        result[f'true_{name}'] = frame[name].to_numpy(dtype=int)
        result[f'predicted_{name}'] = probability.argmax(1)
        for index in range(classes):
            result[f'{name}_logit{index}'] = values[:, index]
            result[f'{name}_p{index}'] = probability[:, index]
    return pd.DataFrame(result)


def fit_quality(model, cache, frame, output, config, *, arm, base_metadata, resume=False):
    """Fit exactly the declared epochs. State replacement is the epoch commit.

    Optimizer, RNG, histories, all epoch logits and the selected state share one
    atomic file, so an interrupted export never repeats completed updates.
    """
    config.validate()
    purpose = base_metadata.get('data_purpose')
    validate_frame(frame, selected_only=True, data_purpose=purpose)
    if arm not in ARMS:
        raise ValueError('Unknown quality study arm')
    if base_metadata.get('heads') != HEADS or base_metadata.get('poor_class_indices') != [0]:
        raise ValueError('All four quality heads and poor class zero required')
    images = cache['images']
    if (images.dtype != torch.uint8 or tuple(images.shape) != (len(frame), config.image_size, config.image_size, 3)
            or cache['metadata']['manifest_rows'] != frame.index.astype(int).tolist()
            or cache['metadata']['partitions'] != frame.split.tolist()):
        raise ValueError('Canonical inputs do not match supervised frame')
    output = Path(output)
    binding = {'base_metadata': base_metadata, 'config': asdict(config), 'arm': arm, 'strategy': STRATEGY}
    binding_sha = canonical_hash(binding)
    if output.exists():
        if not resume:
            raise ValueError('Quality output already exists; preserve completed/partial runs')
        if (output/'completion.json').exists():
            return verify_seal(output, binding_sha)
        if not (output/'state.pt').exists():
            raise ValueError('Partial run has no committed epoch state')
    else:
        output.mkdir(parents=True)
    torch.set_num_threads(config.cpu_threads)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(config.seed)  # After model/cache initialization, independent of cache miss/hit.
    generator = torch.Generator().manual_seed(config.seed)
    model.backbone.requires_grad_(arm == 'adapted')
    model.heads.requires_grad_(True)
    if config.channels_last:
        model.to(memory_format=torch.channels_last)
    initial = {'backbone_parameter_sha256': parameter_hash(model.backbone),
               'head_parameter_sha256': parameter_hash(model.heads),
               'batchnorm_buffer_sha256': batchnorm_buffer_hash(model)}
    groups = [{'params': list(model.heads.parameters()), 'lr': config.head_lr, 'name': 'heads'}]
    if arm == 'adapted':
        groups.insert(0, {'params': list(model.backbone.parameters()), 'lr': config.backbone_lr, 'name': 'backbone'})
    optimizer = torch.optim.AdamW(groups, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=1e-6)
    train_indices = torch.from_numpy(np.flatnonzero(frame.split.eq('train').to_numpy(dtype=bool, copy=True)))
    val_indices = torch.from_numpy(np.flatnonzero(frame.split.eq('validation').to_numpy(dtype=bool, copy=True)))
    targets = {name: torch.tensor(frame[name].to_numpy(dtype=np.int64, copy=True)) for name in HEADS}
    val_targets = {name: values[val_indices] for name, values in targets.items()}
    weights, losses = {}, {}
    for name, classes in HEADS.items():
        counts = torch.bincount(targets[name][train_indices], minlength=classes)
        if purpose == 'real_research' and not (counts > 0).all():
            raise ValueError('Real study requires all fitting classes observed')
        weight = torch.where(counts > 0, len(train_indices)/(classes*counts.clamp_min(1).float()), 1.)
        weights[name] = weight.tolist()
        losses[name] = nn.CrossEntropyLoss(weight=weight)
    state = None
    if (output/'state.pt').exists():
        state = torch.load(output/'state.pt', map_location='cpu', weights_only=True)
        if state['binding'] != binding or state['initial'] != initial:
            raise ValueError('Resume provenance/configuration/source/arm or initial state mismatch')
        model.load_state_dict(state['model_state'], strict=True)
        optimizer.load_state_dict(state['optimizer_state'])
        scheduler.load_state_dict(state['scheduler_state'])
        generator.set_state(state['loader_rng_state'])
        torch.set_rng_state(state['torch_rng_state'])
    else:
        warm_logits, warm_metrics = evaluate(model, images[val_indices], val_targets, config)
        state = {'binding': binding, 'binding_sha256': binding_sha, 'initial': initial,
                 'epoch': 0, 'history': [], 'epoch_logits': [], 'best': None,
                 'warm_logits': warm_logits, 'warm_metrics': warm_metrics,
                 'model_state': deepcopy(model.state_dict()),
                 'optimizer_state': deepcopy(optimizer.state_dict()),
                 'scheduler_state': deepcopy(scheduler.state_dict()),
                 'loader_rng_state': generator.get_state(), 'torch_rng_state': torch.get_rng_state(),
                 'elapsed_training_seconds': 0.}
        _atomic_torch(state, output/'state.pt')
    for epoch in range(state['epoch']+1, config.epochs+1):
        started = time.perf_counter()
        train_with_fixed_batchnorm(model)
        order = train_indices[torch.randperm(len(train_indices), generator=generator)]
        # Draw all flips independently of optimizer/model execution and in order position.
        flips = torch.rand(len(order), generator=generator) < (.5 if config.horizontal_flip else 0.)
        learning_rates = {group['name']: group['lr'] for group in optimizer.param_groups}
        loss_sum = 0.
        for start in range(0, len(order), config.batch_size):
            indices = order[start:start+config.batch_size]
            batch = normalize_cached(images[indices])
            flipped = flips[start:start+config.batch_size]
            batch[flipped] = batch[flipped].flip(-1)
            batch = batch.contiguous(memory_format=torch.channels_last if config.channels_last else torch.contiguous_format)
            optimizer.zero_grad(set_to_none=True)
            predictions = model(batch)
            loss = torch.stack([losses[name](predictions[name], targets[name][indices]) for name in HEADS]).mean()
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite supervised loss')
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach())*len(indices)
        logits, metrics = evaluate(model, images[val_indices], val_targets, config)
        if batchnorm_buffer_hash(model) != initial['batchnorm_buffer_sha256']:
            raise ValueError('BatchNorm running buffers changed')
        score = selection_score(metrics)
        scheduler.step()
        row = {'epoch': epoch, 'optimizer_steps': epoch*math.ceil(len(train_indices)/config.batch_size),
               'training_loss': loss_sum/len(train_indices), 'learning_rates': learning_rates,
               'selection_metrics': metrics, 'mean_head_macro_f1': score[0], 'mean_head_nll': -score[1],
               'order_sha256': hashlib.sha256(order.numpy().tobytes()).hexdigest(),
               'flip_sha256': hashlib.sha256(flips.numpy().tobytes()).hexdigest(), 'flip_count': int(flips.sum())}
        state['history'].append(row)
        state['epoch_logits'].append(logits)
        state.update(epoch=epoch, model_state=deepcopy(model.state_dict()),
                     optimizer_state=deepcopy(optimizer.state_dict()), scheduler_state=deepcopy(scheduler.state_dict()),
                     loader_rng_state=generator.get_state(), torch_rng_state=torch.get_rng_state())
        if state['best'] is None or score > tuple(state['best']['score']):
            state['best'] = deepcopy({key: state[key] for key in (
                'epoch', 'model_state', 'optimizer_state', 'scheduler_state', 'loader_rng_state', 'torch_rng_state')})
            state['best'].update(score=list(score), selection_metrics=metrics)
        state['elapsed_training_seconds'] += time.perf_counter()-started
        _atomic_torch(state, output/'state.pt')
        print(json.dumps({'event': 'quality_epoch_committed', 'arm': arm, 'seed': config.seed,
                          'epoch': epoch, 'selected_epoch': state['best']['epoch'],
                          'mean_head_macro_f1': score[0], 'mean_head_nll': -score[1]}), flush=True)
    return _export(model, frame, output, config, state, weights)


def _export(model, frame, output, config, state, weights):
    binding = state['binding']
    arm = binding['arm']
    initial = state['initial']
    steps_per_epoch = math.ceil(int(frame.split.eq('train').sum())/config.batch_size)
    backbone_changed = parameter_hash(model.backbone) != initial['backbone_parameter_sha256']
    if backbone_changed != (arm == 'adapted'):
        raise ValueError('Backbone update state disagrees with declared arm')
    if parameter_hash(model.heads) == initial['head_parameter_sha256']:
        raise ValueError('Quality heads did not change')
    if batchnorm_buffer_hash(model) != initial['batchnorm_buffer_sha256']:
        raise ValueError('BatchNorm buffers changed')
    selected = state['best']
    metadata = {}
    for name, snapshot in (('best', selected), ('last', state)):
        metrics = selected['selection_metrics'] if name == 'best' else state['history'][-1]['selection_metrics']
        metadata = {**binding['base_metadata'], 'trained': True, 'task': 'quality',
            'training_strategy': STRATEGY, 'arm': arm, 'training_config': asdict(config), 'seed': config.seed,
            'rng_version': RNG_VERSION, 'image_size': config.image_size,
            'normalization_mean': MEAN, 'normalization_std': STD, 'preprocessing': 'fundus_crop_rgb_v1',
            'epoch': snapshot['epoch'], 'optimizer_steps': snapshot['epoch']*steps_per_epoch,
            'selection_metric': SELECTOR, 'selection_metrics': metrics, 'class_weights': weights,
            'backbone_frozen': arm == 'frozen', 'batchnorm_buffers_unchanged': True,
            'model_version': f'quality-adaptation-v1-{arm}-{binding["base_metadata"]["backbone"]}-seed{config.seed}-epoch{snapshot["epoch"]}',
            'initial_state': initial, 'binding_sha256': state['binding_sha256'],
            'confirmation_used': False, 'calibration_used_for_fitting': False}
        torch.save({'format_version': FORMAT_VERSION, 'metadata': metadata,
                    **{key: snapshot[key] for key in ('model_state', 'optimizer_state', 'scheduler_state',
                                                     'loader_rng_state', 'torch_rng_state')}}, output/f'{name}.pt')
        if name == 'best':
            write_json(output/'training_metadata.json', metadata)
    validation = frame.loc[frame.split.eq('validation')]
    all_predictions = [_predictions(validation, state['warm_logits'], 0)]
    all_predictions.extend(_predictions(validation, values, epoch) for epoch, values in enumerate(state['epoch_logits'], 1))
    pd.concat(all_predictions, ignore_index=True).to_csv(output/'selection_epoch_predictions.csv', index=False, float_format='%.17g')
    _predictions(validation, state['epoch_logits'][selected['epoch']-1], selected['epoch']).to_csv(
        output/'selection_predictions.csv', index=False, float_format='%.17g')
    write_json(output/'training_history.json', state['history'])
    result = {'data_purpose': binding['base_metadata']['data_purpose'], 'arm': arm, 'seed': config.seed,
              'model_version': json.loads((output/'training_metadata.json').read_text(encoding='utf-8'))['model_version'],
              'epochs_executed': state['epoch'], 'selected_epoch': selected['epoch'],
              'total_optimizer_steps': state['epoch']*steps_per_epoch,
              'selected_metrics': selected['selection_metrics'], 'warm_metrics': state['warm_metrics'],
              'mean_head_macro_f1': selected['score'][0], 'mean_head_nll': -selected['score'][1],
              'backbone_parameters_changed': backbone_changed, 'heads_changed': True,
              'batchnorm_buffers_unchanged': True, 'elapsed_training_seconds': state['elapsed_training_seconds'],
              'binding_sha256': state['binding_sha256']}
    write_json(output/'training_result.json', result)
    names = ('state.pt', 'best.pt', 'last.pt', 'training_metadata.json', 'training_history.json',
             'selection_epoch_predictions.csv', 'selection_predictions.csv', 'training_result.json')
    write_json(output/'completion.json', {'binding_sha256': state['binding_sha256'],
               'artifact_sha256': {name: sha256_file(output/name) for name in names}})
    return result


def train_quality(manifest, initial_checkpoint, output, cache_dir, *, config=None,
                  arm='adapted', data_purpose='real_research', resume=False):
    config = config or QualityConfig()
    config.validate()
    frame = read_manifest(manifest)
    validate_frame(frame, data_purpose=data_purpose)  # Reject reserved roles before touching weights/pixels.
    if data_purpose == 'real_research':
        if (config.seed not in SEEDS or sha256_file(manifest) != MANIFEST_SHA256 or
                sha256_file(initial_checkpoint) != WARM_SHA256[config.seed]):
            raise ValueError('Real study requires the frozen development manifest and per-seed warm start')
    model, checkpoint = load_checkpoint(initial_checkpoint, allow_software_test=data_purpose == 'software_test')
    previous = checkpoint['metadata']
    expected = {'task': 'quality', 'heads': HEADS, 'poor_class_indices': [0], 'image_size': config.image_size,
                'normalization_mean': MEAN, 'normalization_std': STD, 'preprocessing': 'fundus_crop_rgb_v1',
                'data_purpose': data_purpose, 'manifest_sha256': sha256_file(manifest),
                'backbone': 'mobilenet_v3_small'}
    if any(previous.get(name) != value for name, value in expected.items()):
        raise ValueError('Quality warm start metadata/preprocessing incompatible')
    selected = frame.loc[frame.split.isin(['train', 'validation'])].copy()
    cache, cache_info = prepare_input_cache(selected, manifest, cache_dir,
        image_size=config.image_size, workers=config.cache_workers)
    provenance = source_provenance()
    # Volatile preparation time/cache-hit flags do not change a run's scientific binding.
    cache_info = {name: value for name, value in cache_info.items() if name not in {'reused', 'preparation_or_verification_seconds'}}
    base = {'data_purpose': data_purpose, 'backbone': previous['backbone'], 'heads': HEADS,
            'poor_class_indices': [0], 'manifest_sha256': sha256_file(manifest),
            'initial_checkpoint_sha256': sha256_file(initial_checkpoint),
            'initial_model_version': previous['model_version'], 'source_provenance': provenance,
            'canonical_cache_provenance': cache_info, 'torch_version': str(torch.__version__),
            'split_fingerprints': {name: split_fingerprint(part) for name, part in frame.groupby('split')},
            'split_counts': {name: len(part) for name, part in frame.groupby('split')}}
    return fit_quality(model, cache, selected, output, config, arm=arm, base_metadata=base, resume=resume)


def paired_component_changes(truth, frozen, adapted, groups, *, draws=2000, seed=20261107):
    """Common whole-component draws, fixed class universe, no refitting/reselection."""
    groups = np.asarray(groups)
    unique = np.unique(groups)
    if not len(unique) or type(draws) is not int or draws < 1:
        raise ValueError('Nonempty groups and positive integer bootstrap draws required')
    if any(len(truth[name]) != len(groups) or len(frozen[name]) != len(groups) or len(adapted[name]) != len(groups) for name in HEADS):
        raise ValueError('Paired outcomes and groups must align')
    members = [np.flatnonzero(groups == value) for value in unique]

    def scores(indices, predictions):
        result = {}
        f1s = []
        for name, classes in HEADS.items():
            y, pred = np.asarray(truth[name])[indices], np.asarray(predictions[name])[indices]
            matrix = np.bincount(y*classes+pred, minlength=classes*classes).reshape(classes, classes)
            denominator = matrix.sum(1)+matrix.sum(0)
            f1 = np.divide(2*matrix.diagonal(), denominator, out=np.zeros(classes), where=denominator>0).mean()
            result[name+'_macro_f1'] = float(f1)
            result[name+'_accuracy'] = float(np.trace(matrix)/len(indices))
            f1s.append(f1)
        result['mean_head_macro_f1'] = float(np.mean(f1s))
        return result

    all_indices = np.arange(len(groups))
    first, second = scores(all_indices, frozen), scores(all_indices, adapted)
    samples = {name: [] for name in first}
    rng = np.random.default_rng(seed)
    digest = hashlib.sha256()
    for _ in range(draws):
        picked = rng.integers(0, len(unique), len(unique))
        digest.update(picked.tobytes())
        indices = np.concatenate([members[index] for index in picked])
        a, b = scores(indices, frozen), scores(indices, adapted)
        for name in samples:
            samples[name].append(b[name]-a[name])
    return {'unit': 'whole_split_group_id', 'components': len(unique), 'draws': draws, 'seed': seed,
            'component_draw_sha256': digest.hexdigest(), 'direction': 'adapted_minus_frozen',
            'metrics': {name: {'difference': second[name]-first[name],
                       'lower_95': float(np.quantile(values, .025)), 'upper_95': float(np.quantile(values, .975))}
                        for name, values in samples.items()}}
