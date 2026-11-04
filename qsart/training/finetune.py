"""CPU full-parameter fine-tuning with fixed BatchNorm running statistics.

Canonical uint8 inputs avoid repeated original-image decoding, without applying
geometry twice. These caches accept fitting/selection rows only. The caller must
predeclare its study and retain independent calibration and test populations.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch import nn

from qsart.models.checkpoint import FORMAT_VERSION, load_checkpoint, sha256_file
from qsart.preprocessing import preprocess_image
from qsart.training.data import MEAN, STD, read_manifest, split_fingerprint
from qsart.training.transfer import _head_metrics


INITIAL_APTOS_SHA256 = '52574e61db4b68308348aab3794bcc64c55f6d47b0d02932955286c35d190407'
CACHE_SCHEMA = 'qsart.canonical_uint8_inputs.v1'
TRAINING_STRATEGY = 'full_parameter_finetune_frozen_batchnorm_buffers_v1'
RNG_VERSION = 'seed_after_cache_and_initialization_v1'


@dataclass
class FinetuneConfig:
    image_size: int = 224
    epochs: int = 4
    patience: int = 4
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
        for name in ('image_size', 'epochs', 'patience', 'batch_size', 'cpu_threads', 'cache_workers'):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError('Positive integer fine-tuning size/epoch/batch/thread values required')
        if self.image_size < 32 or self.cache_workers > 2:
            raise ValueError('Fine-tuning image size must be >=32 and cache workers at most two')
        if (type(self.seed) is not int or not 0 <= self.seed < 2**63 or
                type(self.horizontal_flip) is not bool or type(self.channels_last) is not bool):
            raise ValueError('Valid integer seed and boolean flip setting required')
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
               for value in (self.backbone_lr, self.head_lr, self.weight_decay)):
            raise ValueError('Finite fine-tuning optimizer values required')
        if min(self.backbone_lr, self.head_lr) <= 0 or self.weight_decay < 0:
            raise ValueError('Positive learning rates and nonnegative weight decay required')


def _canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def source_provenance():
    root = Path(__file__).resolve().parents[2]
    files = ('qsart/training/finetune.py', 'qsart/training/data.py', 'qsart/training/transfer.py', 'qsart/models/networks.py',
             'qsart/models/checkpoint.py', 'qsart/preprocessing/fundus.py', 'scripts/run_aptos_finetune.py')
    hashes = {name: sha256_file(root/name) for name in files if (root/name).is_file()}
    try:
        revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True,
                                           stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=root, text=True,
                                            stderr=subprocess.DEVNULL).strip())
    except (OSError, subprocess.CalledProcessError):
        revision, dirty = 'uncommitted', True
    return {'git_revision': revision, 'working_tree_dirty': dirty,
            'source_file_sha256': hashes, 'source_sha256': _canonical_hash(hashes)}


def _named_tensor_hash(values):
    digest = hashlib.sha256()
    for name, tensor in sorted(values):
        tensor = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def parameter_hash(module):
    return _named_tensor_hash(module.named_parameters())


def batchnorm_buffer_hash(module):
    values = []
    for prefix, layer in module.named_modules():
        if isinstance(layer, nn.modules.batchnorm._BatchNorm):
            values.extend((f'{prefix}.{name}', value) for name, value in layer.named_buffers(recurse=False))
    return _named_tensor_hash(values)


def train_with_fixed_batchnorm(model):
    """Keep dropout/stochastic-depth active, but freeze running BN buffers."""
    model.train()
    for layer in model.modules():
        if isinstance(layer, nn.modules.batchnorm._BatchNorm):
            layer.eval()


def normalize_cached(images, mean=None, std=None):
    """Apply only normalization, once, to already cropped/letterboxed RGB bytes."""
    if not isinstance(images, torch.Tensor) or images.dtype != torch.uint8 or images.ndim != 4 or images.shape[-1] != 3:
        raise ValueError('Canonical inputs require uint8 NHWC RGB tensors')
    # NumPy float32 division is the established image_to_tensor operation order.
    array = images.numpy().astype(np.float32) / 255.0
    values = torch.from_numpy(array).permute(0, 3, 1, 2)
    return (values-torch.tensor(mean or MEAN)[None, :, None, None]) / torch.tensor(std or STD)[None, :, None, None]


def _validate_training_frame(frame, *, selected_only=False):
    required = {'image_path', 'split', 'dr_grade'}
    allowed = {'train', 'validation'} if selected_only else {'train', 'validation', 'calibration'}
    if frame.empty or not required.issubset(frame) or not frame.split.isin(allowed).all():
        raise ValueError('Fine-tuning accepts development rows only; no holdout/test/confirmation')
    if selected_only and set(frame.split) != {'train', 'validation'}:
        raise ValueError('Canonical fitting cache requires nonempty train and validation only')
    if not pd.to_numeric(frame.dr_grade, errors='coerce').isin(range(5)).all():
        raise ValueError('Verified integer DR grades 0 through 4 required')
    if 'official_split' in frame and not frame.official_split.eq('official_train').all():
        raise ValueError('Only official-training source records accepted')
    if frame.image_path.astype(str).str.casefold().duplicated().any():
        raise ValueError('Duplicate source image paths')
    for column in ('split_group_id', 'sha256', 'subject_id', 'eye_id'):
        if column in frame and frame.dropna(subset=[column]).groupby(column).split.nunique().gt(1).any():
            raise ValueError('Source fingerprint/group crosses development partitions')


def prepare_input_cache(frame, manifest, cache_dir, *, image_size=224, workers=2):
    _validate_training_frame(frame, selected_only=True)
    if type(image_size) is not int or image_size < 32 or workers not in (1, 2):
        raise ValueError('Canonical cache size >=32 and one/two workers required')
    content = hashlib.sha256()
    for row in frame.itertuples():
        actual = sha256_file(row.image_path)
        expected = getattr(row, 'sha256', None)
        if expected is not None and pd.notna(expected) and str(expected) != actual:
            raise ValueError('Input image changed since manifest verification')
        content.update(f'{row.Index}:{row.split}:{actual}\n'.encode())
    metadata = {'schema': CACHE_SCHEMA, 'manifest_sha256': sha256_file(manifest),
        'input_content_sha256': content.hexdigest(), 'image_size': image_size,
        'preprocessing': 'fundus_crop_rgb_v1', 'normalization_mean': MEAN, 'normalization_std': STD,
        'source_sha256': source_provenance()['source_sha256'],
        'manifest_rows': frame.index.astype(int).tolist(), 'partitions': frame.split.tolist()}
    key = _canonical_hash(metadata)
    path = Path(cache_dir)/f'canonical_uint8_{key}.pt'
    reused = path.is_file()
    started = time.perf_counter()
    if reused:
        cached = torch.load(path, map_location='cpu', weights_only=True)
        if cached.get('metadata') != metadata or cached.get('cache_key') != key:
            raise ValueError('Canonical cache provenance mismatch')
    else:
        images = torch.empty((len(frame), image_size, image_size, 3), dtype=torch.uint8)

        def canonical(path):
            with Image.open(path) as image:
                return np.asarray(preprocess_image(image, image_size=image_size, crop=True).convert('RGB')).copy()

        with ThreadPoolExecutor(max_workers=workers) as pool:
            for index, array in enumerate(pool.map(canonical, frame.image_path)):
                images[index] = torch.from_numpy(array)
                if index % 200 == 0:
                    print(json.dumps({'event': 'canonical_input_cache', 'processed': index+1, 'images': len(frame)}), flush=True)
        cached = {'metadata': metadata, 'cache_key': key, 'images': images,
                  'tensor_sha256': _named_tensor_hash([('images', images)])}
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix('.tmp')
        torch.save(cached, temporary)
        temporary.replace(path)
    images = cached.get('images')
    if (not isinstance(images, torch.Tensor) or images.dtype != torch.uint8 or
            tuple(images.shape) != (len(frame), image_size, image_size, 3) or
            cached.get('tensor_sha256') != _named_tensor_hash([('images', images)])):
        raise ValueError('Canonical cached pixels/shape/hash mismatch')
    information = {'cache_key': key, 'cache_file_sha256': sha256_file(path), 'reused': reused,
        'preparation_or_verification_seconds': time.perf_counter()-started, 'tensor_sha256': cached['tensor_sha256'],
        'shape': list(images.shape), 'metadata': metadata}
    print(json.dumps({'event': 'canonical_cache_ready', 'images': len(frame), 'reused': reused, 'cache_key': key}), flush=True)
    return cached, information


def _evaluate(model, images, targets, batch_size, channels_last=True):
    model.eval()
    logits = []
    with torch.inference_mode():
        for indices in torch.arange(len(images)).split(batch_size):
            batch = normalize_cached(images[indices]).contiguous(
                memory_format=torch.channels_last if channels_last else torch.contiguous_format)
            logits.append(model(batch)['dr_grade'].detach().cpu())
    values = torch.cat(logits)
    return values, _head_metrics(values, targets, 5)


def fit_finetune(model, cache, frame, output, config, *, base_metadata, resume_last=False):
    """Fit supplied supervised model; cache creation never determines RNG state."""
    config.validate()
    _validate_training_frame(frame, selected_only=True)
    output = Path(output)
    if output.exists() and not resume_last:
        raise ValueError('Fine-tuning output already exists; preserve completed/partial runs')
    images = cache['images']
    if len(images) != len(frame) or cache['metadata']['manifest_rows'] != frame.index.astype(int).tolist():
        raise ValueError('Canonical input cache and fitting manifest order differ')
    if tuple(images.shape[1:]) != (config.image_size, config.image_size, 3):
        raise ValueError('Canonical input cache and model size differ')
    model.requires_grad_(True)
    model.to(memory_format=torch.channels_last if config.channels_last else torch.contiguous_format)
    if not all(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError('All learnable model parameters must be trainable')
    initial_backbone = parameter_hash(model.backbone)
    initial_bn = batchnorm_buffer_hash(model)
    train_mask, val_mask = (torch.tensor(frame.split.eq(name).to_numpy()) for name in ('train', 'validation'))
    train_indices, val_indices = train_mask.nonzero().flatten(), val_mask.nonzero().flatten()
    targets = torch.tensor(frame.dr_grade.to_numpy(), dtype=torch.long)
    counts = torch.bincount(targets[train_indices], minlength=5).float()
    weights = torch.zeros(5)
    present = counts > 0
    weights[present] = counts.sum()/(5*counts[present])
    optimizer = torch.optim.AdamW([
        {'params': model.backbone.parameters(), 'lr': config.backbone_lr},
        {'params': model.heads.parameters(), 'lr': config.head_lr}], weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss(weight=weights)
    # StochasticDepth uses global RNG. Batch order/flips use their own seeded stream.
    torch.manual_seed(config.seed)
    generator = torch.Generator().manual_seed(config.seed)
    history, steps, stale, best_score = [], 0, 0, (-math.inf, -math.inf)
    start_epoch = 0
    if resume_last:
        if not output.is_dir() or not (output/'last.pt').is_file() or not (output/'best.pt').is_file():
            raise ValueError('Resume requires the existing last and selected sibling checkpoints')
        old = torch.load(output/'last.pt', map_location='cpu', weights_only=True)
        previous = old.get('metadata', {})
        for field in ('manifest_sha256', 'initial_checkpoint_sha256', 'canonical_cache_key',
                      'source_sha256', 'data_purpose', 'backbone', 'split_fingerprints'):
            if previous.get(field) != base_metadata.get(field):
                raise ValueError(f'Resume fine-tuning provenance differs: {field}')
        if (previous.get('training_config') != asdict(config) or previous.get('training_strategy') != TRAINING_STRATEGY or
                previous.get('initial_backbone_parameters_sha256') != initial_backbone or
                previous.get('initial_batchnorm_buffers_sha256') != initial_bn):
            raise ValueError('Resume optimizer/preprocessing/initialization configuration differs')
        selected = torch.load(output/'best.pt', map_location='cpu', weights_only=True)
        if any(selected.get('metadata', {}).get(field) != previous.get(field) for field in
               ('manifest_sha256', 'initial_checkpoint_sha256', 'source_sha256', 'training_config')):
            raise ValueError('Resume selected sibling checkpoint provenance differs')
        best_row = max(old['history'], key=lambda row: (row['validation_macro_f1'], -row['validation_nll']))
        expected_score = (best_row['validation_macro_f1'], -best_row['validation_nll'])
        if (selected['epoch'] != best_row['epoch']-1 or
                tuple(selected['best_selection_score']) != tuple(old['best_selection_score']) or
                tuple(old['best_selection_score']) != expected_score):
            raise ValueError('Resume selected sibling checkpoint rank differs from completed history')
        exported = json.loads((output/'training_metadata.json').read_text(encoding='utf-8'))
        predictions = pd.read_csv(output/'selection_predictions.csv')
        if (exported != selected['metadata'] or
                len(predictions) != len(val_indices) or 'epoch' not in predictions or
                not predictions.epoch.eq(selected['epoch']+1).all() or
                not np.array_equal(predictions.manifest_row.to_numpy(), frame.index[val_mask.numpy()])):
            raise ValueError('Resume selected metadata/prediction exports disagree with the selected epoch')
        model.load_state_dict(old['model_state'], strict=True)
        optimizer.load_state_dict(old['optimizer_state'])
        scheduler.load_state_dict(old['scheduler_state'])
        history, steps, stale = old['history'], previous['optimizer_steps'], old['stale_epochs']
        start_epoch, best_score = old['epoch']+1, tuple(old['best_selection_score'])
        prior_logits = pd.read_csv(output/'selection_epoch_predictions.csv')
        if (len(history) != start_epoch or len(prior_logits) != len(val_indices)*start_epoch or
                set(prior_logits.epoch) != set(range(1, start_epoch+1))):
            raise ValueError('Resume epoch score evidence differs from the last completed checkpoint')
        if batchnorm_buffer_hash(model) != initial_bn:
            raise ValueError('Resume checkpoint has changed BatchNorm buffers')
        generator.set_state(old['loader_rng_state'])
        torch.set_rng_state(old['torch_rng_state'])
    else:
        output.mkdir(parents=True)
    for epoch in range(start_epoch, config.epochs):
        if stale >= config.patience:
            break
        train_with_fixed_batchnorm(model)
        permutation = train_indices[torch.randperm(len(train_indices), generator=generator)]
        losses, flips_count = [], 0
        epoch_backbone_lr, epoch_head_lr = [group['lr'] for group in optimizer.param_groups]
        for indices in permutation.split(config.batch_size):
            batch = normalize_cached(images[indices]).contiguous(
                memory_format=torch.channels_last if config.channels_last else torch.contiguous_format)
            if config.horizontal_flip:
                flipped = torch.rand(len(batch), generator=generator) < .5
                batch[flipped] = batch[flipped].flip(-1)
                flips_count += int(flipped.sum())
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch)['dr_grade']
            loss = criterion(logits, targets[indices])
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite fine-tuning loss')
            loss.backward()
            optimizer.step()
            steps += 1
            losses.append(float(loss.detach()))
        if batchnorm_buffer_hash(model) != initial_bn:
            raise ValueError('Fine-tuning changed fixed BatchNorm running buffers')
        logits, metrics = _evaluate(model, images[val_indices], targets[val_indices], config.batch_size, config.channels_last)
        changed = parameter_hash(model.backbone) != initial_backbone
        if not changed:
            raise ValueError('No supervised backbone-parameter change was observed')
        score = (metrics['macro_f1'], -metrics['nll'])
        improved = score > best_score
        best_score, stale = (score, 0) if improved else (best_score, stale+1)
        row = {'epoch': epoch+1, 'train_loss': float(np.mean(losses)), 'validation_macro_f1': metrics['macro_f1'],
            'validation_accuracy': metrics['accuracy'], 'validation_nll': metrics['nll'],
            'backbone_learning_rate': epoch_backbone_lr, 'head_learning_rate': epoch_head_lr,
            'train_horizontal_flips': flips_count, 'optimizer_steps': steps, 'selected': improved}
        history.append(row)
        scheduler.step()
        epoch_predictions = pd.DataFrame({'epoch': epoch+1, 'manifest_row': frame.index[val_mask.numpy()],
            'true_dr_grade': targets[val_indices].numpy()})
        for grade in range(5):
            epoch_predictions[f'logit_{grade}'] = logits[:, grade].numpy()
        epoch_predictions.to_csv(output/'selection_epoch_predictions.csv', mode='a', header=epoch == 0, index=False)
        metadata = {**base_metadata, 'trained': True, 'optimizer_steps': steps, 'task': 'dr',
            'heads': {'dr_grade': 5}, 'image_size': config.image_size, 'normalization_mean': MEAN, 'normalization_std': STD,
            'preprocessing': 'fundus_crop_rgb_v1', 'seed': config.seed, 'training_config': asdict(config),
            'training_strategy': TRAINING_STRATEGY, 'rng_version': RNG_VERSION, 'backbone_frozen': False,
            'memory_format': 'channels_last' if config.channels_last else 'contiguous',
            'batchnorm_running_statistics': 'fixed', 'batchnorm_affine_trainable': True,
            'backbone_parameters_changed': changed, 'batchnorm_buffers_unchanged': True,
            'initial_backbone_parameters_sha256': initial_backbone,
            'backbone_parameters_sha256': parameter_hash(model.backbone),
            'initial_batchnorm_buffers_sha256': initial_bn, 'batchnorm_buffers_sha256': batchnorm_buffer_hash(model),
            'trainable_parameters': sum(value.numel() for value in model.parameters()),
            'class_counts': {'dr_grade': counts.int().tolist()}, 'selection_metric': 'macro_f1_then_unweighted_nll',
            'selection_metrics': {'dr_grade': metrics}, 'epoch': epoch+1,
            'model_version': f'aptos-finetune-v1-efficientnet_b0-seed{config.seed}-epoch{epoch+1}'}
        checkpoint = {'format_version': FORMAT_VERSION, 'metadata': metadata,
            'model_state': {name: value.detach().cpu() for name, value in model.state_dict().items()},
            'optimizer_state': optimizer.state_dict(), 'scheduler_state': scheduler.state_dict(),
            'epoch': epoch, 'history': history, 'best_selection_score': list(best_score), 'stale_epochs': stale,
            'loader_rng_state': generator.get_state(), 'torch_rng_state': torch.get_rng_state()}
        torch.save(checkpoint, output/'last.tmp')
        (output/'last.tmp').replace(output/'last.pt')
        if improved:
            torch.save(checkpoint, output/'best.tmp')
            (output/'best.tmp').replace(output/'best.pt')
            (output/'training_metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
            (output/'selection_metrics.json').write_text(json.dumps({'dr_grade': metrics}, indent=2), encoding='utf-8')
            predictions = pd.DataFrame({'epoch': epoch+1, 'manifest_row': frame.index[val_mask.numpy()],
                'true_dr_grade': targets[val_indices].numpy(), 'predicted_dr_grade': logits.argmax(1).numpy()})
            for grade in range(5):
                predictions[f'dr_grade_p{grade}'] = logits.softmax(1)[:, grade].numpy()
            predictions.to_csv(output/'selection_predictions.csv', index=False)
        pd.DataFrame(history).to_csv(output/'training_history.csv', index=False)
        print(json.dumps({'event': 'finetune_epoch', 'seed': config.seed, **row}), flush=True)
        if stale >= config.patience:
            break
    selected = torch.load(output/'best.pt', map_location='cpu', weights_only=True)
    result = {'best_checkpoint': str(output/'best.pt'), 'epochs_executed': len(history),
        'total_optimizer_steps': steps, 'selected_epoch': selected['epoch']+1,
        'selected_metrics': selected['metadata']['selection_metrics'], 'data_purpose': base_metadata['data_purpose'],
        'backbone_parameters_changed': selected['metadata']['backbone_parameters_changed'],
        'batchnorm_buffers_unchanged': selected['metadata']['batchnorm_buffers_unchanged']}
    (output/'training_result.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result


def train_finetune(manifest, initial_checkpoint, output, cache_dir, *, config=None, data_purpose='real_research',
                   resume_last=False):
    config = config or FinetuneConfig()
    config.validate()
    if data_purpose not in {'real_research', 'software_test'}:
        raise ValueError('Explicit real-research or software-test data purpose required')
    frame = read_manifest(manifest)
    _validate_training_frame(frame)
    if not {'train', 'validation'}.issubset(set(frame.split)):
        raise ValueError('Both fitting and selection-validation partitions required')
    torch.set_num_threads(config.cpu_threads)
    model, original = load_checkpoint(initial_checkpoint, allow_software_test=data_purpose == 'software_test')
    previous = original['metadata']
    if (previous['data_purpose'] != data_purpose or previous['backbone'] != 'efficientnet_b0' or
            previous['task'] != 'dr' or previous['heads'] != {'dr_grade': 5}):
        raise ValueError('Initial artifact is not a matching supervised EfficientNet-B0 DR model')
    if (previous['image_size'] != config.image_size or previous['normalization_mean'] != MEAN or
            previous['normalization_std'] != STD or previous.get('preprocessing') != 'fundus_crop_rgb_v1'):
        raise ValueError('Initial artifact and declared preprocessing differ')
    initial_sha = sha256_file(initial_checkpoint)
    if data_purpose == 'real_research' and initial_sha != INITIAL_APTOS_SHA256:
        raise ValueError('Real fine-tuning must start from the declared APTOS selected artifact')
    selected = frame.loc[frame.split.isin(['train', 'validation'])].copy()
    cache, cache_info = prepare_input_cache(selected, manifest, cache_dir,
        image_size=config.image_size, workers=config.cache_workers)
    provenance = source_provenance()
    base_metadata = {'data_purpose': data_purpose, 'backbone': 'efficientnet_b0',
        'manifest_sha256': sha256_file(manifest), 'initial_checkpoint_sha256': initial_sha,
        'initial_model_version': previous['model_version'], 'code_version': provenance['git_revision'],
        'code_provenance': provenance, 'source_sha256': provenance['source_sha256'],
        'split_fingerprints': {str(name): split_fingerprint(part) for name, part in frame.groupby('split')},
        'split_counts': {str(name): len(part) for name, part in frame.groupby('split')},
        'canonical_cache_key': cache_info['cache_key'], 'canonical_cache_provenance': cache_info,
        'confirmation_used': False, 'calibration_used_for_fitting': False,
        'torch_version': str(torch.__version__)}
    return fit_finetune(model, cache, selected, output, config, base_metadata=base_metadata, resume_last=resume_last)
