"""Read-only external inputs and ambiguity-sensitive scoring; never training."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
from uuid import uuid4

import numpy as np
from PIL import Image
import torch

from qsart.evaluation import grade_metrics
from qsart.models.checkpoint import sha256_file
from qsart.preprocessing import preprocess_image
from qsart.training.finetune import normalize_cached
from qsart.uncertainty.conformal import coverage_metrics, predict_sets, validate_artifact
from qsart.uncertainty.randomized_ordinal import predict_randomized, validate_randomized

POPULATIONS = ('full_source', 'cross_source_filtered', 'cross_source_filtered_representatives')


def validate_external(frame):
    fields = {'image_id', 'image_path', 'sha256', 'split_group_id', 'dr_grade', 'split',
              'eligible_for_training', 'component_label_conflict'} | {'in_'+name for name in POPULATIONS}
    if frame.empty or not fields.issubset(frame) or frame[list(fields)].isna().any().any():
        raise ValueError('Complete frozen external observation/component records required')
    if (not frame.split.eq('external_holdout').all() or frame.image_id.duplicated().any() or
            frame.image_path.astype(str).str.casefold().duplicated().any() or
            not frame.sha256.astype(str).str.fullmatch('[0-9a-f]{64}').all() or
            frame.dr_grade.dtype.kind not in 'iu' or not frame.dr_grade.isin(range(5)).all()):
        raise ValueError('Only unique, verified external-holdout observations are accepted')
    for field in ['eligible_for_training', 'component_label_conflict']+['in_'+name for name in POPULATIONS]:
        if not frame[field].map(lambda value: isinstance(value, (bool, np.bool_))).all():
            raise ValueError('External eligibility and population masks require actual booleans')
    if frame.eligible_for_training.any() or not frame.in_full_source.all():
        raise ValueError('External observations must remain training-ineligible and preserved')
    if ((frame.in_cross_source_filtered_representatives & ~frame.in_cross_source_filtered).any() or
            frame.groupby('split_group_id').in_cross_source_filtered.nunique().gt(1).any() or
            frame.loc[frame.in_cross_source_filtered_representatives, 'split_group_id'].duplicated().any()):
        raise ValueError('Frozen external masks must be nested and preserve component rules')
    if any(not frame['in_'+name].any() for name in POPULATIONS):
        raise ValueError('Every declared external population must be nonempty')


def pixel_digest(pixels):
    return hashlib.sha256(pixels.contiguous().numpy().tobytes()).hexdigest()


def external_pixels(frame, output, *, workers=2):
    """Build a dedicated external-only one-pass uint8 cache; no label transforms."""
    validate_external(frame)
    if type(workers) is not int or workers not in (1, 2):
        raise ValueError('One or two external decoding workers required')
    for row in frame.itertuples():
        if sha256_file(row.image_path) != row.sha256:
            raise ValueError('External image bytes differ from the frozen manifest')
    metadata = {'scope': 'external_inference_only_no_fitting', 'image_size': 224,
        'observations_sha256': hashlib.sha256('\n'.join(f'{row.image_id}:{row.sha256}' for row in frame.itertuples()).encode()).hexdigest(),
        'preprocessing': 'one_pass_fundus_crop_rgb_letterbox224_v1'}
    output = Path(output)
    if output.exists():
        cache = torch.load(output, map_location='cpu', weights_only=True)
        if cache.get('metadata') != metadata:
            raise ValueError('Existing external cache binding mismatch')
    else:
        def decode(path):
            with Image.open(path) as image:
                return np.asarray(preprocess_image(image, image_size=224, crop=True)).copy()
        pixels = torch.empty((len(frame), 224, 224, 3), dtype=torch.uint8)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for index, values in enumerate(pool.map(decode, frame.image_path)):
                pixels[index] = torch.from_numpy(values)
                if index % 100 == 0:
                    print(json.dumps({'event': 'external_cache', 'images_decoded': index+1, 'images': len(frame)}), flush=True)
        cache = {'metadata': metadata, 'pixels': pixels, 'tensor_sha256': pixel_digest(pixels)}
        output.parent.mkdir(parents=True, exist_ok=True)
        pending = output.with_name(output.name+'.'+uuid4().hex+'.pending')
        torch.save(cache, pending)
        pending.replace(output)
    pixels = cache.get('pixels')
    if (not isinstance(pixels, torch.Tensor) or pixels.dtype != torch.uint8 or
            tuple(pixels.shape) != (len(frame), 224, 224, 3) or pixel_digest(pixels) != cache.get('tensor_sha256')):
        raise ValueError('External canonical pixel cache content mismatch')
    return pixels, {**metadata, 'tensor_sha256': cache['tensor_sha256'], 'cache_file_sha256': sha256_file(output)}


def infer_logits(model, pixels, *, batch_size=16):
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError('Positive inference batch size required')
    model.eval()
    model.to(memory_format=torch.channels_last)
    logits = []
    with torch.inference_mode():
        for indices in torch.arange(len(pixels)).split(batch_size):
            batch = normalize_cached(pixels[indices]).contiguous(memory_format=torch.channels_last)
            logits.append(model(batch)['dr_grade'].detach().cpu())
    values = torch.cat(logits)
    if tuple(values.shape) != (len(pixels), 5) or not torch.isfinite(values).all():
        raise ValueError('External inference must produce finite five-grade logits')
    return values


def ambiguity_labels(frame):
    """All source grades in the ORIGINAL component, before any filtering."""
    allowed = np.zeros((len(frame), 5), dtype=bool)
    for _, indices in frame.reset_index(drop=True).groupby('split_group_id').groups.items():
        positions = list(indices)
        grades = frame.iloc[positions].dr_grade.unique()
        allowed[np.ix_(positions, grades)] = True
    return allowed


def annotation_envelope(sets, allowed):
    allowed = np.asarray(allowed)
    if (allowed.dtype != bool or allowed.ndim != 2 or allowed.shape[1] != 5 or
            len(sets) != len(allowed) or not len(allowed) or not allowed.any(axis=1).all()):
        raise ValueError('Matching nonempty observed-component grade sets required')
    included = np.zeros_like(allowed)
    for index, labels in enumerate(sets):
        if any(not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_))
               or not 0 <= value < 5 for value in labels):
            raise ValueError('Valid predicted grade sets required')
        included[index, labels] = True
    lower = (~allowed | included).all(axis=1)
    upper = (allowed & included).any(axis=1)
    return {'images': len(allowed), 'ambiguous_observations': int((allowed.sum(1) > 1).sum()),
        'conditional_lower_count': int(lower.sum()), 'conditional_upper_count': int(upper.sum()),
        'conditional_lower': float(lower.mean()), 'conditional_upper': float(upper.mean()),
        'interpretation': 'Hypothetical source-annotation sensitivity, not confidence bounds or clinical truth.'}


def grading_views(frame, probabilities):
    allowed = ambiguity_labels(frame)
    result = {}
    for name in POPULATIONS:
        mask = frame['in_'+name].to_numpy(bool)
        truth = frame.loc[mask, 'dr_grade'].to_numpy(int)
        p = probabilities[mask]
        metrics = grade_metrics(truth, p)
        metrics.update(nll=float(-np.log(np.clip(p[np.arange(len(truth)), truth], 1e-15, 1)).mean()), ece_bins=10)
        metrics['source_annotation_sensitivity'] = annotation_envelope([[int(i)] for i in p.argmax(1)], allowed[mask])
        clean = mask & ~frame.component_label_conflict.to_numpy(bool)
        metrics['unambiguous_component_stratum'] = {
            'images': int(clean.sum()), 'label_dependent_diagnostic_only': True,
            'grading': grade_metrics(frame.loc[clean, 'dr_grade'].to_numpy(int), probabilities[clean]) if clean.any() else None}
        result[name] = metrics
    return result


def frozen_comparisons(rows, *, checkpoint_sha256, temperature):
    expected = {(alpha, method, seed) for alpha in (.1, .05) for method, seed in
                [('lac', None), ('ordinal_greedy', None)]+[('ordinal_greedy_randomized', s) for s in range(2026, 2031)]}
    if len(rows) != 14 or {(r['alpha'], r['method'], r['randomization_seed']) for r in rows} != expected:
        raise ValueError('All fourteen fixed source conformal artifacts required')
    for row in rows:
        artifact = row['artifact']
        validator = validate_randomized if row['randomization_seed'] is not None else validate_artifact
        validator(artifact, expected_metadata={'checkpoint_sha256': checkpoint_sha256,
                                              'independent_temperature': temperature})
        if (artifact['alpha'] != row['alpha'] or artifact['method'] != row['method'] or
                artifact['n_calibration_units'] != 254 or artifact['unit'] != 'image' or
                artifact['metadata']['randomization_seed'] != row['randomization_seed'] or
                (row['randomization_seed'] is None and artifact['empty_policy'] != 'allow')):
            raise ValueError('Frozen conformal artifact scope/key mismatch')
    return rows


def external_uniforms(n):
    return {seed: np.random.default_rng(np.random.SeedSequence(seed).spawn(3)[2]).random(n)
            for seed in range(2026, 2031)}


def uncertainty_views(frame, probabilities, rows, uniforms):
    allowed = ambiguity_labels(frame)
    public, private = [], []
    for row in rows:
        artifact, seed = row['artifact'], row['randomization_seed']
        raw = (predict_sets(probabilities, artifact) if seed is None else
               predict_randomized(probabilities, artifact, uniforms[seed], empty_policy='allow'))
        fallback = [labels or [int(prob.argmax())] for labels, prob in zip(raw, probabilities, strict=True)]
        outcome = {key: row[key] for key in ('method', 'alpha', 'randomization_seed')}
        outcome['source_artifact'] = artifact
        for kind, sets in [('raw', raw), ('argmax_fallback', fallback)]:
            outcome[kind] = {}
            for name in POPULATIONS:
                mask = frame['in_'+name].to_numpy(bool)
                selected = [s for s, keep in zip(sets, mask, strict=True) if keep]
                metrics = coverage_metrics(selected, frame.loc[mask, 'dr_grade'].to_numpy(int))
                metrics['source_annotation_sensitivity'] = annotation_envelope(selected, allowed[mask])
                outcome[kind][name] = metrics
        public.append(outcome)
        private.append({**{key: row[key] for key in ('method', 'alpha', 'randomization_seed')},
                        'raw_sets': raw, 'argmax_fallback_sets': fallback})
    return public, private
