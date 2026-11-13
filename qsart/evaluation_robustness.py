"""Research-only perturbations after a single canonical fundus preprocessing pass.

The ordinary inference API always preprocesses uploaded images. These helpers
deliberately score already canonical RGB pixels without recropping them, so a
perturbation cannot silently change the crop geometry. They do not alter API
inference, fit any parameters, or give synthetic images natural quality labels.
"""
from __future__ import annotations

from io import BytesIO
import math

import numpy as np
from PIL import Image, ImageFilter
import torch

from qsart.models.checkpoint import sha256_file
from qsart.preprocessing import preprocess_image

CANONICAL_SIZE = 224
CONDITIONS = ('clean', 'blur_1', 'blur_2', 'exposure_0_5', 'exposure_1_5', 'jpeg_50', 'jpeg_20')
CONDITION_PARAMETERS = {
    'clean': {'operation': 'identity'},
    'blur_1': {'operation': 'PIL.GaussianBlur', 'radius_pixels': 1.0},
    'blur_2': {'operation': 'PIL.GaussianBlur', 'radius_pixels': 2.0},
    'exposure_0_5': {'operation': 'multiply_srgb_uint8', 'factor': .5},
    'exposure_1_5': {'operation': 'multiply_srgb_uint8', 'factor': 1.5},
    'jpeg_50': {'operation': 'JPEG_roundtrip', 'quality': 50, 'subsampling': 2,
                'optimize': False, 'progressive': False},
    'jpeg_20': {'operation': 'JPEG_roundtrip', 'quality': 20, 'subsampling': 2,
                'optimize': False, 'progressive': False},
}


def canonical_rgb(image: Image.Image, size: int = CANONICAL_SIZE) -> Image.Image:
    """Apply the existing crop/pad/resize exactly once to the original image."""
    if not isinstance(image, Image.Image):
        raise ValueError('A decoded PIL source image is required')
    if type(size) is not int or size != CANONICAL_SIZE:
        raise ValueError('This declared stress protocol requires 224px canonical inputs')
    return preprocess_image(image, image_size=size, crop=True).copy()


def _check_canonical(image):
    if (not isinstance(image, Image.Image) or image.mode != 'RGB'
            or image.size != (CANONICAL_SIZE, CANONICAL_SIZE)):
        raise ValueError('Canonical input must be an RGB PIL image of exactly 224x224 pixels')


def apply_degradation(canonical: Image.Image, condition: str) -> Image.Image:
    """Return a new RGB image; fixed transforms never modify their source pixels.

    Exposure is an sRGB-code-value multiplier, not physical exposure simulation.
    It uses float32 arithmetic, clipping, and NumPy round-to-nearest-even before
    uint8 conversion. JPEG results are reproducible within the recorded codec
    environment, which must be bound to experimental output.
    """
    _check_canonical(canonical)
    if condition not in CONDITIONS:
        raise ValueError('Unknown controlled-degradation condition')
    if condition == 'clean':
        return canonical.copy()
    if condition.startswith('blur_'):
        return canonical.filter(ImageFilter.GaussianBlur(CONDITION_PARAMETERS[condition]['radius_pixels']))
    if condition.startswith('exposure_'):
        array = np.asarray(canonical, dtype=np.float32)
        factor = CONDITION_PARAMETERS[condition]['factor']
        changed = np.rint(np.clip(array * factor, 0., 255.)).astype(np.uint8)
        return Image.fromarray(changed)
    buffer = BytesIO()
    parameters = {key: value for key, value in CONDITION_PARAMETERS[condition].items() if key != 'operation'}
    canonical.save(buffer, format='JPEG', **parameters)
    buffer.seek(0)
    with Image.open(buffer) as image:
        image.load()
        return image.convert('RGB').copy()


def _normalization(metadata):
    if metadata.get('image_size') != CANONICAL_SIZE:
        raise ValueError('Both selected models must use the declared 224px input size')
    mean = np.asarray(metadata.get('normalization_mean'), dtype=float)
    std = np.asarray(metadata.get('normalization_std'), dtype=float)
    if (mean.shape != (3,) or std.shape != (3,) or not np.isfinite(mean).all()
            or not np.isfinite(std).all() or (std <= 0).any()):
        raise ValueError('Both models need finite three-channel normalization with positive standard deviations')
    return torch.tensor(mean, dtype=torch.float32)[:, None, None], torch.tensor(std, dtype=torch.float32)[:, None, None]


def score_canonical(engine, images: list[Image.Image]) -> list[dict]:
    """First-pass scores from loaded models, with no crop, resize, TTA or fitting.

    Normalization and temperature application match InferenceEngine's first-pass
    path. Raw softmax values are retained for the existing independent-temperature
    ConformalSidecar.enrich method. This path is a model-input stress diagnostic;
    its outputs and timing must not be described as production upload inference.
    """
    dr_mean, dr_std = _normalization(engine.dr_metadata)
    quality_mean, quality_std = _normalization(engine.quality_metadata)
    if engine.dr_model.training or engine.quality_model.training:
        raise ValueError('Controlled evaluation requires both loaded models in evaluation mode')
    if any(not isinstance(image, Image.Image) for image in images):
        raise ValueError('A sequence of canonical PIL images is required')
    for image in images:
        _check_canonical(image)
    output = []
    with torch.inference_mode():
        for image in images:
            # Do not call image_to_tensor: it would repeat condition-dependent cropping.
            pixels = torch.from_numpy(np.asarray(image, dtype=np.float32).copy()).permute(2, 0, 1) / 255.
            dr_logits = engine.dr_model(((pixels - dr_mean) / dr_std).unsqueeze(0))
            quality_logits = engine.quality_model(((pixels - quality_mean) / quality_std).unsqueeze(0))
            dr = engine._probabilities(dr_logits, 'dr')['dr_grade']
            quality = engine._probabilities(quality_logits, 'quality')
            overall = quality['quality_grade']
            entropy = float(-(dr * dr.clamp_min(1e-12).log()).sum() / math.log(5))
            output.append({
                'model_version': engine.model_version, 'predicted_grade': int(dr.argmax()),
                'grade_probabilities': dr.tolist(), 'referable_probability': min(1., float(dr[2:].sum())),
                'quality': {'probabilities': overall.tolist(), 'predicted_class': int(overall.argmax()),
                            'poor_probability': min(1., float(overall[engine.quality_metadata['poor_class_indices']].sum())),
                            'defects': {head: values.tolist() for head, values in quality.items()
                                        if head != 'quality_grade'}},
                'uncertainty': {'entropy': min(1., max(0., entropy)), 'confidence': float(dr.max()),
                                'method': 'temperature_scaled_normalized_entropy'},
                'second_pass_activated': False, 'data_purpose': engine.dr_metadata['data_purpose'],
                'input_scope': 'canonical_224px_model_input_stress',
                'uncalibrated': {'grade_probabilities': dr_logits['dr_grade'].softmax(1)[0].tolist(),
                                'quality_probabilities': quality_logits['quality_grade'].softmax(1)[0].tolist()},
            })
    return output


def development_selection(frame):
    """Validate a development manifest and byte identity before returning selection.

    The caller must additionally bind the full manifest hash to the frozen model
    and conformal artifacts. No confirmation, holdout, test or unknown split is
    accepted. Returned patient identifiers remain local and must not be exported.
    """
    required = {'split', 'subject_id', 'image_path', 'sha256', 'dr_grade', 'quality_grade'}
    if not required.issubset(frame.columns) or frame.empty:
        raise ValueError('A nonempty verified development manifest with patient and label fields is required')
    if frame[list(required)].isna().any().any():
        raise ValueError('Development grouping, labels, paths and hashes must be complete')
    if not set(frame.split).issubset({'train', 'validation', 'calibration'}):
        raise ValueError('Only train/validation/calibration development partitions are allowed')
    for column in ('subject_id', 'sha256', 'image_path', 'split_group_id'):
        if column in frame and (frame.groupby(column)['split'].nunique() > 1).any():
            raise ValueError(f'Cross-split overlap in {column}')
    if frame.image_path.astype(str).str.casefold().duplicated().any():
        raise ValueError('Development manifest contains duplicate image paths')
    selected = frame.loc[frame.split == 'validation'].reset_index(drop=True).copy()
    if selected.empty:
        raise ValueError('The development selection partition is empty')
    for column, maximum in (('dr_grade', 4), ('quality_grade', 1)):
        values = np.asarray(selected[column], dtype=float)
        if not np.isfinite(values).all() or not np.equal(values, values.astype(int)).all() or (values < 0).any() or (values > maximum).any():
            raise ValueError(f'Invalid original source labels in {column}')
    for row in selected.itertuples():
        if sha256_file(row.image_path) != row.sha256:
            raise ValueError('A selection source image differs from its frozen SHA256')
    return selected


def paired_patient_bootstrap(baseline, conditioned, groups, *, repeats: int = 1000,
                             seed: int = 20261022) -> dict:
    """Image-weighted paired mean delta, resampling whole patients with replacement.

    The same patient's views and both conditions remain together. Intervals are
    exploratory percentile intervals, without correction for multiple metrics or
    conditions. Input arrays must have identical original-image ordering.
    """
    before, after = np.asarray(baseline, dtype=float), np.asarray(conditioned, dtype=float)
    ids = np.asarray(groups)
    if (before.ndim != 1 or after.shape != before.shape or ids.ndim != 1
            or len(ids) != len(before) or not len(before)
            or not np.isfinite(before).all() or not np.isfinite(after).all()):
        raise ValueError('Finite paired vectors and one patient identifier per image are required')
    if (not isinstance(repeats, int) or isinstance(repeats, bool) or repeats < 1
            or not isinstance(seed, int) or isinstance(seed, bool) or seed < 0):
        raise ValueError('Positive integer repeats and a nonnegative integer seed are required')
    if any(value is None or not str(value).strip() or str(value).casefold() in {'nan', 'none'} for value in ids):
        raise ValueError('Complete patient identifiers are required')
    # Subject identifiers from the local manifest share one representation.
    ids = ids.astype(str)
    unique = np.unique(ids)
    indices = {group: np.flatnonzero(ids == group) for group in unique}
    delta = after - before
    result = {'mean_delta': float(delta.mean()), 'n_images': len(delta), 'n_patients': len(unique),
              'repeats': repeats, 'seed': seed, 'direction': 'condition_minus_clean',
              'method': 'paired_patient_cluster_percentile_bootstrap', 'confidence_level': .95,
              'weighting': 'image_weighted_mean_with_whole_patient_resampling',
              'multiplicity_adjusted': False}
    if len(unique) < 2:
        return {**result, 'lower': None, 'upper': None}
    rng = np.random.default_rng(seed)
    values = np.empty(repeats)
    for index in range(repeats):
        sampled = rng.choice(unique, len(unique), replace=True)
        values[index] = delta[np.concatenate([indices[group] for group in sampled])].mean()
    lower, upper = np.quantile(values, [.025, .975])
    return {**result, 'lower': float(lower), 'upper': float(upper)}
