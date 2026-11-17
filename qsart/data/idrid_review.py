"""Data-only pair evidence and label-independent IDRiD sensitivity populations."""
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

from qsart.data.manifests import sha256_file
from qsart.preprocessing import preprocess_image


def compare_images(first_path, second_path, first_sha, second_sha):
    if sha256_file(first_path) != first_sha or sha256_file(second_path) != second_sha:
        raise ValueError('Pair source image bytes changed since the frozen audit')
    originals, canonical = [], []
    for path in (first_path, second_path):
        with Image.open(Path(path)) as handle:
            rgb = ImageOps.exif_transpose(handle).convert('RGB')
            originals.append(np.asarray(rgb).copy())
            canonical.append(np.asarray(preprocess_image(rgb, image_size=224, crop=True)).copy())
    original_equal = np.array_equal(*originals)
    first, second = [values.astype(np.float64) for values in canonical]
    delta = first-second
    correlation = (float(np.corrcoef(first.ravel(), second.ravel())[0, 1])
                   if first.std() > 0 and second.std() > 0 else None)
    return {'byte_exact': first_sha == second_sha, 'decoded_rgb_exact': original_equal,
        'original_dimensions_match': originals[0].shape == originals[1].shape,
        'evidence_category': ('byte_exact' if first_sha == second_sha else
                              'decoded_rgb_exact_different_bytes' if original_equal else 'fingerprint_candidate_unresolved'),
        'canonical_224_rgb_exact': np.array_equal(*canonical),
        'canonical_224_mae_0_255': float(np.abs(delta).mean()),
        'canonical_224_rmse_0_255': float(np.sqrt(np.square(delta).mean())),
        'canonical_224_pearson_r': correlation}


def freeze_populations(frame: pd.DataFrame, candidates: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    required = {'image_id', 'split_group_id', 'dr_grade', 'component_label_conflict',
                'split', 'eligible_for_training'}
    if (frame.empty or not required.issubset(frame) or frame.image_id.duplicated().any() or
            frame[list(required)].isna().any().any() or not frame.split.eq('external_holdout').all() or
            not frame.eligible_for_training.eq(False).all()):
        raise ValueError('Verified reserved external-only component records required')
    if not frame.dr_grade.isin(range(5)).all() or not frame.component_label_conflict.isin([True, False]).all():
        raise ValueError('Valid unchanged source labels and conflict flags required')
    if not {'external_image_id', 'reference_dataset'}.issubset(candidates):
        raise ValueError('Bound candidate identities and reference datasets required')
    if not set(candidates.external_image_id).issubset(set(frame.image_id)):
        raise ValueError('Cross-source candidate is outside the frozen external cohort')
    if not candidates.reference_dataset.eq('APTOS_2019').all():
        raise ValueError('This declared review expects the frozen APTOS candidate ledger')
    result = frame.copy()
    blocked_components = set(result.loc[result.image_id.isin(candidates.external_image_id), 'split_group_id'])
    result['in_full_source'] = True
    result['in_cross_source_filtered'] = ~result.split_group_id.isin(blocked_components)
    representatives = set()
    retained = result.loc[result.in_cross_source_filtered]
    for _, part in retained.groupby('split_group_id'):
        identities = part.image_id.astype(str).tolist()
        representatives.add(min(identities, key=lambda identity: (
            hashlib.sha256(('idrid-review-v1:'+identity).encode()).hexdigest(), identity)))
    result['in_cross_source_filtered_representatives'] = result.image_id.isin(representatives)
    summaries = {}
    for name in ('full_source', 'cross_source_filtered', 'cross_source_filtered_representatives'):
        part = result.loc[result['in_'+name]]
        summaries[name] = {'images': len(part), 'fingerprint_components': int(part.split_group_id.nunique()),
            'dr_grade_counts': {str(k): int(v) for k, v in part.dr_grade.value_counts().sort_index().items()},
            'label_conflicting_components': int(part.loc[part.component_label_conflict, 'split_group_id'].nunique())}
    return result, {'populations': summaries, 'cross_source_flagged_components': len(blocked_components),
        'cross_source_flagged_observations': int((~result.in_cross_source_filtered).sum()),
        'representative_rule': 'Minimum SHA256(idrid-review-v1: + namespaced identity), independent of labels/order/outcomes',
        'clinical_label_adjudication_performed': False, 'source_labels_modified': False,
        'full_source_observations_deleted': 0, 'model_inference_performed': False}
