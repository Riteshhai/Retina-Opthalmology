"""Frozen development/confirmation cohorts derived only from verified sources."""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from .manifests import assign_subject_splits

DEFECT_TARGETS = {
    'clarity_class': {'source_column': 'clarity', 'source_codes': [1, 4, 6, 8, 10],
                      'higher_class_means': 'better clarity'},
    'field_definition_class': {'source_column': 'field_definition', 'source_codes': [1, 4, 6, 8, 10],
                               'higher_class_means': 'better field definition'},
    'artifact_class': {'source_column': 'artifact', 'source_codes': [0, 1, 4, 6, 8, 10],
                       'higher_class_means': 'more extensive artifact; class 0 is no artifact'},
}


def add_defect_targets(frame: pd.DataFrame) -> pd.DataFrame:
    """Map the documented source ordinal codes without altering original labels."""
    out = frame.copy()
    for target, definition in DEFECT_TARGETS.items():
        source = definition['source_column']
        if source not in frame:
            raise ValueError(f'Missing verified natural defect column: {source}')
        numbers = pd.to_numeric(frame[source], errors='coerce')
        invalid = frame[source].notna() & (numbers.isna() | ~numbers.isin(definition['source_codes']))
        if invalid.any():
            raise ValueError(f'Unrecognized verified natural defect code in {source}')
        mapping = {code: index for index, code in enumerate(definition['source_codes'])}
        out[target] = numbers.map(mapping).astype('Int64')
    return out


def defect_mapping_report() -> dict:
    return {target: {**definition,
                     'code_to_class': {str(code): index for index, code in enumerate(definition['source_codes'])},
                     'class_to_code': {str(index): code for index, code in enumerate(definition['source_codes'])},
                     'classes': len(definition['source_codes'])}
            for target, definition in DEFECT_TARGETS.items()}


def select_development_and_confirmation(train: pd.DataFrame, official_validation: pd.DataFrame,
                                         pilot: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Exclude whole ambiguous patients and every previously evaluated holdout patient."""
    if not train.official_split.eq('official_train').all():
        raise ValueError('Development must derive exclusively from official training')
    if not official_validation.official_split.eq('official_validation').all():
        raise ValueError('Confirmation must derive from official validation')
    if set(train.subject_id) & set(official_validation.subject_id):
        raise ValueError('Authoritative official source patient overlap')
    excluded = set(train.loc[train.annotation_status.ne('verified'), 'subject_id'])
    development = train.loc[~train.subject_id.isin(excluded)].copy()
    inspected = set(pilot.loc[pilot.split.eq('holdout'), 'subject_id'])
    if not inspected or not inspected.issubset(set(official_validation.subject_id)):
        raise ValueError('Pilot holdout identifiers must be a nonempty subset of official validation')
    confirmation = official_validation.loc[~official_validation.subject_id.isin(inspected)].copy()
    if confirmation.annotation_status.ne('verified').any():
        raise ValueError('Unresolved confirmation annotation rows')
    development['cohort_role'] = 'development_v1'
    confirmation['cohort_role'] = 'new_confirmation_reserved'
    confirmation['split'] = 'holdout'
    return add_defect_targets(development), add_defect_targets(confirmation), {
        'excluded_ambiguous_subjects': sorted(excluded), 'excluded_ambiguous_subject_count': len(excluded),
        'excluded_training_images': int(train.subject_id.isin(excluded).sum()),
        'previously_evaluated_pilot_holdout_subjects': sorted(inspected),
        'previously_evaluated_pilot_holdout_subject_count': len(inspected),
        'development_subject_count': development.subject_id.nunique(),
        'confirmation_subject_count': confirmation.subject_id.nunique(),
        'confirmation_rule': 'all official validation patients minus every pilot holdout patient; no prediction-based selection',
    }


def duplicate_safe_splits(frame: pd.DataFrame, seed=2026, distance=2) -> tuple[pd.DataFrame, dict]:
    """Conservatively keep cross-subject fingerprint candidates in one partition.

    These connected components are split groups, not invented patient identities.
    Exact source patient IDs remain unchanged. No outcome predictions are used.
    """
    if frame.empty or not frame.official_split.eq('official_train').all():
        raise ValueError('Nonempty official-training images required for development splitting')
    subjects = sorted(frame.subject_id.unique())
    parent = {subject: subject for subject in subjects}
    def find(subject):
        while parent[subject] != subject:
            parent[subject] = parent[parent[subject]]
            subject = parent[subject]
        return subject
    def union(first, second):
        a, b = sorted([find(first), find(second)])
        parent[b] = a
    records = frame[['image_id', 'subject_id', 'sha256', 'phash']].to_dict('records')
    candidates = []
    for i, first in enumerate(records):
        if pd.isna(first['phash']) or pd.isna(first['sha256']):
            raise ValueError('All images require actual SHA256/pHash before frozen split assignment')
        for second in records[i + 1:]:
            if first['subject_id'] == second['subject_id']:
                continue
            hamming = (int(first['phash'], 16) ^ int(second['phash'], 16)).bit_count()
            exact = first['sha256'] == second['sha256']
            if exact or hamming <= distance:
                union(first['subject_id'], second['subject_id'])
                candidates.append({'first_image_id': first['image_id'], 'second_image_id': second['image_id'],
                                   'hamming_distance': hamming, 'byte_exact': exact})
    components = {}
    for subject in subjects:
        components.setdefault(find(subject), []).append(subject)
    group_map = {}
    for group in components.values():
        key = group[0] if len(group) == 1 else 'splitgroup_' + hashlib.sha256('|'.join(group).encode()).hexdigest()[:16]
        group_map.update({subject: key for subject in group})
    result = frame.copy()
    result['split_group_id'] = result.subject_id.map(group_map)
    grouped = result.copy()
    grouped['subject_id'] = grouped.split_group_id
    result['split'] = assign_subject_splits(grouped, seed, .15, .15)
    counts = result.groupby('split').subject_id.nunique()
    return result, {'seed': seed, 'requested_subject_proportions': {'train': .70, 'validation': .15, 'calibration': .15},
                    'stratification': 'maximum verified eye grade per conservative subject/fingerprint split group',
                    'rounding': 'floor within severity strata; minimum one validation/calibration group when stratum >=3',
                    'cross_subject_perceptual_candidates': candidates,
                    'merged_subject_components': [group for group in components.values() if len(group) > 1],
                    'actual_subject_counts': {str(key): int(value) for key, value in counts.items()},
                    'actual_subject_proportions': {str(key): float(value / len(subjects)) for key, value in counts.items()}}


def check_confirmation_independence(confirmation: pd.DataFrame, development: pd.DataFrame,
                                     inspected_subjects: set[str]) -> None:
    if set(confirmation.subject_id) & (set(development.subject_id) | inspected_subjects):
        raise ValueError('Confirmation overlaps development or previously evaluated pilot patients')
    if not confirmation.split.eq('holdout').all():
        raise ValueError('Confirmation must remain holdout only')
    if not confirmation.official_split.eq('official_validation').all():
        raise ValueError('Confirmation contains non-validation source records')
    if confirmation.empty or np.any(confirmation.annotation_status.ne('verified')):
        raise ValueError('Confirmation must contain nonempty verified source annotations')
