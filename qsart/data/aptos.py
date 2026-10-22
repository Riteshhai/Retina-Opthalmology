"""Strict local ingestion of the official APTOS labeled-training export.

This module never downloads data or assumes that competition test labels,
patient IDs, eye IDs or image-quality annotations are available. Fingerprint
components are split groups, never surrogate patient identities.
"""
from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import math
from pathlib import Path
import re

import pandas as pd

from .manifests import inspect_image, sha256_file

APTOS_SOURCE = 'https://www.kaggle.com/competitions/aptos2019-blindness-detection/data'
DEFAULT_FRACTIONS = {'train': .60, 'validation': .15, 'calibration': .15, 'test': .10}
PATIENT_LIMITATION = (
    'Official id_code/diagnosis annotations provide no authoritative patient or eye identifiers. '
    'SHA256 and central-green pHash connected components reduce detectable image overlap; '
    'they cannot establish patient-disjoint partitions or exclude every same-eye view.'
)


def _fractions(values: Mapping[str, float] | None) -> dict[str, float]:
    values = dict(DEFAULT_FRACTIONS if values is None else values)
    if set(values) != set(DEFAULT_FRACTIONS):
        raise ValueError('APTOS split fractions require train, validation, calibration and test')
    if not all(isinstance(value, (int, float)) and math.isfinite(value) and 0 < value < 1
               for value in values.values()) or not math.isclose(sum(values.values()), 1, abs_tol=1e-10):
        raise ValueError('APTOS split fractions must be finite, positive and sum to one')
    return {key: float(values[key]) for key in DEFAULT_FRACTIONS}


def parse_aptos_labels(dataset_root: str | Path) -> pd.DataFrame:
    """Accept only train.csv with id_code/diagnosis, using train_images/<id>.png.

    The caller must obtain the official competition export under its terms.
    No filename guessing, image mirrors or public test labels are used.
    """
    root = Path(dataset_root).expanduser().resolve()
    annotations = root / 'train.csv'
    if not root.is_dir() or not annotations.is_file():
        raise ValueError('APTOS root must contain the official labeled train.csv')
    raw = pd.read_csv(annotations, dtype={'id_code': 'string', 'diagnosis': 'string'}, keep_default_na=False)
    if set(raw.columns) != {'id_code', 'diagnosis'} or raw.empty:
        raise ValueError('APTOS train.csv must contain only nonempty id_code and diagnosis columns')
    identifiers = raw.id_code.astype(str)
    if not identifiers.map(lambda value: bool(re.fullmatch(r'[A-Za-z0-9_-]+', value))).all():
        raise ValueError('APTOS id_code values must be nonempty safe image stems')
    if identifiers.str.casefold().duplicated().any():
        raise ValueError('Duplicate APTOS id_code values')
    labels = pd.to_numeric(raw.diagnosis, errors='coerce')
    if labels.isna().any() or not labels.isin(range(5)).all():
        raise ValueError('APTOS diagnosis must contain integer grades 0 through 4')
    image_root = root / 'train_images'
    paths = [(image_root / f'{value}.png').resolve() for value in identifiers]
    if any(not path.is_relative_to(root) for path in paths):
        raise ValueError('APTOS image paths must remain within the caller-provided root')
    return pd.DataFrame({
        'dataset_source': 'APTOS_2019', 'official_split': 'official_train',
        'image_id': identifiers, 'original_id_code': identifiers,
        'original_diagnosis': raw.diagnosis, 'dr_grade': labels.astype(int),
        'image_path': [str(path) for path in paths],
        'annotation_status': 'verified', 'source_annotation_sha256': sha256_file(annotations),
    })


def assign_aptos_splits(frame: pd.DataFrame, *, seed: int = 20261020,
                       fractions: Mapping[str, float] | None = None,
                       near_duplicate_distance: int = 2) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Allocate duplicate-connected components without claiming patient identity.

    Conflicting grade labels anywhere in a component block allocation; they need
    documented source review rather than arbitrary majority voting or relabeling.
    Fractions apply to groups within each grade stratum. Integer rounding means
    actual image proportions can differ, particularly for large components.
    """
    proportions = _fractions(fractions)
    if not isinstance(seed, int) or not 0 <= near_duplicate_distance <= 63:
        raise ValueError('Integer split seed and pHash distance in [0, 63] required')
    required = {'image_id', 'dr_grade', 'sha256', 'phash'}
    if frame.empty or not required.issubset(frame.columns):
        raise ValueError('Nonempty APTOS records with actual labels and fingerprints required')
    if frame.image_id.isna().any() or frame.image_id.astype(str).str.casefold().duplicated().any():
        raise ValueError('APTOS split allocation requires unique image IDs')
    if not pd.to_numeric(frame.dr_grade, errors='coerce').isin(range(5)).all():
        raise ValueError('APTOS split allocation requires verified grade labels')
    for field, pattern in [('sha256', r'[0-9a-f]{64}'), ('phash', r'[0-9a-f]{16}')]:
        if frame[field].isna().any() or not frame[field].astype(str).str.fullmatch(pattern).all():
            raise ValueError('Actual SHA256 and central-green pHash fingerprints required')
    records = frame.reset_index(drop=True).to_dict('records')
    parent = list(range(len(records)))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    candidates = []
    fingerprints = [int(row['phash'], 16) for row in records]
    for index, first in enumerate(records):
        for other in range(index + 1, len(records)):
            second = records[other]
            distance = (fingerprints[index] ^ fingerprints[other]).bit_count()
            exact = first['sha256'] == second['sha256']
            if exact or distance <= near_duplicate_distance:
                first_root, second_root = sorted((find(index), find(other)))
                parent[second_root] = first_root
                candidates.append({'first_image_id': first['image_id'], 'second_image_id': second['image_id'],
                                   'byte_exact': exact, 'phash_distance': distance,
                                   'label_conflict': int(first['dr_grade']) != int(second['dr_grade'])})
    components = {}
    for index in range(len(records)):
        components.setdefault(find(index), []).append(index)
    groups = []
    for indices in components.values():
        if len({int(records[index]['dr_grade']) for index in indices}) != 1:
            raise ValueError('Conflicting DR labels in an APTOS exact/perceptual duplicate component; '
                             'resolve source annotations before splitting')
        identities = sorted(str(records[index]['image_id']) for index in indices)
        key = 'aptos_splitgroup_' + hashlib.sha256('|'.join(identities).encode()).hexdigest()[:24]
        groups.append({'group': key, 'grade': int(records[indices[0]]['dr_grade']), 'indices': indices})
    assignments = {}
    for grade in range(5):
        stratum = [group for group in groups if group['grade'] == grade]
        stratum.sort(key=lambda group: hashlib.sha256(f'{seed}:{group["group"]}'.encode()).hexdigest())
        counts = {role: int(len(stratum) * proportions[role]) for role in ('validation', 'calibration', 'test')}
        if len(stratum) >= 4:
            counts = {role: max(1, count) for role, count in counts.items()}
        # Retain a training group under every allowed fraction/stratum combination.
        while sum(counts.values()) >= len(stratum) and stratum:
            role = max(counts, key=lambda item: (counts[item], item))
            counts[role] -= 1
        position = 0
        for role in ('test', 'calibration', 'validation', 'train'):
            limit = counts.get(role, len(stratum) - position)
            for group in stratum[position:position + limit]:
                assignments[group['group']] = role
            position += limit
    result = frame.copy().reset_index(drop=True)
    for group in groups:
        result.loc[group['indices'], 'split_group_id'] = group['group']
    result['split'] = result.split_group_id.map(assignments)
    report = {
        'seed': seed, 'requested_group_proportions': proportions,
        'grouping': 'SHA256 or central_green_dct_phash_v1 distance connected components',
        'near_duplicate_distance': near_duplicate_distance,
        'stratification': 'verified DR grade per label-consistent fingerprint component',
        'rounding': 'floor per grade; minimum one held-out group per role when >=4 groups; retain training group',
        'patient_identifiers_available': False, 'patient_disjointness_established': False,
        'patient_grouping_limitation': PATIENT_LIMITATION,
        'images': len(result), 'split_groups': len(groups),
        'duplicate_components': sum(len(group['indices']) > 1 for group in groups),
        'duplicate_candidate_pairs': len(candidates),
        'exact_candidate_pairs': sum(pair['byte_exact'] for pair in candidates),
        'conflicting_components': 0,
        'partitions': {role: {
            'images': int((result.split == role).sum()),
            'split_groups': int(result.loc[result.split == role, 'split_group_id'].nunique()),
            'dr_grade_counts': {str(grade): int(count) for grade, count in
                                result.loc[result.split == role, 'dr_grade'].value_counts().sort_index().items()},
        } for role in DEFAULT_FRACTIONS},
    }
    columns = ['first_image_id', 'second_image_id', 'byte_exact', 'phash_distance', 'label_conflict']
    return result, pd.DataFrame(candidates, columns=columns), report


def build_aptos_manifest(dataset_root: str | Path, *, seed: int = 20261020,
                         fractions: Mapping[str, float] | None = None,
                         near_duplicate_distance: int = 2) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    frame = parse_aptos_labels(dataset_root)
    inspections = [inspect_image(Path(path)) for path in frame.image_path]
    if not all(row['readable'] and row['image_status'] == 'ok' for row in inspections):
        raise ValueError('APTOS contains missing, corrupt or unsupported labeled images; no split was created')
    for field in ('exists', 'readable', 'image_status', 'width', 'height', 'sha256', 'phash'):
        frame[field] = [row[field] for row in inspections]
    frame['eligible_for_training'] = True
    result, duplicates, split_report = assign_aptos_splits(
        frame, seed=seed, fractions=fractions, near_duplicate_distance=near_duplicate_distance)
    split_report.update(source_url=APTOS_SOURCE, source_annotation_sha256=str(frame.source_annotation_sha256.iloc[0]),
                        dataset_source='APTOS_2019',
                        supervision={'dr_grade': 'official diagnosis 0-4', 'quality': 'unavailable',
                                     'patient_id': 'unavailable', 'eye_id': 'unavailable'},
                        test_definition='internal reserved split of official labeled training export; '
                                        'not inaccessible Kaggle competition test labels')
    return result, duplicates, split_report


def write_aptos_outputs(frame: pd.DataFrame, duplicates: pd.DataFrame, report: dict,
                        output: str | Path) -> None:
    """Write private per-image outputs; only aggregate reports are shareable."""
    output = Path(output)
    names = ['dataset_manifest.csv', 'duplicates_report.csv', 'split_report.json', 'dataset_summary.json',
             'data_validation_report.md', 'class_distribution.png']
    if any((output / name).exists() for name in names):
        raise ValueError('APTOS outputs already exist; preserve the frozen manifest and choose a new output directory')
    if report.get('images') != len(frame) or frame.empty:
        raise ValueError('APTOS report and manifest do not match')
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / 'dataset_manifest.csv'
    frame.to_csv(manifest, index=False)
    duplicates.to_csv(output / 'duplicates_report.csv', index=False)
    summary = {**report, 'manifest_sha256': sha256_file(manifest),
               'quality_distribution': 'not generated: no official quality labels available'}
    for name in ('split_report.json', 'dataset_summary.json'):
        (output / name).write_text(json.dumps(summary, indent=2), encoding='utf-8')
    (output / 'data_validation_report.md').write_text(
        '# APTOS local source validation\n\n'
        f'Verified readable labeled images: {len(frame)}. Labels derive only from official train.csv.\n\n'
        f'{PATIENT_LIMITATION}\n\n'
        'Exact/perceptual fingerprint components remain in one split. Conflicting labels block ingestion. '
        'Quality labels and competition test labels are unavailable. The internal test partition is reserved.\n\n'
        'Per-image manifests and duplicate identifiers are local private artifacts. '
        'Temperature and conformal calibration require distinct subsets before either fitting step.\n', encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    figure, axis = plt.subplots(figsize=(7, 4))
    counts = frame.dr_grade.value_counts().reindex(range(5), fill_value=0)
    axis.bar([str(grade) for grade in range(5)], counts.to_numpy())
    axis.set(xlabel='Official diagnosis / DR grade', ylabel='Images', title='APTOS labeled training export')
    figure.tight_layout()
    figure.savefig(output / 'class_distribution.png', dpi=150)
    plt.close(figure)
