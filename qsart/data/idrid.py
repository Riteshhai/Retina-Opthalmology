"""Strict disease-grading ingestion for reserved IDRiD external evaluation.

No patient identifiers are inferred, and DME labels remain distinct from DR.
Pairwise candidates describe observable fingerprints, never clinical identity.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
from PIL import Image

from .aptos import fingerprint_components
from .manifests import inspect_image, sha256_file

IDRID_SOURCE = 'https://idrid.grand-challenge.org/Data/'
PARTITIONS = {
    'official_train': ('a. Training Set', 'a. IDRiD_Disease Grading_Training Labels.csv', 413),
    'official_test': ('b. Testing Set', 'b. IDRiD_Disease Grading_Testing Labels.csv', 103),
}
PATIENT_LIMITATION = (
    'Serial image names provide no authoritative patient or eye identities. '
    'Fingerprint components are not patients; unknown related views and undetected overlap remain possible.'
)


def parse_idrid_labels(dataset_root: str | Path, *, require_release_counts: bool = False) -> pd.DataFrame:
    """Accept the disease-grading folder, preserving both original source splits."""
    root = Path(dataset_root).expanduser().resolve()
    required = {'Image name', 'Retinopathy grade', 'Risk of macular edema'}
    frames = []
    for split, (directory, filename, expected_count) in PARTITIONS.items():
        annotations = root/'2. Groundtruths'/filename
        if not annotations.is_file() or not annotations.resolve().is_relative_to(root):
            raise ValueError('Missing IDRiD disease-grading annotations or source path outside root')
        raw = pd.read_csv(annotations, dtype=str, keep_default_na=False)
        raw.columns = raw.columns.str.strip()
        if raw.columns.duplicated().any():
            raise ValueError('Duplicate IDRiD annotation columns')
        extras = set(raw) - required
        if (not required.issubset(raw) or raw.empty or any(
                not name.startswith('Unnamed:') or not raw[name].str.strip().eq('').all() for name in extras)):
            raise ValueError('IDRiD requires the three grading fields; only blank Unnamed columns are allowed')
        if require_release_counts and len(raw) != expected_count:
            raise ValueError('Incomplete IDRiD release: expected 413 training and 103 testing rows')
        identities = raw['Image name']
        if not identities.str.fullmatch(r'IDRiD_[0-9]+').all():
            raise ValueError('IDRiD image stems must be safe unmodified official serial names')
        if identities.str.casefold().duplicated().any():
            raise ValueError('Duplicate IDRiD image stems within original partition')
        labels = {}
        for name, grades in [('Retinopathy grade', range(5)), ('Risk of macular edema', range(3))]:
            if not raw[name].str.fullmatch(r'[0-9]').all():
                raise ValueError('IDRiD DR and DME fields require integer source labels')
            labels[name] = pd.to_numeric(raw[name])
            if not labels[name].isin(grades).all():
                raise ValueError('IDRiD DR must be 0-4 and DME risk must be 0-2')
        image_root = root/'1. Original Images'/directory
        if not image_root.is_dir() or not image_root.resolve().is_relative_to(root):
            raise ValueError('Missing IDRiD image directory or source path outside root')
        paths = [(image_root/f'{stem}.jpg').resolve() for stem in identities]
        if any(not path.is_relative_to(root) for path in paths):
            raise ValueError('IDRiD image paths must stay inside the provided root')
        if len(set(paths)) != len(paths):
            raise ValueError('Duplicate resolved IDRiD image paths')
        supplied = {path.resolve() for path in image_root.iterdir() if path.is_file()}
        if supplied != set(paths):
            raise ValueError('IDRiD annotation/image inventory mismatch: missing or extra image files')
        frames.append(pd.DataFrame({
            'dataset_source': 'IDRiD', 'official_split': split,
            'image_id': split+':'+identities, 'original_image_id': identities,
            'original_retinopathy_grade': raw['Retinopathy grade'],
            'original_dme_risk': raw['Risk of macular edema'],
            'dr_grade': labels['Retinopathy grade'].astype(int),
            'dme_risk': labels['Risk of macular edema'].astype(int),
            'image_path': [str(path) for path in paths],
            'annotation_status': 'verified', 'source_annotation_sha256': sha256_file(annotations),
            'split': 'external_holdout', 'eligible_for_training': False,
            'evaluation_status': 'reserved_not_inferred',
        }))
    return pd.concat(frames, ignore_index=True)


def validate_fingerprints(frame: pd.DataFrame) -> None:
    if frame.empty or not {'dataset_source', 'image_id', 'sha256', 'phash'}.issubset(frame):
        raise ValueError('Nonempty reference identities and fingerprints required')
    for field, pattern in [('sha256', r'[0-9a-f]{64}'), ('phash', r'[0-9a-f]{16}')]:
        if frame[field].isna().any() or not frame[field].astype(str).str.fullmatch(pattern).all():
            raise ValueError('Complete valid SHA256 and pHash reference fingerprints required')
    for field in ('dataset_source', 'image_id'):
        if frame[field].isna().any() or frame[field].astype(str).str.strip().eq('').any():
            raise ValueError('Nonempty reference identities and dataset names required')
    grouped = frame.groupby(['dataset_source', 'image_id'])[['sha256', 'phash']].nunique()
    if grouped.gt(1).any().any():
        raise ValueError('Conflicting fingerprints for the same reference identity')


def cross_dataset_candidates(external: pd.DataFrame, references: pd.DataFrame, *, distance: int = 2) -> pd.DataFrame:
    if type(distance) is not int or not 0 <= distance <= 63:
        raise ValueError('Integer pHash distance in [0,63] required')
    validate_fingerprints(external)
    validate_fingerprints(references)
    if external.image_id.duplicated().any():
        raise ValueError('Unique external identities required')
    rows = []
    references = references.drop_duplicates(['dataset_source', 'image_id'])
    ref = [(row.dataset_source, row.image_id, row.sha256, int(row.phash, 16))
           for row in references.itertuples()]
    for first in external.itertuples():
        fingerprint = int(first.phash, 16)
        for dataset, identity, digest, phash in ref:
            hamming = (fingerprint ^ phash).bit_count()
            exact = first.sha256 == digest
            if exact or hamming <= distance:
                rows.append({'external_image_id': first.image_id, 'reference_dataset': dataset,
                             'reference_image_id': identity, 'byte_exact': exact, 'phash_distance': hamming})
    return pd.DataFrame(rows, columns=['external_image_id', 'reference_dataset', 'reference_image_id',
                                       'byte_exact', 'phash_distance'])


def audit_idrid(dataset_root: str | Path, references: pd.DataFrame, *, require_release_counts: bool = True,
                distance: int = 2) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    root = Path(dataset_root).resolve()
    license_paths = [root/'LICENSE.txt', root/'CC-BY-4.0.txt']
    if any(not path.is_file() or not path.resolve().is_relative_to(root) for path in license_paths):
        raise ValueError('Included IDRiD attribution and CC BY 4.0 license files required')
    license_text = license_paths[0].read_text(encoding='utf-8')
    if not all(value in license_text for value in ('Porwal', 'Pachade', 'Kokare', 'Attribution 4.0')):
        raise ValueError('Unrecognized IDRiD attribution/license notice')
    frame = parse_idrid_labels(root, require_release_counts=require_release_counts)
    inspections = [inspect_image(Path(path)) for path in frame.image_path]
    if not all(row['readable'] and row['image_status'] == 'ok' for row in inspections):
        raise ValueError('IDRiD contains unreadable or corrupt images; manifest not frozen')
    for path in frame.image_path:
        with Image.open(path) as image:
            if image.format != 'JPEG':
                raise ValueError('IDRiD grading source must contain actual JPEG images')
    for field in ('exists', 'readable', 'image_status', 'width', 'height', 'sha256', 'phash'):
        frame[field] = [row[field] for row in inspections]
    frame, internal = fingerprint_components(frame, near_duplicate_distance=distance)
    frame['split_group_id'] = frame.split_group_id.str.replace('aptos_splitgroup_', 'idrid_component_', regex=False)
    cross = cross_dataset_candidates(frame, references, distance=distance)
    report = {
        'schema_version': 1, 'experiment_id': 'idrid_data_v1', 'dataset': 'IDRiD',
        'scope': 'external_data_integrity_and_overlap_readiness_only', 'source_url': IDRID_SOURCE,
        'license': 'CC BY 4.0', 'license_file_sha256': {p.name: sha256_file(p) for p in license_paths},
        'images': len(frame), 'images_decoded': len(inspections),
        'official_partitions': {split: {
            'images': len(part), 'annotation_sha256': str(part.source_annotation_sha256.iloc[0]),
            'dr_grade_counts': {str(k): int(v) for k, v in part.dr_grade.value_counts().sort_index().items()},
            'dme_risk_counts': {str(k): int(v) for k, v in part.dme_risk.value_counts().sort_index().items()},
        } for split, part in frame.groupby('official_split')},
        'dimensions': [{'width': int(w), 'height': int(h), 'images': int(n)}
                       for (w, h), n in frame.groupby(['width', 'height']).size().items()],
        'fingerprint_method': 'central_green_dct_phash_v1', 'near_duplicate_distance': distance,
        'internal_components': int(frame.split_group_id.nunique()),
        'internal_duplicate_components': int((frame.groupby('split_group_id').size() > 1).sum()),
        'internal_candidate_pairs': len(internal), 'internal_exact_pairs': int(internal.byte_exact.sum()),
        'internal_label_conflict_components': int(frame.loc[frame.component_label_conflict, 'split_group_id'].nunique()),
        'cross_source_candidate_pairs': len(cross), 'cross_source_exact_pairs': int(cross.byte_exact.sum()),
        'cross_source_candidate_images': int(cross.external_image_id.nunique()),
        'reference_datasets': {str(dataset): {
            'rows': len(part), 'unique_identities': int(part.image_id.nunique()),
            'candidate_pairs': int(cross.reference_dataset.eq(dataset).sum()),
            'exact_pairs': int((cross.reference_dataset.eq(dataset) & cross.byte_exact).sum()),
        } for dataset, part in references.groupby('dataset_source')},
        'independence_claim_blocked_by_detected_overlap': bool(len(cross)),
        'observations_excluded': 0, 'dr_labels_modified': False, 'dme_used_as_quality': False,
        'model_inference_performed': False, 'model_fitting_performed': False,
        'calibration_performed': False, 'external_model_evaluation_completed': False,
        'patient_identity_available': False, 'quality_labels_available': False,
        'patient_grouping_limitation': PATIENT_LIMITATION,
        'limitations': [PATIENT_LIMITATION,
            'pHash matches are review candidates; no candidate is automatically excluded or relabeled.',
            'Reference fingerprints were previously verified; reference images are not freshly reinspected here.',
            'Label distributions are inspected for data auditing, not for model or threshold selection.',
            'Included license/layout do not authenticate download origin or establish absence of every related view.',
            'The complete release differs from the base-paper 455-image reported cohort.'],
    }
    return frame, internal, cross, report


def content_digest(frame: pd.DataFrame) -> str:
    values = sorted(f'{row.image_id}:{row.sha256}' for row in frame.itertuples())
    return hashlib.sha256('\n'.join(values).encode()).hexdigest()
