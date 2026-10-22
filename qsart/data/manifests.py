"""Traceable DeepDRiD image manifests; no images or labels are invented."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PureWindowsPath

import numpy as np
import pandas as pd
from PIL import Image

from qsart.deepdrid import EXPECTED, normalize_frame
from qsart.preprocessing import crop_fundus

QUALITY_CODES = {'clarity': {1, 4, 6, 8, 10}, 'field_definition': {1, 4, 6, 8, 10},
                 'artifact': {0, 1, 4, 6, 8, 10}}
ALLOWED_SPLITS = {'train', 'calibration', 'validation', 'holdout'}
_COORDINATES = np.arange(32)
_DCT_BASIS = np.cos(np.pi * (2 * _COORDINATES + 1) * _COORDINATES[:, None] / 64)


def retinal_phash(image: Image.Image) -> str:
    """63-bit green-channel DCT fingerprint with border suppression.

    Generic whole-image dHash is dominated by circular dark fundus borders.
    Crop the non-dark box and central 75% before DCT; remove the DC component.
    This catches perceptual candidates and does not prove patient identity.
    """
    crop = crop_fundus(image)
    center = crop.crop((crop.width // 8, crop.height // 8, crop.width * 7 // 8, crop.height * 7 // 8))
    pixels = np.asarray(center.getchannel('G').resize((32, 32), Image.Resampling.BICUBIC), dtype=float)
    coefficients = (_DCT_BASIS @ pixels @ _DCT_BASIS.T)[:8, :8].ravel()[1:]
    bits = coefficients > np.median(coefficients)
    return f'{int("".join("1" if bit else "0" for bit in bits), 2):016x}'


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_order(value: str, seed: int) -> str:
    return hashlib.sha256(f'{seed}:{value}'.encode()).hexdigest()


def assign_subject_splits(frame: pd.DataFrame, seed: int = 2026,
                          calibration_fraction: float = .15,
                          validation_fraction: float = .15) -> pd.Series:
    """Stratify by maximum verified eye grade, keeping every patient together.

    Only official training labels enter this allocator. Official validation is
    independently mapped to holdout and never enters selection/calibration.
    """
    if not 0 < calibration_fraction < 1 or not 0 < validation_fraction < 1:
        raise ValueError('calibration/validation fractions must be in (0, 1)')
    if calibration_fraction + validation_fraction >= 1:
        raise ValueError('calibration + validation fractions must be < 1')
    by_subject = frame.groupby('subject_id').dr_grade.max().fillna(-1)
    mapping = {}
    for _, stratum in by_subject.groupby(by_subject):
        subjects = sorted(stratum.index, key=lambda subject: _hash_order(subject, seed))
        n = len(subjects)
        nc = int(n * calibration_fraction)
        nv = int(n * validation_fraction)
        if n >= 3:
            nc, nv = max(nc, 1), max(nv, 1)
        for position, subject in enumerate(subjects):
            mapping[subject] = ('calibration' if position < nc else
                                'validation' if position < nc + nv else 'train')
    return frame.subject_id.map(mapping)


def parse_deepdrid(csv_path: str | Path, official_split: str) -> pd.DataFrame:
    """Reuse starter parser for conventional rows; quarantine ambiguous rows."""
    csv_path = Path(csv_path)
    raw = pd.read_csv(csv_path)
    missing = (EXPECTED | {'image_path'}) - set(raw.columns)
    if missing:
        raise ValueError(f'{csv_path}: missing official columns {sorted(missing)}')
    if raw.patient_id.isna().any() or raw.image_id.isna().any() or raw.image_path.isna().any():
        raise ValueError('Missing authoritative identifiers or original image paths')
    if raw.image_id.duplicated().any():
        raise ValueError('Duplicate image_id in official annotations')
    annotation_paths = raw.image_path.astype(str).map(lambda value: PureWindowsPath(value.strip()))
    if not annotation_paths.map(lambda path: path.stem).eq(raw.image_id.astype(str)).all():
        raise ValueError('Original image_path filename conflicts with image_id')
    if not annotation_paths.map(lambda path: path.parent.name).eq(raw.patient_id.astype(str)).all():
        raise ValueError('Original image_path subject folder conflicts with patient_id')
    side = raw.image_id.astype(str).str.extract(r'^\d+_([lr])[12]$', expand=False)
    eye_label = pd.to_numeric(raw.left_eye_DR_Level, errors='coerce').where(
        side.eq('l'), pd.to_numeric(raw.right_eye_DR_Level, errors='coerce'))
    prefix = raw.image_id.astype(str).str.extract(r'^(\d+)_', expand=False)
    if not prefix.eq(raw.patient_id.astype(str)).all():
        raise ValueError('image_id patient prefix conflicts with authoritative patient_id')
    conflict = side.isna() | eye_label.isna() | ~eye_label.isin(range(5))
    clean = normalize_frame(raw.loc[~conflict].reset_index(drop=True), official_split)
    out = pd.DataFrame(index=raw.index)
    out['dataset_source'] = 'DeepDRiD_regular_fundus'
    out['official_split'] = official_split
    out['subject_id_original'] = raw.patient_id.astype(str)
    out['subject_id'] = 'deepdrid_' + out.subject_id_original
    out['image_id'] = raw.image_id.astype(str)
    out['eye'] = pd.Series(pd.NA, index=raw.index, dtype='string')
    out['dr_grade'] = pd.Series(pd.NA, index=raw.index, dtype='Int64')
    if len(clean):
        indexed = clean.set_index('image_id')
        out.loc[~conflict, 'eye'] = out.loc[~conflict, 'image_id'].map(indexed.eye)
        out.loc[~conflict, 'dr_grade'] = out.loc[~conflict, 'image_id'].map(indexed.grade)
    quality = pd.to_numeric(raw['Overall quality'], errors='coerce')
    if quality.isna().any() or not quality.isin([0, 1]).all():
        raise ValueError('Unrecognized Overall quality coding')
    out['quality_grade'] = quality.astype(int)
    for target, source in [('clarity', 'Clarity'), ('field_definition', 'Field definition'),
                           ('artifact', 'Artifact')]:
        values = pd.to_numeric(raw[source], errors='coerce')
        if (~values.dropna().isin(QUALITY_CODES[target])).any():
            raise ValueError(f'Unrecognized {source} code')
        out[target] = values.astype('Int64')
    out['annotation_status'] = np.where(conflict, 'quarantined_eye_or_grade_conflict', 'verified')
    out['source_annotation'] = str(csv_path.resolve())
    out['source_annotation_sha256'] = sha256_file(csv_path)
    for column in raw:
        out['original_' + column.replace(' ', '_')] = raw[column]
    return out


def resolve_image_path(root: str | Path, original_path: str) -> tuple[Path, str]:
    """Resolve official path layouts, rejecting traversal and ambiguous matches."""
    root = Path(root).expanduser().resolve()
    pure = PureWindowsPath(str(original_path).strip())
    parts = [part for part in pure.parts if part not in {'\\', '/'}]
    if pure.drive or '..' in parts or len(parts) < 3:
        raise ValueError(f'Unsafe or unrecognized annotation image_path: {original_path}')
    split, *tail = parts
    candidates = [root / split / 'Images' / Path(*tail), root / Path(*parts),
                  root / 'regular_fundus_images' / split / 'Images' / Path(*tail),
                  root / 'Images' / Path(*tail), root / Path(*tail)]
    unique = list(dict.fromkeys(path.resolve() for path in candidates))
    found = [path for path in unique if path.is_file()]
    if len(found) > 1:
        return found[0], 'ambiguous_path'
    return (found[0], 'found') if found else (unique[0], 'missing')


def inspect_image(path: Path, path_status: str = 'found') -> dict:
    result = {'image_path': str(path), 'exists': path.is_file(), 'readable': False,
              'image_status': path_status, 'width': None, 'height': None,
              'sha256': None, 'dhash': None, 'phash': None}
    if path_status != 'found':
        return result
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            image.load()
            result.update(width=image.width, height=image.height)
            small = np.asarray(image.convert('L').resize((9, 8), Image.Resampling.BILINEAR))
            bits = (small[:, 1:] > small[:, :-1]).ravel()
            result['dhash'] = f'{int("".join("1" if x else "0" for x in bits), 2):016x}'
            result['phash'] = retinal_phash(image)
        result.update(readable=True, image_status='ok', sha256=sha256_file(path))
    except (OSError, ValueError, Image.DecompressionBombError) as exc:
        result['image_status'] = 'corrupt:' + type(exc).__name__
    return result


def build_deepdrid_manifest(train_csv: str | Path, validation_csv: str | Path,
                            dataset_root: str | Path, seed: int = 2026,
                            calibration_fraction: float = .15,
                            validation_fraction: float = .15,
                            selected_subjects: dict | None = None) -> pd.DataFrame:
    train = parse_deepdrid(train_csv, 'official_train')
    holdout = parse_deepdrid(validation_csv, 'official_validation')
    if set(train.subject_id) & set(holdout.subject_id):
        raise ValueError('Official split patient overlap detected')
    if selected_subjects:
        train = train[train.subject_id_original.isin(map(str, selected_subjects['official_train']))].copy()
        holdout = holdout[holdout.subject_id_original.isin(map(str, selected_subjects['official_validation']))].copy()
    train['split'] = assign_subject_splits(train, seed, calibration_fraction, validation_fraction)
    holdout['split'] = 'holdout'
    combined = pd.concat([train, holdout], ignore_index=True)
    if combined.empty:
        raise ValueError('No annotated images remain after subject selection')
    rows = [inspect_image(*resolve_image_path(dataset_root, value))
            for value in combined.original_image_path]
    for field in rows[0] if rows else []:
        combined[field] = [row[field] for row in rows]
    combined['eligible_for_training'] = combined.readable & combined.annotation_status.eq('verified')
    return combined


def validate_manifest(frame: pd.DataFrame, strict: bool = True,
                      near_duplicate_distance: int = 2, check_near_duplicates: bool = True) -> dict:
    """Reject leakage/duplicates; near duplicates are candidates requiring review.

    A central green-channel pHash suppresses fundus-border false positives.
    Legacy manifests without pHash retain conservative dHash checking until
    revalidated. Cross-split candidates block strict use until review/removal.
    """
    required = {'image_id', 'subject_id', 'split', 'image_path', 'dr_grade', 'quality_grade',
                'annotation_status', 'readable', 'image_status', 'sha256', 'dhash'}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f'Manifest missing fields: {sorted(missing)}')
    errors = []
    if frame.empty:
        errors.append('Manifest contains no images')
    if frame.image_id.duplicated().any():
        errors.append('Duplicate image_id')
    if frame.image_path.duplicated().any():
        errors.append('Multiple records resolve to the same image_path')
    if frame.subject_id.isna().any() or frame.subject_id.astype(str).str.strip().eq('').any():
        errors.append('Missing authoritative subject_id')
    if (~frame.split.isin(ALLOWED_SPLITS)).any():
        errors.append('Unknown split')
    overlaps = frame.groupby('subject_id').split.nunique()
    if (overlaps > 1).any():
        errors.append('Subject leakage across partitions')
    for label, allowed in [('dr_grade', set(range(5))), ('quality_grade', {0, 1})]:
        vals = pd.to_numeric(frame[label], errors='coerce')
        if vals.isna().any() or not vals.isin(allowed).all():
            errors.append(f'Missing/invalid {label}')
    unreadable = int((~frame.readable.fillna(False).eq(True)).sum())
    if unreadable:
        errors.append(f'{unreadable} missing, corrupt or ambiguous images')
    quarantined = int(frame.annotation_status.ne('verified').sum())
    if quarantined:
        errors.append(f'{quarantined} quarantined annotation rows')
    exact = []
    for digest, group in frame.dropna(subset=['sha256']).groupby('sha256'):
        if len(group) > 1:
            exact.append({'sha256': digest, 'image_ids': group.image_id.tolist(),
                          'splits': sorted(group.split.unique().tolist())})
    if exact:
        errors.append(f'{len(exact)} exact duplicate groups; remove/reconcile before training')
    near = []
    fingerprint = 'phash' if 'phash' in frame else 'dhash'
    if check_near_duplicates:
        records = frame.dropna(subset=[fingerprint]).to_dict('records')
        for index, first in enumerate(records):
            for second in records[index + 1:]:
                if first['sha256'] == second['sha256']:
                    continue
                distance = (int(first[fingerprint], 16) ^ int(second[fingerprint], 16)).bit_count()
                if distance <= near_duplicate_distance:
                    near.append({'image_ids': [first['image_id'], second['image_id']],
                                 'distance': distance, 'cross_split': first['split'] != second['split']})
    cross_near = sum(pair['cross_split'] for pair in near)
    if cross_near:
        errors.append(f'{cross_near} cross-partition perceptual duplicate candidates require review')
    partitions = {}
    for split, group in frame.groupby('split'):
        partitions[split] = {'images': len(group), 'subjects': group.subject_id.nunique(),
                             'dr_grade_counts': {str(k): int(v) for k, v in group.dr_grade.value_counts().sort_index().items()},
                             'quality_counts': {str(k): int(v) for k, v in group.quality_grade.value_counts().sort_index().items()}}
    report = {'images': len(frame), 'subjects': frame.subject_id.nunique(), 'unreadable_images': unreadable,
              'quarantined_rows': quarantined, 'partitions': partitions, 'errors': errors,
              'exact_duplicate_groups': exact, 'near_duplicate_candidates': near,
              'near_duplicate_distance': near_duplicate_distance,
              'perceptual_fingerprint': 'central_green_dct_phash_v1' if fingerprint == 'phash' else 'legacy_whole_image_dhash',
              'missing_label_counts': {field: int(frame[field].isna().sum())
                                      for field in ['dr_grade', 'quality_grade', 'clarity', 'field_definition', 'artifact']
                                      if field in frame},
              'ready_for_training': not errors}
    if strict and errors:
        raise ValueError('Manifest is not suitable for training: ' + '; '.join(errors))
    return report


def write_manifest(frame: pd.DataFrame, output: str | Path) -> tuple[Path, Path]:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    report = validate_manifest(frame, strict=False)
    report['manifest_sha256'] = sha256_file(output)
    report_path = output.with_suffix('.audit.json')
    report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    return output, report_path
