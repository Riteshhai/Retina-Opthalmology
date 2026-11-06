"""Safely import a user-provided official APTOS archive into private local data."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
from io import BytesIO
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import sys
import time
from zipfile import ZipFile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd

from qsart.data.aptos import (APTOS_SOURCE, PATIENT_LIMITATION, assign_aptos_splits,
                             fingerprint_components, parse_aptos_labels, write_aptos_outputs)
from qsart.data.manifests import inspect_image, sha256_file


ROLE_PREFIX = 'aptos-v1-temperature-20261103:'
REPRESENTATIVE_PREFIX = 'aptos-v1-representative-20261103:'


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8', newline='\n')


def _labels_bytes(raw):
    frame = pd.read_csv(BytesIO(raw), dtype='string', keep_default_na=False)
    if set(frame) != {'id_code', 'diagnosis'} or frame.empty:
        raise ValueError('Archive train.csv must contain only id_code and diagnosis')
    if not frame.id_code.map(lambda value: bool(re.fullmatch(r'[A-Za-z0-9_-]+', value))).all():
        raise ValueError('Archive contains unsafe image stems')
    if frame.id_code.str.casefold().duplicated().any():
        raise ValueError('Archive contains duplicate image IDs')
    if not pd.to_numeric(frame.diagnosis, errors='coerce').isin(range(5)).all():
        raise ValueError('Archive diagnosis must be integer grades 0 through 4')
    return frame


def archive_plan(archive):
    """Validate all names; select only the exact labeled-training members."""
    with ZipFile(archive) as zipped:
        members = zipped.infolist()
        names = set()
        lookup = {}
        for member in members:
            name = member.filename
            parsed = PurePosixPath(name)
            parts = name.rstrip('/').split('/')
            if (not name or '\\' in name or parsed.is_absolute() or
                    any(part in {'', '.', '..'} or ':' in part for part in parts)):
                raise ValueError('Unsafe archive member path')
            if name.casefold() in names:
                raise ValueError('Duplicate case-insensitive archive member')
            names.add(name.casefold())
            if stat.S_ISLNK(member.external_attr >> 16):
                raise ValueError('Archive symlinks are not accepted')
            if member.flag_bits & 1:
                raise ValueError('Encrypted archive members are not accepted')
            lookup[name] = member
        if 'train.csv' not in lookup or lookup['train.csv'].file_size > 16 * 1024**2:
            raise ValueError('Missing or oversized archive train.csv')
        # ZipFile.read verifies this member's CRC before its labels are parsed.
        labels = _labels_bytes(zipped.read('train.csv'))
        expected = {'train.csv'} | {f'train_images/{value}.png' for value in labels.id_code}
        actual_training = {name for name, member in lookup.items()
                           if name.startswith('train_images/') and not member.is_dir()}
        if actual_training != expected - {'train.csv'}:
            raise ValueError('Training image members do not exactly match train.csv')
        total = sum(lookup[name].file_size for name in expected)
        if total > 100 * 1024**3:
            raise ValueError('Labeled training export exceeds the local import bound')
        return {'training_members': sorted(expected), 'training_images': len(labels),
                'training_bytes': total, 'archive_members': len(members),
                'ignored_members': len(members) - len(expected)}


def extract_labeled_training(archive, destination, *, receipt):
    """Stream member CRC checks; never extract hidden-test files or overwrite data."""
    archive, destination, receipt = Path(archive), Path(destination), Path(receipt)
    if destination.exists() or receipt.exists():
        raise ValueError('Extraction outputs already exist; no data were overwritten')
    plan = archive_plan(archive)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(destination.parent).free < plan['training_bytes'] + 1024**3:
        raise ValueError('Insufficient free disk space for labeled training import')
    archive_sha = sha256_file(archive)
    destination.mkdir()
    root = destination.resolve()
    hashes = {}
    with ZipFile(archive) as zipped:
        for index, name in enumerate(plan['training_members']):
            target = root / name
            if not target.resolve().is_relative_to(root):
                raise ValueError('Archive target escaped the declared extraction directory')
            target.parent.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha256()
            # Exclusive creation protects both originals and partially imported files.
            with zipped.open(name) as source, target.open('xb') as output:
                for chunk in iter(lambda: source.read(1024 * 1024), b''):
                    digest.update(chunk)
                    output.write(chunk)
            hashes[name] = digest.hexdigest()
            if index % 250 == 0:
                print(json.dumps({'event': 'archive_extract', 'members_written': index + 1,
                                  'total_training_members': len(plan['training_members'])}), flush=True)
    result = {**plan, 'archive_sha256': archive_sha, 'archive_bytes': archive.stat().st_size,
              'extracted_sha256': hashes, 'crc_verified_for_all_extracted_members': True,
              'competition_test_extracted': False}
    _write_json(receipt, result)
    return result


def assign_calibration_roles(frame):
    """Split complete calibration components and mark a stable image representative."""
    frame = frame.copy()
    frame['is_group_representative'] = False
    for _, part in frame.groupby('split_group_id'):
        index = min(part.index, key=lambda i: (hashlib.sha256(
            (REPRESENTATIVE_PREFIX + str(frame.at[i, 'image_id'])).encode()).hexdigest(),
            str(frame.at[i, 'image_id'])))
        frame.loc[index, 'is_group_representative'] = True
    calibration = frame.loc[frame.split == 'calibration'].copy()
    groups = sorted(calibration.split_group_id.unique(), key=lambda group: (
        hashlib.sha256((ROLE_PREFIX + str(group)).encode()).hexdigest(), str(group)))
    if len(groups) < 2:
        raise ValueError('At least two calibration components required for separate temperature/CP fitting')
    temperature_groups = set(groups[:len(groups) // 2])
    calibration['calibration_role'] = calibration.split_group_id.map(
        lambda group: 'temperature' if group in temperature_groups else 'conformal')
    return frame, calibration


def _inspect_records(frame, cache, expected_hashes, *, workers):
    """A private cache avoids repeated PNG decoding while rechecking every byte hash."""
    cache = Path(cache)
    project = Path(__file__).resolve().parents[1]
    metadata_path = cache.with_suffix('.metadata.json')
    metadata = {'inspection_source_sha256': {name: sha256_file(project/name) for name in
                ('qsart/data/manifests.py', 'qsart/preprocessing/fundus.py')},
                'images': len(frame), 'annotation_sha256': str(frame.source_annotation_sha256.iloc[0])}
    if cache.is_file():
        if not metadata_path.is_file() or json.loads(metadata_path.read_text(encoding='utf-8')) != metadata:
            raise ValueError('Private inspection cache source/annotation metadata mismatch')
        inspections = pd.read_csv(cache, keep_default_na=False,
                                  dtype={'sha256': str, 'phash': str}).to_dict('records')
        if len(inspections) != len(frame):
            raise ValueError('Private inspection cache population mismatch')
    else:
        inspections = []
        paths = [Path(path) for path in frame.image_path]
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for index, inspected in enumerate(pool.map(inspect_image, paths)):
                inspections.append(inspected)
                if index % 100 == 0:
                    print(json.dumps({'event': 'inspect_images', 'images_inspected': index + 1,
                                      'total': len(paths)}), flush=True)
        cache.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(inspections).to_csv(cache, index=False)
        _write_json(metadata_path, metadata)
    for path, inspection, image_id in zip(frame.image_path, inspections, frame.image_id, strict=True):
        if (str(Path(inspection['image_path']).resolve()) != str(Path(path).resolve()) or
                str(inspection['readable']).casefold() != 'true' or inspection['image_status'] != 'ok'):
            raise ValueError('APTOS contains missing/corrupt images or incompatible cached inspections')
        if inspection['sha256'] != expected_hashes[f'train_images/{image_id}.png']:
            raise ValueError('Image inspection bytes disagree with CRC-checked archive extraction')
        if cache.is_file() and sha256_file(path) != inspection['sha256']:
            raise ValueError('Extracted image bytes changed since verification')
    return inspections


def prepare(args):
    started = time.perf_counter()
    project = Path(__file__).resolve().parents[1]
    for target, allowed in ((args.raw_root, project/'data/raw'),
                            (args.output, project/'data/manifests'), (args.private, project/'artifacts')):
        if not target.resolve().is_relative_to(allowed.resolve()):
            raise ValueError('APTOS private outputs must remain inside their ignored project data/artifact trees')
    if not args.audit.resolve().is_relative_to((project/'reports').resolve()):
        raise ValueError('Public audit must remain in the project reports tree')
    if args.audit.exists() or args.output.exists():
        raise ValueError('Frozen APTOS manifest/audit outputs already exist')
    receipt_path = args.private / 'extraction_receipt.json'
    if args.reuse_extracted:
        receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
        if sha256_file(args.archive) != receipt['archive_sha256']:
            raise ValueError('Original archive changed since extraction')
        if sha256_file(args.raw_root / 'train.csv') != receipt['extracted_sha256']['train.csv']:
            raise ValueError('Extracted annotation bytes changed')
    else:
        receipt = extract_labeled_training(args.archive, args.raw_root, receipt=receipt_path)
    frame = parse_aptos_labels(args.raw_root)
    inspections = _inspect_records(frame, args.private / 'image_inspections.csv',
                                    receipt['extracted_sha256'], workers=args.workers)
    for field in ('exists', 'readable', 'image_status', 'width', 'height', 'sha256', 'phash'):
        frame[field] = [row[field] for row in inspections]
    frame['eligible_for_training'] = True
    included, duplicates, split_report = assign_aptos_splits(frame, conflict_policy=args.conflict_policy)
    components, _ = fingerprint_components(frame)
    quarantined = components.loc[components.component_label_conflict].copy()
    quarantined['annotation_status'] = 'quarantined_conflicting_duplicate_component'
    quarantined['eligible_for_training'] = False
    quarantined['split'] = 'quarantine'
    included, roles = assign_calibration_roles(included)
    split_report.update(source_url=APTOS_SOURCE, source_annotation_sha256=sha256_file(args.raw_root/'train.csv'),
                        dataset_source='APTOS_2019', supervision={'dr_grade': 'official diagnosis 0-4',
                        'quality': 'unavailable', 'patient_id': 'unavailable', 'eye_id': 'unavailable'})
    write_aptos_outputs(included, duplicates, split_report, args.output)
    development = included.loc[included.split != 'test'].copy()
    development.to_csv(args.output / 'development.csv', index=False)
    roles.to_csv(args.output / 'calibration_roles.csv', index=False)
    quarantined.to_csv(args.output / 'quarantine_report.csv', index=False)
    public = {**split_report, 'schema_version': 1, 'experiment_id': 'aptos_v1',
        'dataset': 'APTOS_2019', 'scope': 'local_source_integrity_and_declared_split_preparation',
        'status': 'verified', 'source_confirmation': 'User confirmed official Kaggle competition download on 2026-11-03',
        'archive_sha256': receipt['archive_sha256'], 'archive_bytes': receipt['archive_bytes'],
        'archive_member_count': receipt['archive_members'], 'training_members_extracted': len(receipt['training_members']),
        'training_images_extracted': receipt['training_images'], 'training_bytes_extracted': receipt['training_bytes'],
        'crc_verified_for_all_extracted_members': receipt['crc_verified_for_all_extracted_members'],
        'competition_test_extracted': False, 'reserved_test_inference_performed': False,
        'manifest_sha256': sha256_file(args.output / 'dataset_manifest.csv'),
        'development_manifest_sha256': sha256_file(args.output / 'development.csv'),
        'calibration_roles_sha256': sha256_file(args.output / 'calibration_roles.csv'),
        'private_inspection_sha256': sha256_file(args.private / 'image_inspections.csv'),
        'private_quarantine_sha256': sha256_file(args.output / 'quarantine_report.csv'),
        'calibration_role_prefix': ROLE_PREFIX, 'representative_prefix': REPRESENTATIVE_PREFIX,
        'calibration_roles': {str(role): {'images': len(part), 'groups': int(part.split_group_id.nunique()),
                              'representatives': int(part.is_group_representative.sum())}
                             for role, part in roles.groupby('calibration_role')},
        'representative_counts': {str(split): int(part.is_group_representative.sum())
                                  for split, part in included.groupby('split')},
        'source_sha256': {path: sha256_file(project/path) for path in
                          ('qsart/data/aptos.py', 'qsart/data/manifests.py', 'scripts/prepare_aptos_archive.py')},
        'limitations': [PATIENT_LIMITATION, 'Source provenance is user-confirmed; no published official archive digest was supplied.',
            'Whole conflicting components are excluded without relabeling; this label-dependent exclusion changes the study population.',
            'Full-source labels are read for quarantine and stratification. Reserved-test inference is prohibited.',
            'No authoritative APTOS quality annotations are available; no quality-label or triage claim is generated.'],
        'elapsed_seconds': time.perf_counter() - started}
    text = json.dumps(public, allow_nan=False)
    if any(token in text for token in ('"image_id"', '"image_path"', str(project), 'C:\\Users\\')):
        raise ValueError('Public APTOS audit contains a private identifier/path')
    _write_json(args.audit, public)
    print(json.dumps(public, indent=2), flush=True)
    return public


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--raw-root', type=Path, default=Path('data/raw/aptos2019'))
    parser.add_argument('--output', type=Path, default=Path('data/manifests/aptos_v1'))
    parser.add_argument('--private', type=Path, default=Path('artifacts/aptos_v1/data'))
    parser.add_argument('--audit', type=Path, default=Path('reports/aptos_v1/data_audit.json'))
    parser.add_argument('--workers', type=int, choices=(1, 2), default=2)
    parser.add_argument('--conflict-policy', choices=('error', 'quarantine'), default='error')
    parser.add_argument('--reuse-extracted', action='store_true')
    prepare(parser.parse_args())


if __name__ == '__main__':
    main()
