"""Inventory local research data without downloads or disclosing paths/identifiers."""
from __future__ import annotations

import argparse
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
import psutil
import torch

from qsart.data.aptos import (APTOS_SOURCE, DEFAULT_FRACTIONS, PATIENT_LIMITATION,
                             build_aptos_manifest, write_aptos_outputs)
from qsart.data.manifests import sha256_file


def manifest_summary(path: Path, *, role: str) -> dict:
    if not path.is_file():
        return {'status': 'missing', 'role': role}
    frame = pd.read_csv(path)
    required = {'split', 'dr_grade', 'dataset_source'}
    if frame.empty or not required.issubset(frame.columns):
        return {'status': 'invalid_manifest', 'role': role}
    partitions = {}
    for split, group in frame.groupby('split'):
        partitions[str(split)] = {'images': len(group)}
        for column in ('dr_grade', 'quality_grade'):
            if column in group:
                partitions[str(split)][column + '_counts'] = {
                    str(value): int(count) for value, count in group[column].value_counts().sort_index().items()}
        if 'subject_id' in group:
            partitions[str(split)]['subjects'] = int(group.subject_id.nunique())
    return {'status': 'local_manifest_present', 'role': role, 'manifest_sha256': sha256_file(path),
            'images': len(frame), 'dataset_sources': sorted(frame.dataset_source.dropna().astype(str).unique()),
            'partitions': partitions,
            'integrity_note': 'Inventory reads manifest metadata; it does not revalidate every current image byte.'}


def inspect_local_datasets(*, aptos_root: Path, deepdrid_development: Path,
                           deepdrid_confirmation: Path) -> dict:
    aptos_csv = aptos_root / 'train.csv'
    aptos = {
        'status': 'local_labeled_export_present_not_yet_validated' if aptos_csv.is_file() and
                  (aptos_root / 'train_images').is_dir() else 'official_labeled_export_missing',
        'official_source': APTOS_SOURCE, 'automatic_download_attempted': False,
        'patient_identifiers_available_from_standard_annotations': False,
        'quality_labels_available_from_standard_annotations': False,
        'patient_grouping_limitation': PATIENT_LIMITATION,
        'next_action': 'Obtain the official labeled export under Kaggle competition terms, then run '
                       'this script with --aptos-root and --prepare-aptos under a frozen split protocol.',
    }
    if aptos_csv.is_file():
        aptos['annotation_file_sha256'] = sha256_file(aptos_csv)
    versions = {}
    for package in ('torch', 'torchvision', 'numpy', 'pandas', 'scikit-learn', 'Pillow'):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = 'not_installed'
    # Presence only: never read, emit or validate credential contents.
    credentials = {
        'kaggle_api_token_environment_present': bool(os.getenv('KAGGLE_API_TOKEN')),
        'kaggle_legacy_environment_pair_present': bool(os.getenv('KAGGLE_USERNAME') and os.getenv('KAGGLE_KEY')),
        'kaggle_credentials_file_present': (Path.home() / '.kaggle' / 'kaggle.json').is_file(),
    }
    return {
        'schema_version': 1, 'aptos': aptos,
        'deepdrid_development': manifest_summary(deepdrid_development,
                                               role='official-training development; fitting/selection only'),
        'deepdrid_confirmation': manifest_summary(deepdrid_confirmation,
                                                role='consumed historical confirmation; no future tuning'),
        'environment': {'python': platform.python_version(), 'operating_system': platform.system(),
                        'logical_cpu_count': psutil.cpu_count(),
                        'physical_memory_gib': round(psutil.virtual_memory().total / 1024**3, 2),
                        'cuda_available': bool(torch.cuda.is_available()), 'package_versions': versions},
        'credential_presence_only': credentials,
        'other_datasets': {
            'EyeQ': 'Annotation access alone does not establish authorized original-image access.',
            'IDRiD': 'Reserved independent evaluation; no fitting/tuning or automatic download by this script.',
        },
        'privacy': 'Aggregate inventory only. Local paths, image IDs, patient IDs, weights and secrets are omitted.',
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--aptos-root', type=Path,
                        default=Path(os.getenv('QSART_APTOS_ROOT', 'data/raw/aptos2019')))
    parser.add_argument('--deepdrid-development', type=Path,
                        default=Path('data/processed/deepdrid/development_v1.csv'))
    parser.add_argument('--deepdrid-confirmation', type=Path,
                        default=Path('data/processed/deepdrid/confirmation_v1.csv'))
    parser.add_argument('--output', type=Path, default=Path('reports/dataset_inventory.json'))
    parser.add_argument('--prepare-aptos', action='store_true',
                        help='Validate official local files and freeze a new private manifest; no downloads')
    parser.add_argument('--aptos-output', type=Path, default=Path('data/manifests/aptos_spec_v1'))
    parser.add_argument('--split-seed', type=int, default=20261020)
    parser.add_argument('--split-fractions', type=float, nargs=4,
                        metavar=('TRAIN', 'VALIDATION', 'CALIBRATION', 'TEST'),
                        default=list(DEFAULT_FRACTIONS.values()),
                        help='Predeclare four group fractions before ingesting a dataset; default .60 .15 .15 .10')
    parser.add_argument('--near-duplicate-distance', type=int, default=2)
    args = parser.parse_args()
    summary = inspect_local_datasets(aptos_root=args.aptos_root, deepdrid_development=args.deepdrid_development,
                                     deepdrid_confirmation=args.deepdrid_confirmation)
    if args.prepare_aptos:
        # Refuse to export per-image IDs outside already ignored local artifact trees.
        private_root = Path(__file__).resolve().parents[1]
        target = args.aptos_output.expanduser().resolve()
        allowed = [private_root / 'data' / 'manifests', private_root / 'data' / 'processed', private_root / 'artifacts']
        if not any(target.is_relative_to(root.resolve()) for root in allowed):
            parser.error('--aptos-output must be inside ignored data/manifests, data/processed or artifacts')
        try:
            frame, duplicates, report = build_aptos_manifest(
                args.aptos_root, seed=args.split_seed, fractions=dict(zip(DEFAULT_FRACTIONS, args.split_fractions)),
                near_duplicate_distance=args.near_duplicate_distance)
            write_aptos_outputs(frame, duplicates, report, target)
        except ValueError as exc:
            parser.error(str(exc))
        summary['aptos'].update(status='local_export_validated_manifest_frozen', validation=report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
