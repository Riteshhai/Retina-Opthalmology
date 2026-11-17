"""Acquire frozen DeepDRiD development data; keep new confirmation prediction-free."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path, PureWindowsPath
import sys
import threading
import time
from urllib.request import urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd

from qsart.data.development import (check_confirmation_independence, defect_mapping_report,
                                    duplicate_safe_splits, select_development_and_confirmation)
from qsart.data.manifests import (inspect_image, parse_deepdrid, resolve_image_path,
                                  sha256_file, validate_manifest)

COMMIT = '56d8af71319803150d0e7b90533e0918c7bf85ee'
BASE = f'https://raw.githubusercontent.com/deepdrdoc/DeepDRiD/{COMMIT}/regular_fundus_images/'
METADATA_SHA256 = {
    'regular-fundus-training.csv': '0f26db5cd2cf14939f6a96732a7e9e5aef243ae70a29e099e678d412cb4c47b6',
    'regular-fundus-validation.csv': '54c2f11bf0647fe3cc64496cd83646ea4a4fdf0aab82b1cd4891ad998f220332',
    'Readme.docx': '50543a329d15c0954721c0c1a20289bc4d8d98b9ef0eeea1b963768e12ea3bb5',
    'LICENSE': '87a816969906840bf7af8d4d01cdfad4741b18946365e1f286007935509f2edb',
}


class TransferBudget:
    """Reserve bytes before reading each response so concurrent reads cannot exceed the cap."""
    def __init__(self, limit):
        self.limit, self.used, self.lock = int(limit), 0, threading.Lock()
        if self.limit <= 0:
            raise ValueError('Additional transfer budget must be positive')

    def read(self, response):
        with self.lock:
            reserved = min(64 * 1024, self.limit - self.used)
            if reserved <= 0:
                raise ValueError('Additional network transfer cap exhausted')
            self.used += reserved
        try:
            block = response.read(reserved)
        except Exception:
            # Reservations remain charged when a transport fails mid-read.
            raise
        with self.lock:
            self.used -= reserved - len(block)
        return block


def acquire_images(frame, root, budget, workers=8):
    records = []
    def one(row):
        target, status = resolve_image_path(root, row['original_image_path'])
        if status == 'ambiguous_path':
            raise ValueError(f'Ambiguous actual image path: {row["image_id"]}')
        parts = [part for part in PureWindowsPath(row['original_image_path'].strip()).parts if part != '\\']
        url = BASE + parts[0] + '/Images/' + '/'.join(parts[1:])
        reused = target.exists()
        if not reused:
            target.parent.mkdir(parents=True, exist_ok=True)
            temp = target.with_suffix('.jpg.part')
            for attempt in range(3):
                try:
                    size = 0
                    with urlopen(url, timeout=60) as response, temp.open('wb') as stream:
                        while True:
                            block = budget.read(response)
                            if not block:
                                break
                            size += len(block)
                            if size > 20 * 1024 * 1024:
                                raise ValueError('Source JPEG exceeds the 20MiB per-image limit')
                            stream.write(block)
                    temp.replace(target)
                    break
                except (OSError, TimeoutError):
                    if attempt == 2:
                        raise
                    time.sleep(attempt + 1)
        inspected = inspect_image(target, 'found')
        if not inspected['readable']:
            raise ValueError(f'Actual image failed readability: {row["image_id"]}')
        return {'image_id': row['image_id'], **inspected, 'source_image_url': url,
                'source_commit': COMMIT, 'source_bytes': target.stat().st_size, 'reused_existing_file': reused}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(one, row) for row in frame.to_dict('records')]
        for future in as_completed(futures):
            records.append(future.result())
            if len(records) % 50 == 0 or len(records) == len(frame):
                print(f'Verified {len(records)}/{len(frame)} images; additional transfer {budget.used / 1024**2:.1f} MiB', flush=True)
    inspected = pd.DataFrame(records).set_index('image_id')
    result = frame.copy()
    for column in inspected:
        result[column] = result.image_id.map(inspected[column])
    result['eligible_for_training'] = result.readable & result.annotation_status.eq('verified')
    return result


def freeze(frame, path, details):
    if path.exists():
        raise FileExistsError(f'Frozen manifest already exists, choose a new version: {path}')
    audit = validate_manifest(frame, strict=False)
    if audit['errors']:
        failed = path.with_suffix('.unresolved_audit.json')
        failed.write_text(json.dumps(audit, indent=2), encoding='utf-8')
        raise ValueError(f'Manifest not frozen because audit failed: {failed}')
    frame.to_csv(path, index=False)
    audit['manifest_sha256'] = sha256_file(path)
    audit['frozen'] = True
    path.with_suffix('.audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    provenance = {**details, 'manifest_sha256': audit['manifest_sha256'], 'images': len(frame),
                  'subject_count': frame.subject_id.nunique(),
                  'total_source_bytes': int(frame.source_bytes.sum()),
                  'reused_images': int(frame.reused_existing_file.sum()),
                  'newly_downloaded_images': int((~frame.reused_existing_file).sum()),
                  'newly_downloaded_source_bytes': int(frame.loc[~frame.reused_existing_file, 'source_bytes'].sum()),
                  'image_urls_hashes': frame[['image_id', 'source_image_url', 'sha256', 'source_bytes']].to_dict('records')}
    path.with_suffix('.provenance.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')
    print(f'FROZEN {path} SHA256={audit["manifest_sha256"]}', flush=True)
    print(json.dumps(audit['partitions'], indent=2), flush=True)
    return provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(os.environ.get('QSART_DATA_ROOT', 'data/raw/deepdrid')))
    parser.add_argument('--pilot-manifest', type=Path, default=Path('data/processed/deepdrid/pilot_manifest.csv'))
    parser.add_argument('--output-dir', type=Path, default=Path('data/processed/deepdrid'))
    parser.add_argument('--maximum-additional-mib', type=int, default=1500)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--confirmation-only', action='store_true', help='After development freeze, acquire the reserved85patient cohort only')
    args = parser.parse_args()
    if not 1 <= args.workers <= 12:
        parser.error('Use between1and12 bounded download workers')
    root, output = args.root.resolve(), args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    for filename, expected in METADATA_SHA256.items():
        if sha256_file(root / 'metadata' / filename) != expected:
            raise ValueError(f'Local metadata differs from verified pinned official source: {filename}')
    pilot_hash_before = sha256_file(args.pilot_manifest)
    train = parse_deepdrid(root / 'metadata/regular-fundus-training.csv', 'official_train')
    official_validation = parse_deepdrid(root / 'metadata/regular-fundus-validation.csv', 'official_validation')
    development, confirmation, selection = select_development_and_confirmation(
        train, official_validation, pd.read_csv(args.pilot_manifest))
    inspected = set(selection['previously_evaluated_pilot_holdout_subjects'])
    check_confirmation_independence(confirmation, development, inspected)
    development_path, confirmation_path = output / 'development_v1.csv', output / 'confirmation_v1.csv'
    details = {'source_repository': 'https://github.com/deepdrdoc/DeepDRiD', 'source_commit': COMMIT,
               'license': 'CC-BY-SA-4.0', 'metadata_sha256': METADATA_SHA256,
               'pilot_manifest_sha256_preserved': pilot_hash_before,
               'selection': selection, 'defect_targets': defect_mapping_report(),
               'confirmation_predictions_run': False}
    # This selection contains original annotations only, not any model outputs.
    (output / 'development_v1.defect_mappings.json').write_text(json.dumps(defect_mapping_report(), indent=2), encoding='utf-8')
    (output / 'confirmation_v1.reservation.json').write_text(json.dumps({
        **details, 'reserved_subjects': sorted(confirmation.subject_id.unique()),
        'reserved_images': len(confirmation), 'status': 'reserved; no model predictions'}, indent=2), encoding='utf-8')
    if not args.confirmation_only:
        if development_path.exists():
            raise FileExistsError('development_v1 is frozen; use --confirmation-only or a new output directory')
        budget = TransferBudget(args.maximum_additional_mib * 1024**2)
        development = acquire_images(development, root, budget, args.workers)
        development, grouping = duplicate_safe_splits(development, args.seed)
        details.update(split_assignment=grouping, additional_network_bytes_charged=budget.used)
        freeze(development, development_path, details)
    else:
        prior = json.loads(development_path.with_suffix('.provenance.json').read_text(encoding='utf-8'))
        if sha256_file(development_path) != prior['manifest_sha256']:
            raise ValueError('Frozen development manifest changed before confirmation acquisition')
        budget = TransferBudget(args.maximum_additional_mib * 1024**2 - prior['additional_network_bytes_charged'])
        confirmation = acquire_images(confirmation, root, budget, args.workers)
        details.update(additional_network_bytes_charged=budget.used,
                        development_plus_confirmation_network_bytes=prior['additional_network_bytes_charged'] + budget.used,
                        associated_development_manifest_sha256=prior['manifest_sha256'])
        freeze(confirmation, confirmation_path, details)
    if sha256_file(args.pilot_manifest) != pilot_hash_before:
        raise RuntimeError('Existing pilot manifest unexpectedly changed')


if __name__ == '__main__':
    main()
