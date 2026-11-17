"""Prepare verified manifests, optionally acquiring an official research pilot."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import time
from urllib.request import urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd

from qsart.data.manifests import (build_deepdrid_manifest, parse_deepdrid,
                                  sha256_file, write_manifest)

COMMIT = '56d8af71319803150d0e7b90533e0918c7bf85ee'
BASE = f'https://raw.githubusercontent.com/deepdrdoc/DeepDRiD/{COMMIT}/'


def _download(url: str, target: Path, maximum_bytes: int = 20 * 1024 * 1024) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(3):
        try:
            with urlopen(url, timeout=45) as response:
                content = response.read(maximum_bytes + 1)
            if len(content) > maximum_bytes:
                raise ValueError(f'Download exceeds per-file limit: {url}')
            temp = target.with_suffix(target.suffix + '.part')
            temp.write_bytes(content)
            temp.replace(target)
            return
        except (OSError, TimeoutError):
            if attempt == 2:
                raise
            time.sleep(attempt + 1)


def acquire_metadata(root: Path) -> dict:
    files = {'regular-fundus-training.csv': 'regular_fundus_images/regular-fundus-training/regular-fundus-training.csv',
             'regular-fundus-validation.csv': 'regular_fundus_images/regular-fundus-validation/regular-fundus-validation.csv',
             'Readme.docx': 'regular_fundus_images/regular-fundus-training/Readme.docx', 'LICENSE': 'LICENSE'}
    provenance = {'repository': 'https://github.com/deepdrdoc/DeepDRiD', 'commit': COMMIT,
                  'license': 'CC-BY-SA-4.0', 'files': {}}
    for name, source in files.items():
        target = root / 'metadata' / name
        _download(BASE + source, target)
        provenance['files'][name] = {'url': BASE + source, 'sha256': sha256_file(target)}
    return provenance


def select_pilot(train_csv: Path, holdout_csv: Path, train_per_grade: int,
                 holdout_per_grade: int, seed: int = 2026) -> dict:
    """Predetermine class-stratified subjects, before any model fitting.

    Exclude entire subjects with ambiguous annotations; never correct their eyes.
    Holdout class stratification is a sampling definition, not model selection.
    This enriched pilot does not estimate population prevalence/performance.
    """
    if train_per_grade < 3 or holdout_per_grade < 1:
        raise ValueError('Require >=3 train subjects and >=1 holdout subject per grade')
    selection = {}
    for csv, split, count in [(train_csv, 'official_train', train_per_grade),
                               (holdout_csv, 'official_validation', holdout_per_grade)]:
        frame = parse_deepdrid(csv, split)
        excluded = set(frame.loc[frame.annotation_status.ne('verified'), 'subject_id_original'])
        frame = frame[~frame.subject_id_original.isin(excluded)]
        grades = frame.groupby('subject_id_original').dr_grade.max()
        chosen = []
        for grade in range(5):
            ids = grades[grades.eq(grade)].index.tolist()
            ids.sort(key=lambda subject: hashlib.sha256(f'{seed}:{split}:{subject}'.encode()).hexdigest())
            if len(ids) < count:
                raise ValueError(f'Not enough grade-{grade} subjects in {split}: {len(ids)} < {count}')
            chosen.extend(ids[:count])
        selection[split] = sorted(chosen, key=int)
    return selection


def download_pilot(root: Path, selected: dict, maximum_mb: int = 500) -> dict:
    records = []
    lock = threading.Lock()
    used = 0
    jobs = []
    for name, split in [('regular-fundus-training', 'official_train'),
                         ('regular-fundus-validation', 'official_validation')]:
        raw = pd.read_csv(root / 'metadata' / f'{name}.csv')
        raw = raw[raw.patient_id.astype(str).isin(selected[split])]
        for row in raw.itertuples():
            relative = f'regular_fundus_images/{name}/Images/{row.patient_id}/{row.image_id}.jpg'
            target = root / name / 'Images' / str(row.patient_id) / f'{row.image_id}.jpg'
            jobs.append((BASE + relative, target, row.image_id))

    def one(job):
        nonlocal used
        url, target, image_id = job
        if not target.exists():
            _download(url, target)
        size = target.stat().st_size
        with lock:
            used += size
            if used > maximum_mb * 1024 * 1024:
                raise ValueError('Pilot download byte cap exceeded; do not enlarge without a deliberate choice')
        return {'image_id': image_id, 'source_url': url, 'sha256': sha256_file(target), 'bytes': size}

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(one, job) for job in jobs]
        for future in as_completed(futures):
            records.append(future.result())
            if len(records) % 20 == 0 or len(records) == len(jobs):
                print(f'Acquired {len(records)}/{len(jobs)} images ({used / 1024**2:.1f} MiB)', flush=True)
    return {'image_count': len(records), 'total_bytes': used,
            'images': sorted(records, key=lambda item: item['image_id'])}


def main():
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument('--config', type=Path, help='Optional data YAML; CLI/environment paths take precedence')
    initial, _ = config_parser.parse_known_args()
    config = {}
    if initial.config:
        import yaml
        config = yaml.safe_load(initial.config.read_text(encoding='utf-8')) or {}
        if not isinstance(config, dict):
            raise ValueError('Data YAML must contain a mapping')
    parser = argparse.ArgumentParser(description=__doc__, parents=[config_parser])
    parser.add_argument('--root', type=Path, default=Path(os.environ.get('QSART_DATA_ROOT', config.get('dataset_root', 'data/raw/deepdrid'))))
    parser.add_argument('--train-csv', type=Path)
    parser.add_argument('--validation-csv', type=Path)
    parser.add_argument('--out', type=Path, default=Path(config.get('manifest', 'data/processed/deepdrid/manifest.csv')))
    parser.add_argument('--download-pilot', action='store_true')
    parser.add_argument('--train-per-grade', type=int, default=10)
    parser.add_argument('--holdout-per-grade', type=int, default=3)
    parser.add_argument('--maximum-mb', type=int, default=500)
    parser.add_argument('--seed', type=int, default=int(config.get('seed', 2026)))
    parser.add_argument('--calibration-fraction', type=float, default=float(config.get('calibration_fraction', .15)))
    parser.add_argument('--validation-fraction', type=float, default=float(config.get('validation_fraction', .15)))
    parser.add_argument('--selection', type=Path, help='Previously saved pilot subject selection JSON')
    args = parser.parse_args()
    root = args.root.resolve()
    selected, provenance = None, None
    if args.selection:
        selected = json.loads(args.selection.read_text(encoding='utf-8'))
    train_csv = args.train_csv or root / config.get('train_annotations', 'metadata/regular-fundus-training.csv')
    holdout_csv = args.validation_csv or root / config.get('holdout_annotations', 'metadata/regular-fundus-validation.csv')
    if args.download_pilot:
        provenance = acquire_metadata(root)
        selected = selected or select_pilot(train_csv, holdout_csv, args.train_per_grade,
                                            args.holdout_per_grade, args.seed)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.with_suffix('.selection.json').write_text(json.dumps(selected, indent=2), encoding='utf-8')
        provenance['pilot_sampling'] = {'seed': args.seed, 'train_subjects_per_max_grade': args.train_per_grade,
                                        'holdout_subjects_per_max_grade': args.holdout_per_grade}
        provenance['download'] = download_pilot(root, selected, args.maximum_mb)
        args.out.with_suffix('.provenance.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')
    frame = build_deepdrid_manifest(train_csv, holdout_csv, root, seed=args.seed,
                                    selected_subjects=selected,
                                    calibration_fraction=args.calibration_fraction,
                                    validation_fraction=args.validation_fraction)
    manifest, report = write_manifest(frame, args.out)
    print(f'Saved {len(frame)} actual annotation records: {manifest}\nAudit: {report}')
    audit = json.loads(report.read_text())
    print(json.dumps({key: audit[key] for key in ['images', 'subjects', 'unreadable_images', 'quarantined_rows', 'errors', 'partitions']}, indent=2))


if __name__ == '__main__':
    main()
