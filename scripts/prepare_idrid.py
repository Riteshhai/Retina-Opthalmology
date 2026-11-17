"""Freeze IDRiD external-data integrity evidence without any model inference."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd

from qsart.data.idrid import audit_idrid, content_digest, validate_fingerprints
from qsart.data.manifests import sha256_file

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ('qsart/data/idrid.py', 'qsart/data/aptos.py', 'qsart/data/manifests.py',
           'qsart/preprocessing/fundus.py', 'scripts/prepare_idrid.py',
           'configs/idrid_data_v1.json', 'docs/research/IDRID_DATA_V1_PROTOCOL.md')


def default_idrid_root(project_root=None):
    project = Path(project_root or ROOT).resolve()
    configured = os.getenv('QSART_IDRID_ROOT', '').strip()
    settings = project/'.env'
    if not configured and settings.is_file():
        for line in settings.read_text(encoding='utf-8').splitlines():
            key, separator, value = line.partition('=')
            if separator and key.strip() == 'QSART_IDRID_ROOT':
                configured = value.strip().strip('\"\'')
    path = Path(configured).expanduser() if configured else Path('../idrid/B. Disease Grading')
    return (path if path.is_absolute() else project/path).resolve()


def load_references(config, root=ROOT):
    if config.get('near_duplicate_distance') != 2 or len(config.get('references', [])) != 4:
        raise ValueError('The declared distance and four frozen reference manifests are required')
    frames, bindings = [], []
    for item in config['references']:
        path = (Path(root)/item['path']).resolve()
        if not path.is_relative_to(Path(root).resolve()) or sha256_file(path) != item['sha256']:
            raise ValueError('Reference path or frozen manifest hash mismatch')
        frame = pd.read_csv(path, dtype={'phash': str, 'sha256': str, 'image_id': str})
        if len(frame) != item['rows'] or not frame.dataset_source.eq(item['dataset']).all():
            raise ValueError('Reference dataset identity or row count mismatch')
        validate_fingerprints(frame)
        frames.append(frame[['dataset_source', 'image_id', 'sha256', 'phash']])
        bindings.append(item)
    combined = pd.concat(frames, ignore_index=True)
    validate_fingerprints(combined)
    return combined, bindings


def validate_output_paths(private, public, root=ROOT):
    root = Path(root).resolve()
    private, public = Path(private).resolve(), Path(public).resolve()
    if not any(private.is_relative_to(root/name) for name in ('data/manifests', 'artifacts')):
        raise ValueError('Private IDRiD outputs must stay in ignored manifests or artifacts')
    if not public.is_relative_to(root/'reports') or public.is_relative_to(private):
        raise ValueError('Public aggregate report must stay under reports, outside private outputs')
    if private.exists() or public.exists():
        raise ValueError('Preserve completed or partial IDRiD outputs; choose a new output identity')


def execute(args):
    private, public = args.private.resolve(), args.output.resolve()
    validate_output_paths(private, public)
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    if (revision != args.source_commit or
            subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip()):
        raise ValueError('IDRiD execution requires the declared clean committed source')
    bindings = {}
    for name in SOURCES:
        expected = subprocess.check_output(['git', 'show', f'{revision}:{name}'], cwd=ROOT)
        if expected != (ROOT/name).read_bytes():
            raise ValueError('Declared IDRiD execution source differs from committed bytes')
        bindings[name] = sha256_file(ROOT/name)
    config = json.loads((ROOT/'configs/idrid_data_v1.json').read_text())
    references, reference_bindings = load_references(config)
    frame, internal, cross, report = audit_idrid(args.root, references, distance=config['near_duplicate_distance'])
    private.mkdir(parents=True)
    for name, records in [('external_manifest.csv', frame), ('internal_candidates.csv', internal),
                          ('cross_source_candidates.csv', cross)]:
        records.to_csv(private/name, index=False, lineterminator='\n')
    report.update(source_commit=revision, execution_source_sha256=bindings,
        audited_at_utc=datetime.now(timezone.utc).isoformat(), reference_manifest_bindings=reference_bindings,
        source_provenance_status=('official_download_user_confirmed' if args.source_confirmed_official
                                  else 'local_package_download_origin_unconfirmed'),
        source_confirmation_note='User source statement is separate from byte authentication against an official archive.',
        image_content_sha256=content_digest(frame),
        private_output_sha256={name: sha256_file(private/name) for name in
                              ('external_manifest.csv', 'internal_candidates.csv', 'cross_source_candidates.csv')})
    serialized = json.dumps(report, indent=2, allow_nan=False)+'\n'
    if any(token in serialized for token in ('"image_id"', '"image_path"', '"subject_id"', str(ROOT), 'C:\\Users\\')):
        raise ValueError('Private identities or local paths in the public aggregate report')
    public.parent.mkdir(parents=True, exist_ok=True)
    with public.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(serialized)
    print(json.dumps({'status': 'data_audit_completed_external_evaluation_not_performed',
        'images': report['images'], 'cross_source_candidate_pairs': len(cross),
        'source_provenance_status': report['source_provenance_status'], 'report_sha256': sha256_file(public)}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=default_idrid_root())
    parser.add_argument('--private', type=Path, default=ROOT/'data/manifests/idrid_v1')
    parser.add_argument('--output', type=Path, default=ROOT/'reports/idrid_data_v1/data_audit.json')
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--source-confirmed-official', action='store_true',
                        help='Record only an actual user confirmation of the official download source')
    args = parser.parse_args()
    try:
        execute(args)
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
