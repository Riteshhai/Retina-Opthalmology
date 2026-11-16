"""Recheck frozen local dataset imports without downloads, training, or inference."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
from io import BytesIO
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd

from api.datasets import dataset_catalog
from qsart.data.aptos import parse_aptos_labels
from qsart.data.external_evaluation import POPULATIONS, validate_external
from qsart.data.eyeq import annotation_profile, load_annotations, verify_source_files
from qsart.data.idrid import parse_idrid_labels
from qsart.data.manifests import sha256_file
from scripts.inspect_datasets import default_aptos_root
from scripts.prepare_idrid import default_idrid_root


ROOT = Path(__file__).resolve().parents[1]


class VerificationError(ValueError):
    """A safe aggregate diagnostic that contains no row identifiers or paths."""


def require(condition, message):
    if not condition:
        raise VerificationError(message)


def report(name):
    return json.loads((ROOT / 'reports' / name).read_bytes())


def bound_bytes(path, expected):
    raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == expected, 'A frozen manifest or witness digest differs.')
    return raw


def manifest(path, expected):
    return pd.read_csv(BytesIO(bound_bytes(path, expected)), dtype={'image_id': str, 'sha256': str})


def verify_photos(paths_and_hashes):
    def matches(pair):
        path, digest = pair
        return sha256_file(path) == digest

    with ThreadPoolExecutor(max_workers=4) as workers:
        require(all(workers.map(matches, paths_and_hashes)), 'A local photograph differs from its frozen SHA-256.')


def match_source(fresh, frozen, columns):
    joined = fresh.merge(frozen[['image_id', 'sha256', *columns]], on='image_id', how='outer',
                         validate='one_to_one', indicator=True, suffixes=('', '_frozen'))
    require(joined['_merge'].eq('both').all()
            and all(joined[column].eq(joined[f'{column}_frozen']).all() for column in columns),
            'The local source inventory or labels differ from the frozen manifest.')
    verify_photos(zip(joined.image_path, joined.sha256, strict=True))


def verify_aptos(root):
    audit = report('aptos_v1/data_audit.json')
    folder = ROOT / 'data/manifests/aptos_v1'
    full = manifest(folder / 'dataset_manifest.csv', audit['manifest_sha256'])
    development = manifest(folder / 'development.csv', audit['development_manifest_sha256'])
    quarantine = manifest(folder / 'quarantine_report.csv', audit['private_quarantine_sha256'])
    calibration = manifest(folder / 'calibration_roles.csv', audit['calibration_roles_sha256'])

    require(full.split.value_counts().to_dict() == {role: row['images'] for role, row in audit['partitions'].items()}
            and full.groupby('split_group_id').split.nunique().eq(1).all()
            and development.equals(full.loc[full.split.ne('test')].reset_index(drop=True)),
            'APTOS development and reserved-test partitions differ from the frozen split.')
    require(len(quarantine) == audit['quarantined_images'] and quarantine.split.eq('quarantine').all()
            and set(full.split_group_id).isdisjoint(quarantine.split_group_id),
            'APTOS conflict quarantine overlaps an accepted partition.')
    require(calibration.drop(columns='calibration_role').equals(full.loc[full.split.eq('calibration')].reset_index(drop=True))
            and calibration.calibration_role.value_counts().to_dict() == {
                role: row['images'] for role, row in audit['calibration_roles'].items()}
            and calibration.groupby('split_group_id').calibration_role.nunique().eq(1).all(),
            'APTOS temperature and conformal roles differ from the frozen split.')

    fresh = parse_aptos_labels(root)
    require(fresh.source_annotation_sha256.eq(audit['source_annotation_sha256']).all()
            and {path.resolve() for path in (root / 'train_images').glob('*.png')} == set(map(Path, fresh.image_path)),
            'APTOS labeled source files differ from the audited inventory.')
    match_source(fresh, pd.concat([full, quarantine], ignore_index=True), ['dr_grade', 'official_split'])
    return {'photos_verified': len(fresh), 'excluded_photos': len(quarantine),
            'development_photos': len(development), 'reserved_internal_test_photos': int(full.split.eq('test').sum())}


def verify_deepdrid(author_root):
    summary = report('development_v1/data_summary.json')
    online = report('deepdrid_online_readiness_v1/evaluation.json')
    folder = ROOT / 'data/processed/deepdrid'
    development = manifest(folder / 'development_v1.csv', summary['development_v1']['manifest_sha256'])
    confirmation = manifest(folder / 'confirmation_v1.csv', summary['confirmation_v1']['manifest_sha256'])
    bound_bytes(folder / 'pilot_manifest.csv', online['prior_manifest_coverage']['pilot_manifest']['sha256'])
    require(development.official_split.eq('official_train').all()
            and development.split.value_counts().to_dict() == {
                role: row['images'] for role, row in summary['development_v1']['partitions'].items()}
            and development.groupby('subject_id').split.nunique().eq(1).all()
            and confirmation.official_split.eq('official_validation').all() and confirmation.split.eq('holdout').all()
            and set(development.subject_id).isdisjoint(confirmation.subject_id),
            'DeepDRiD development and historical confirmation roles or patient groups differ.')
    for part in (development, confirmation):
        verify_photos(zip(part.image_path, part.sha256, strict=True))

    witness_path = ROOT / 'artifacts/deepdrid_online_readiness_v1/witness.json'
    witness = json.loads(bound_bytes(witness_path, online['witness_sha256']))
    photos = pd.DataFrame(witness['online'])
    included = photos.loc[photos.eligible]
    counts = online['population']
    require(len(photos) == counts['images'] and photos.patient_id.nunique() == counts['patients']
            and photos.groupby('patient_id').eligible.nunique().eq(1).all()
            and len(included) == counts['eligible_images']
            and included.patient_id.nunique() == counts['eligible_patients'],
            'DeepDRiD Set-C whole-patient eligibility differs from the audited mask.')
    verify_photos((author_root / row.path, row.sha256) for row in photos.itertuples())
    return {'development_photos_verified': len(development), 'confirmation_photos_verified': len(confirmation),
            'online_photos_verified': len(photos), 'eligible_online_photos': len(included),
            'eligible_online_patients': counts['eligible_patients']}


def verify_idrid(root):
    audit = report('idrid_data_v1/data_audit.json')
    review = report('idrid_review_v1/data_review.json')
    original = manifest(ROOT / 'data/manifests/idrid_v1/external_manifest.csv',
                        audit['private_output_sha256']['external_manifest.csv'])
    reviewed = manifest(ROOT / 'artifacts/idrid_review_v1/external_population.csv',
                        review['private_output_sha256']['external_population.csv'])
    validate_external(reviewed)
    counts = {name: int(reviewed['in_' + name].sum()) for name in POPULATIONS}
    require(reviewed[original.columns].equals(original) and counts == {
        name: row['images'] for name, row in review['summary']['populations'].items()},
        'IDRiD external observation membership or source labels differ.')
    fresh = parse_idrid_labels(root, require_release_counts=True)
    require(all(fresh.loc[fresh.official_split.eq(split), 'source_annotation_sha256'].eq(row['annotation_sha256']).all()
                for split, row in audit['official_partitions'].items())
            and all(sha256_file(root / name) == digest for name, digest in audit['license_file_sha256'].items()),
            'IDRiD grading annotation or license bytes differ.')
    match_source(fresh, original, ['dr_grade', 'dme_risk', 'official_split'])
    return {'external_photos_verified': len(fresh), 'frozen_populations': counts, 'outcomes_consumed': True}


def verify_eyeq():
    audit = report('eyeq_data_v1/data_audit.json')
    folder = ROOT / 'artifacts/eyeq_data_v1'
    source = verify_source_files(folder / 'source')
    frame = load_annotations(folder / 'source')
    require(annotation_profile(frame) == audit['annotation_profile'], 'EyeQ pinned annotation aggregates differ.')
    bound_bytes(folder / 'external_annotations.csv', audit['private_manifest_sha256'])
    bound_bytes(folder / 'source_group_lookup.csv', audit['private_group_lookup_sha256'])
    return {'author_metadata_files_verified': len(source), 'annotation_rows_verified': len(frame),
            'original_photo_bytes_verified': False, 'ready_for_model_evaluation': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--aptos-root', type=Path, default=default_aptos_root())
    parser.add_argument('--idrid-root', type=Path, default=default_idrid_root())
    parser.add_argument('--deepdrid-author-root', type=Path, default=ROOT.parent / 'DeepDRiD-master')
    parser.add_argument('--include-eyeq', action='store_true', help='Also verify the optional EyeQ metadata import.')
    args = parser.parse_args()

    entries = {entry.dataset_id: entry for entry in dataset_catalog(ROOT / 'reports').datasets}
    checks = {'aptos_2019': lambda: verify_aptos(args.aptos_root),
              'deepdrid': lambda: verify_deepdrid(args.deepdrid_author_root),
              'idrid': lambda: verify_idrid(args.idrid_root)}
    if args.include_eyeq:
        checks['eyeq'] = verify_eyeq
    for dataset, verify in checks.items():
        try:
            require(entries[dataset].status != 'unavailable', 'Required aggregate audit evidence is unavailable.')
            result = verify()
        except (KeyError, OSError, ValueError) as exc:
            diagnostic = str(exc) if isinstance(exc, VerificationError) else type(exc).__name__
            parser.exit(1, f'{dataset} verification failed ({diagnostic})\n')
        print(json.dumps({'dataset': dataset, **result}, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
