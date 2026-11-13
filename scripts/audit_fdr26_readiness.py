"""Independently reconstruct FDR26 readiness from the publisher archive."""

from collections import Counter, defaultdict
import csv
import hashlib
from io import BytesIO
from itertools import combinations
import json
from pathlib import Path
import re
import subprocess
from zipfile import ZipFile

import numpy as np
from openpyxl import load_workbook
from PIL import Image, ImageChops, ImageOps
from scipy.fft import dctn


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = 'fdr26_readiness_v1'
RAW = ROOT / 'data/raw/fdr26_v3'
REPORTS = ROOT / 'reports' / EXPERIMENT
WITNESS = ROOT / 'artifacts' / EXPERIMENT / 'witness.json'
REFERENCES = 'configs/idrid_data_v1.json'
SOURCE_FILES = {
    'docs/research/FDR26_READINESS_V1_PROTOCOL.md', 'scripts/prepare_fdr26.py',
    'scripts/audit_fdr26_readiness.py', 'scripts/prepare_idrid.py', REFERENCES,
    'qsart/data/aptos.py', 'qsart/data/idrid.py', 'qsart/data/manifests.py',
    'qsart/preprocessing/fundus.py',
}
RECORD_ID = 23101200
RECORD_DOI = '10.5281/zenodo.23101200'
FILES = {
    'archive': ('Fundus_FDR26.zip', 1_681_871_182, 'be97fb330cb06cadf3926de2ff8be7c9'),
    'workbook': ('metadata_classification_DR.xlsx', 129_085, '128b725a17177d38e419993c423ad4b9'),
}
GRADE_LABELS = ('DR0 - Pas de RD', 'DR1 - Minime', 'DR2 - Moderee', 'DR3 - Severe', 'DR4 - Proliferante')
JOINED_ID = re.compile(r'[A-Z0-9]{8}[0-9]{2}')
IMAGE_MEMBER = re.compile(r'images/([A-Z0-9]{8}[0-9]{2}(?:\(2\))?)\.jpg')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_json(path):
    return json.loads(path.read_bytes())


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def verify_sources(report):
    commit, bindings = report['source_commit'], report['source_sha256']
    require(re.fullmatch(r'[0-9a-f]{40}', commit) and bindings.keys() == SOURCE_FILES,
            'Execution commit or source file list differs')
    for name, expected in bindings.items():
        raw = (ROOT / name).read_bytes()
        git_bytes = subprocess.check_output(['git', 'show', f'{commit}:{name}'], cwd=ROOT)
        require(raw == git_bytes and hashlib.sha256(raw).hexdigest() == expected,
                f'Locked source differs: {name}')


def verify_inputs(report):
    record_path = RAW / 'record.json'
    record_sha = sha256(record_path)
    record = load_json(record_path)
    metadata = record['metadata']
    require(record['id'] == RECORD_ID and record['doi'] == RECORD_DOI
            and metadata['access_right'] == 'open' and metadata['license']['id'] == 'cc-by-4.0',
            'Zenodo record identity or license differs')
    record_files = {row['key']: row for row in record['files']}
    hashes = {'record': record_sha, 'reference_config': sha256(ROOT / REFERENCES)}
    paths = {}
    for kind, (name, size, checksum) in FILES.items():
        path = RAW / name
        row = record_files[name]
        require(row['size'] == size == path.stat().st_size and row['checksum'] == f'md5:{checksum}',
                'Publisher file metadata or local byte count differs')
        md5 = hashlib.md5(usedforsecurity=False)
        sha = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                md5.update(chunk)
                sha.update(chunk)
        require(md5.hexdigest() == checksum, 'Local file differs from the publisher checksum')
        paths[kind] = path
        hashes[kind] = sha.hexdigest()
    require(hashes == report['input_sha256'], 'Public input bindings differ')
    return paths


def read_labels(path):
    workbook = load_workbook(path, read_only=True, data_only=False)
    try:
        require(workbook.sheetnames == ['Toutes'], 'Unexpected FDR26 worksheet')
        rows = workbook['Toutes'].values
        require(next(rows) == ('image_id', 'classe', 'grade_DR', 'sexe', 'age', 'type_diabete'),
                'Unexpected FDR26 workbook columns')
        labels = {}
        prefixes = defaultdict(list)
        for identity, grade, grade_label, *metadata in rows:
            require(isinstance(identity, str) and JOINED_ID.fullmatch(identity)
                    and identity not in labels and type(grade) is int and grade in range(5)
                    and grade_label == GRADE_LABELS[grade], 'Invalid source image key or ALFEDIAM grade')
            labels[identity] = grade
            prefixes[identity[:8]].append((int(identity[8:]), tuple(metadata)))
        require(len(labels) == len({key.casefold() for key in labels}) == 3000,
                'Expected 3,000 unique source image keys')
        profile = {
            'prefixes': len(prefixes),
            'multi_image_prefixes': sum(len(views) > 1 for views in prefixes.values()),
            'contiguous_numbered_prefixes': sum(
                sorted(view for view, _ in views) == list(range(1, len(views) + 1)) for views in prefixes.values()
            ),
            'metadata_consistent_prefixes': sum(len({values for _, values in views}) == 1 for views in prefixes.values()),
        }
        return labels, profile
    finally:
        workbook.close()


def independent_phash(image):
    rgb = ImageOps.exif_transpose(image).convert('RGB')
    red, green, blue = rgb.split()
    channel_max = ImageChops.lighter(ImageChops.lighter(red, green), blue)
    foreground = channel_max.point([0] * 11 + [255] * 245)
    lit_pixels = foreground.histogram()[255]
    if lit_pixels >= 0.01 * (rgb.width * rgb.height):
        rgb = rgb.crop(foreground.getbbox())
    left, top = rgb.width // 8, rgb.height // 8
    right, bottom = rgb.width * 7 // 8, rgb.height * 7 // 8
    center = rgb.crop((left, top, right, bottom)).getchannel('G')
    pixels = np.asarray(center.resize((32, 32), Image.Resampling.BICUBIC), dtype=float)
    frequencies = dctn(pixels, type=2)[:8, :8].ravel()[1:]
    bits = 0
    for flag in frequencies > np.median(frequencies):
        bits = (bits << 1) | int(flag)
    return f'{bits:016x}'


def read_images(path, labels):
    rows = []
    with ZipFile(path) as bundle:
        names = bundle.namelist()
        require(names.count('images/') == 1 and bundle.getinfo('images/').is_dir(), 'Archive root differs')
        member_to_id = {}
        for member in names:
            if member == 'images/':
                continue
            match = IMAGE_MEMBER.fullmatch(member)
            require(match and member not in member_to_id, 'Invalid or duplicate archive image member')
            member_to_id[member] = match[1]
        image_ids = set(member_to_id.values())
        unmatched_workbook = sorted(labels.keys() - image_ids)
        unmatched_images = image_ids - labels.keys()
        require(len(image_ids) == len(member_to_id) == 3000
                and len(unmatched_workbook) == len(unmatched_images) == 1,
                'Expected the observed 2,999 exact archive and workbook matches')
        orphan = next(iter(unmatched_images))
        require(orphan.endswith('(2)') and orphan[:-3] in labels.keys() & image_ids,
                'The unmatched archive name must retain its literal suffix')
        for member, identity in sorted(member_to_id.items()):
            payload = bundle.read(member)
            with Image.open(BytesIO(payload)) as image:
                require(image.format == 'JPEG' and image.mode == 'RGB' and getattr(image, 'n_frames', 1) == 1,
                        'Archived photo must be single-frame RGB JPEG')
                image.load()
                rows.append({'member': member, 'image_id': identity, 'dr_grade': labels.get(identity),
                             'format': image.format, 'width': image.width, 'height': image.height,
                             'sha256': hashlib.sha256(payload).hexdigest(), 'phash': independent_phash(image)})
    return rows, unmatched_workbook


def reference_rows():
    config = load_json(ROOT / REFERENCES)
    require(config['near_duplicate_distance'] == 2 and len(config['references']) == 4,
            'Unexpected reference screen definition')
    references = {}
    counts = Counter()
    for item in config['references']:
        path = (ROOT / item['path']).resolve()
        require(path.is_relative_to(ROOT) and sha256(path) == item['sha256'], 'Reference file digest differs')
        dataset = item['dataset']
        with path.open(encoding='utf-8', newline='') as stream:
            rows = list(csv.DictReader(stream))
        require(len(rows) == item['rows'], 'Reference file row count differs')
        counts[dataset] += len(rows)
        for row in rows:
            identity, sha, phash = row['image_id'], row['sha256'], row['phash']
            require(row['dataset_source'] == dataset and identity and re.fullmatch(r'[0-9a-f]{64}', sha)
                    and re.fullmatch(r'[0-9a-f]{16}', phash), 'Invalid stored reference fingerprint')
            evidence = sha, int(phash, 16)
            require(references.setdefault((dataset, identity), evidence) == evidence,
                    'Conflicting reference image fingerprints')
    require(counts == {'APTOS_2019': 3662, 'DeepDRiD_regular_fundus': 1524} and len(references) == 5186,
            'Reference cohort coverage differs')
    return references, counts


def screen_overlaps(rows, references, reference_counts):
    records = {row['image_id']: row for row in rows}
    fingerprints = {identity: int(row['phash'], 16) for identity, row in records.items()}
    neighbors = defaultdict(set)
    internal = []
    for first, second in combinations(records, 2):
        distance = (fingerprints[first] ^ fingerprints[second]).bit_count()
        exact = records[first]['sha256'] == records[second]['sha256']
        if exact or distance <= 2:
            neighbors[first].add(second)
            neighbors[second].add(first)
            pair_grades = {records[first]['dr_grade'], records[second]['dr_grade']} - {None}
            internal.append({'first_image_id': first, 'second_image_id': second, 'byte_exact': exact,
                             'phash_distance': distance, 'label_conflict': len(pair_grades) > 1})

    cross = []
    for identity, row in records.items():
        fingerprint, digest = fingerprints[identity], row['sha256']
        for (dataset, reference_id), (sha, phash) in references.items():
            distance = (fingerprint ^ phash).bit_count()
            exact = digest == sha
            if exact or distance <= 2:
                cross.append({'external_image_id': identity, 'reference_dataset': dataset,
                              'reference_image_id': reference_id, 'byte_exact': exact, 'phash_distance': distance})

    cross_images = {row['external_image_id'] for row in cross}
    seen = set()
    components = conflicting = unmatched_component_excluded = 0
    for identity in records:
        if identity in seen:
            continue
        component, pending = {identity}, [identity]
        while pending:
            new = neighbors[pending.pop()] - component
            component.update(new)
            pending.extend(new)
        seen.update(component)
        members = sorted(component)
        digest = hashlib.sha256('|'.join(members).encode()).hexdigest()[:24]
        grades = {records[member]['dr_grade'] for member in members}
        unmatched = None in grades
        conflict = len(grades - {None}) > 1
        cross_hit = not component.isdisjoint(cross_images)
        eligible = not (unmatched or conflict or cross_hit)
        components += 1
        conflicting += conflict
        if unmatched:
            unmatched_component_excluded += len(members)
        for member in members:
            records[member].update(component=f'aptos_splitgroup_{digest}', label_conflict=conflict,
                                   cross_candidate=cross_hit, eligible=eligible,
                                   representative=eligible and member == members[0])

    excluded = {identity for identity, row in records.items() if not row['eligible']}
    prefixes = {identity: identity[:8] for identity, row in records.items() if row['dr_grade'] is not None}
    prefix_members = defaultdict(set)
    for identity, prefix in prefixes.items():
        prefix_members[prefix].add(identity)
    prefix_excluded, pending = set(excluded), list(excluded)
    while pending:
        identity = pending.pop()
        linked = neighbors[identity] | prefix_members.pop(prefixes.get(identity), set())
        new = linked - prefix_excluded
        prefix_excluded.update(new)
        pending.extend(new)

    pairs_by_dataset = Counter(row['reference_dataset'] for row in cross)
    summary = {
        'internal_pairs': len(internal), 'internal_exact_pairs': sum(row['byte_exact'] for row in internal),
        'components': components, 'conflicting_components': conflicting, 'cross_pairs': len(cross),
        'cross_exact_pairs': sum(row['byte_exact'] for row in cross), 'cross_images': len(cross_images),
        'excluded_images': len(excluded), 'eligible_images': len(rows) - len(excluded),
        'unmatched_component_excluded_images': unmatched_component_excluded,
        'representatives': sum(row['representative'] for row in rows), 'reference_rows': sum(reference_counts.values()),
        'reference_pairs_by_dataset': {dataset: pairs_by_dataset[dataset] for dataset in reference_counts},
        'prefix_sensitivity_excluded_images': len(prefix_excluded),
        'prefix_sensitivity_eligible_images': len(rows) - len(prefix_excluded),
        'prefix_sensitivity_representatives': sum(
            row['representative'] and identity not in prefix_excluded for identity, row in records.items()
        ),
    }
    return internal, cross, summary


def canonical_pairs(rows, keys):
    return sorted(rows, key=lambda row: tuple(row[key] for key in keys))


def run():
    evaluation = REPORTS / 'evaluation.json'
    target = REPORTS / 'independent_audit.json'
    require(not target.exists(), 'Preserve the existing audit receipt')
    report, witness = load_json(evaluation), load_json(WITNESS)
    require(report['experiment_id'] == EXPERIMENT and report['witness_sha256'] == sha256(WITNESS),
            'Public report identity or private witness binding differs')
    require(report['dataset'] == 'FDR26' and report['partition'] == 'external_candidate_unscored'
            and report['scope'] == 'external_grading_data_readiness' and report['grade_standard'] == 'ALFEDIAM'
            and report['patient_linkage_available'] is False and report['model_inference_performed'] is False,
            'External readiness report scope differs')
    require(witness.keys() == {'rows', 'unmatched_workbook_ids', 'internal_candidates', 'cross_candidates'},
            'Unexpected witness fields')
    verify_sources(report)
    paths = verify_inputs(report)
    references, reference_counts = reference_rows()
    labels, prefix_profile = read_labels(paths['workbook'])
    rows, unmatched_workbook = read_images(paths['archive'], labels)
    require(witness['unmatched_workbook_ids'] == unmatched_workbook, 'Unmatched workbook witness differs')
    joined = [row['dr_grade'] for row in rows if row['dr_grade'] is not None]
    inventory = {
        'images': len(rows), 'archive_images': len(rows), 'workbook_rows': len(labels), 'joined_images': len(joined),
        'unmatched_archive_images': len(rows) - len(joined), 'unmatched_workbook_rows': len(unmatched_workbook),
        'grade_counts': dict(Counter(str(value) for value in labels.values())),
        'joined_grade_counts': dict(Counter(str(value) for value in joined)),
        'formats': dict(Counter(row['format'] for row in rows)), 'filename_prefix_profile': prefix_profile,
    }
    require(all(report['inventory'][key] == value for key, value in inventory.items()), 'Image inventory differs')
    internal, cross, overlap = screen_overlaps(rows, references, reference_counts)
    require(all(report['overlap'][key] == value for key, value in overlap.items()), 'Overlap summary differs')
    require(sorted(witness['rows'], key=lambda row: row['image_id']) == sorted(rows, key=lambda row: row['image_id']),
            'Image witness differs from independent source reconstruction')
    for key, expected, keys in (
        ('internal_candidates', internal, ('first_image_id', 'second_image_id')),
        ('cross_candidates', cross, ('external_image_id', 'reference_dataset', 'reference_image_id')),
    ):
        require(canonical_pairs(expected, keys) == canonical_pairs(witness[key], keys), f'{key} differs')

    receipt = {
        'audit_status': 'passed', 'experiment_id': EXPERIMENT, 'source_commit': report['source_commit'],
        'evaluation_sha256': sha256(evaluation), 'auditor_sha256': sha256(Path(__file__)),
        'private_witness_sha256': sha256(WITNESS), 'input_sha256': report['input_sha256'],
        'images_reconstructed': len(rows), 'exact_joins_reconstructed': len(joined),
        'unmatched_archive_images': inventory['unmatched_archive_images'],
        'unmatched_workbook_rows': inventory['unmatched_workbook_rows'],
        'reference_manifest_rows': sum(reference_counts.values()),
        'internal_candidate_pairs': len(internal), 'cross_candidate_pairs': len(cross),
        'components_reconstructed': overlap['components'], 'model_inference_performed': False,
    }
    with target.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(receipt, indent=2, allow_nan=False) + '\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    run()
