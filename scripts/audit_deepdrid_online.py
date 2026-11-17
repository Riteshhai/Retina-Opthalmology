"""Independently verify the unscored DeepDRiD regular online challenge cohort."""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
from io import BytesIO
from itertools import combinations
import json
from pathlib import Path, PurePosixPath
import re
import subprocess

import numpy as np
from openpyxl import load_workbook
from PIL import Image, ImageChops, ImageOps
from scipy.fft import dctn


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = 'deepdrid_online_readiness_v1'
PUBLIC = ROOT / 'reports' / EXPERIMENT
WITNESS = ROOT / 'artifacts' / EXPERIMENT / 'witness.json'
RELEASE = {
    'commit': '56d8af71319803150d0e7b90533e0918c7bf85ee',
    'tree': '8340aa4fa31819be0e6da05bb842804f40284bdf',
    'license': 'CC-BY-SA-4.0',
}
REGULAR = 'regular_fundus_images'
ONLINE = f'{REGULAR}/Online-Challenge1&2-Evaluation'
FOLDERS = {ONLINE: 'online', **{f'{REGULAR}/regular-fundus-{split}': split for split in ('training', 'validation')}}
GRADING = f'{ONLINE}/Challenge1_labels.xlsx'
QUALITY = f'{ONLINE}/Challenge2_labels.xlsx'
METADATA = ('LICENSE', GRADING, QUALITY, f'{ONLINE}/Readme.docx')
VIEWS = {('l', 1), ('l', 2), ('r', 1), ('r', 2)}
PRIOR_COUNTS = {'development_v1': 1184, 'confirmation_v1': 340, 'pilot_manifest': 260}
SOURCE_FILES = {
    'docs/research/DEEPDRID_ONLINE_READINESS_V1_PROTOCOL.md', 'scripts/prepare_deepdrid_online.py',
    'scripts/audit_deepdrid_online.py', 'qsart/data/manifests.py', 'qsart/preprocessing/fundus.py',
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(path.read_bytes())


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def release_blobs(source_git):
    git = ['git', '--no-lazy-fetch', f'--git-dir={source_git.resolve()}']
    commit = RELEASE['commit']
    tree = subprocess.check_output([*git, 'rev-parse', f'{commit}^{{tree}}']).decode().strip()
    require(tree == RELEASE['tree'], 'Pinned author tree differs')
    listing = subprocess.check_output([*git, 'ls-tree', '-r', '-z', commit, '--', REGULAR, 'LICENSE'])
    blobs = {}
    for entry in listing.split(b'\x00'):
        if entry:
            info, raw_name = entry.split(b'\t', 1)
            mode, kind, oid = info.decode().split()
            if kind == 'blob' and mode in ('100644', '100755'):
                blobs[raw_name.decode('utf-8')] = oid
    require(all(name in blobs for name in METADATA), 'Required release metadata is absent from the author tree')
    return blobs


def authenticated(root, name, blobs):
    path = root / name
    require(path.resolve().is_relative_to(root), 'Local release path leaves the source root')
    data = path.read_bytes()
    prefix = f'blob {len(data)}\0'.encode()
    git_hash = hashlib.sha1(prefix, usedforsecurity=False)
    git_hash.update(data)
    require(git_hash.hexdigest() == blobs.get(name), 'Local file bytes differ from their pinned author blob')
    return data


def read_labels(payload, names, allowed):
    book = load_workbook(BytesIO(payload), read_only=True, data_only=False)
    try:
        require(book.sheetnames == ['Sheet1'], 'Unexpected online challenge workbook sheets')
        rows = book['Sheet1'].values
        require(next(rows) == names, 'Unexpected online challenge workbook columns')
        labels = {}
        for identity, *codes in rows:
            require(isinstance(identity, str) and re.fullmatch(r'[0-9]+_[lr][12]', identity)
                    and identity not in labels and len(codes) == len(allowed)
                    and all(type(code) is int and code in domain for code, domain in zip(codes, allowed, strict=True)),
                    'Invalid or duplicate online challenge label row')
            labels[identity] = codes
        require(len(labels) == 400, 'Expected exactly 400 online challenge labels')
        return labels
    finally:
        book.close()


def independent_phash(photo):
    image = ImageOps.exif_transpose(photo).convert('RGB')
    red, green, blue = image.split()
    maximum = ImageChops.lighter(ImageChops.lighter(red, green), blue)
    foreground = maximum.point([0] * 11 + [255] * 245)
    if foreground.histogram()[255] >= .01 * (image.width * image.height):
        image = image.crop(foreground.getbbox())
    box = (image.width // 8, image.height // 8, image.width * 7 // 8, image.height * 7 // 8)
    small = image.crop(box).getchannel('G').resize((32, 32), Image.Resampling.BICUBIC)
    frequencies = dctn(np.asarray(small, dtype=float), type=2)[:8, :8].ravel()[1:]
    bits = 0
    for bit in frequencies > np.median(frequencies):
        bits = (bits << 1) | int(bit)
    return f'{bits:016x}'


def image_names(root, blobs):
    result = {}
    for folder, split in FOLDERS.items():
        prefix = f'{folder}/Images/'
        names = sorted(name for name in blobs if name.startswith(prefix) and name.endswith('.jpg'))
        local = {path.relative_to(root).as_posix() for path in (root / prefix).rglob('*') if path.is_file()}
        require(local == set(names), 'Local JPEG membership differs from its author tree')
        result.update({name: split for name in names})
    require(Counter(result.values()) == {'online': 400, 'training': 1200, 'validation': 400},
            'Unexpected number of pinned source JPEGs')
    return result


def image_rows(root, blobs, grades, quality):
    online, reference = [], []
    for name, split in sorted(image_names(root, blobs).items()):
        path = PurePosixPath(name)
        identity, patient = path.stem, path.parent.name
        require(len(path.parts) == 5 and patient.isdecimal() and identity.partition('_')[0] == patient,
                'JPEG source folder and identity do not agree')
        payload = authenticated(root, name, blobs)
        with Image.open(BytesIO(payload)) as photo:
            require(photo.format == 'JPEG' and photo.mode == 'RGB' and getattr(photo, 'n_frames', 1) == 1,
                    'Author image must decode as a single RGB JPEG frame')
            photo.load()
            record = {'path': name, 'image_id': identity, 'patient_id': patient,
                      'width': photo.width, 'height': photo.height,
                      'sha256': hashlib.sha256(payload).hexdigest(), 'phash': independent_phash(photo)}
        if split == 'online':
            view = re.fullmatch(r'[0-9]+_([lr])([12])', identity)
            require(view and identity in grades and identity in quality, 'Missing exact online label or pose join')
            record.update(eye=view[1], view=int(view[2]), dr_grade=grades[identity][0])
            record.update(zip(('overall_quality', 'artifact', 'clarity', 'field_definition'), quality[identity], strict=True))
            online.append(record)
        else:
            record['split'] = split
            reference.append(record)
    require({row['image_id'] for row in online} == grades.keys() == quality.keys(),
            'Online workbook and JPEG identities do not join one-to-one')
    return online, reference


def code_counts(values):
    return {str(code): count for code, count in sorted(Counter(values).items())}


def prior_manifest_coverage(reference):
    hashes = {row['image_id']: row['sha256'] for row in reference}
    require(len(hashes) == len(reference) == 1600, 'Authenticated reference photo stems are not unique')
    output = {}
    for name, expected in PRIOR_COUNTS.items():
        manifest = ROOT / 'data/processed/deepdrid' / f'{name}.csv'
        observed = set()
        with manifest.open(encoding='utf-8-sig', newline='') as stream:
            for row in csv.DictReader(stream):
                identity = row['image_id']
                require(identity not in observed and identity in hashes and row['sha256'] == hashes[identity],
                        'A prior manifest contains a duplicate or unauthenticated image')
                observed.add(identity)
        require(len(observed) == expected, 'Prior manifest photo count differs')
        output[name] = {'images': len(observed), 'sha256': sha256(manifest)}
    return output


def joint_annotations(online):
    grouped, eye_grades, patient_grades = defaultdict(list), [], []
    for row in online:
        grouped[row['patient_id']].append(row)
    for views in grouped.values():
        require(len(views) == 4 and {(r['eye'], r['view']) for r in views} == VIEWS,
                'An online patient has an incomplete left/right view roster')
        eyes = []
        for side in ('l', 'r'):
            grades = {row['dr_grade'] for row in views if row['eye'] == side}
            require(len(grades) == 1, 'Same-eye grading labels differ')
            eyes.extend(grades)
        eye_grades.extend(eyes)
        patient_grades.append(max(eyes))
    require(len(grouped) == 100 and len(eye_grades) == 200, 'Unexpected native patient or eye grouping')
    return {
        'image_grade_counts': code_counts(row['dr_grade'] for row in online),
        'eye_grade_counts': code_counts(eye_grades), 'patient_grade_counts': code_counts(patient_grades),
        **{f'{label}_counts': code_counts(row[label] for row in online)
           for label in ('overall_quality', 'artifact', 'clarity', 'field_definition')},
    }


def screen(online, reference):
    fingerprints = {row['path']: int(row['phash'], 16) for row in [*online, *reference]}
    online_keys = {row['patient_id'] for row in online}
    shared = online_keys.intersection(row['patient_id'] for row in reference)
    excluded = set(shared)
    internal_pairs, reference_pairs, reference_patients = [], [], set()
    stats = Counter()
    for first, second in combinations(online, 2):
        exact = first['sha256'] == second['sha256']
        distance = (fingerprints[first['path']] ^ fingerprints[second['path']]).bit_count()
        if exact or distance <= 2:
            a, b = first['patient_id'], second['patient_id']
            internal_pairs.append({'first': first['path'], 'second': second['path']})
            stats['internal_exact_pairs'] += exact
            if a != b:
                stats['internal_cross_patient_pairs'] += 1
                excluded.update((a, b))
            elif first['eye'] == second['eye']:
                stats['internal_same_patient_same_eye_pairs'] += 1
            else:
                stats['internal_same_patient_opposite_eye_pairs'] += 1
                stats['internal_same_patient_opposite_eye_exact_pairs'] += exact
                if exact:
                    excluded.add(a)
    for first in online:
        hash_first = fingerprints[first['path']]
        for second in reference:
            exact = first['sha256'] == second['sha256']
            distance = (hash_first ^ fingerprints[second['path']]).bit_count()
            if exact or distance <= 2:
                reference_pairs.append({'first': first['path'], 'second': second['path']})
                stats['reference_exact_pairs'] += exact
                reference_patients.add(first['patient_id'])
    excluded.update(reference_patients)
    for row in online:
        row['eligible'] = row['patient_id'] not in excluded
    population = {'images': len(online), 'patients': len(online_keys), 'eyes': 2 * len(online_keys),
                  'eligible_images': sum(row['eligible'] for row in online),
                  'eligible_patients': len(online_keys - excluded), 'excluded_patients': len(excluded)}
    overlap = {
        'reference_images': dict(Counter(row['split'] for row in reference)),
        'shared_source_patient_keys': len(shared), 'internal_pairs': len(internal_pairs),
        'reference_pairs': len(reference_pairs), 'reference_patients': len(reference_patients),
        **{field: stats[field] for field in (
            'internal_exact_pairs', 'internal_cross_patient_pairs', 'internal_same_patient_same_eye_pairs',
            'internal_same_patient_opposite_eye_pairs', 'internal_same_patient_opposite_eye_exact_pairs',
            'reference_exact_pairs',
        )},
    }
    return internal_pairs, reference_pairs, sorted(excluded), population, overlap


def run(root, source_git):
    evaluation = PUBLIC / 'evaluation.json'
    target = PUBLIC / 'independent_audit.json'
    require(not target.exists(), 'Preserve the completed independent audit receipt')
    report = read_json(evaluation)
    require(report['experiment_id'] == EXPERIMENT and report['source_release'] == RELEASE
            and report['dataset'] == 'DeepDRiD' and report['partition'] == 'online_challenge_unscored'
            and report['scope'] == 'same_release_held_out_patient_data_readiness'
            and report['model_inference_performed'] is False
            and report['witness_sha256'] == sha256(WITNESS), 'Readiness report scope or source bindings differ')
    require(re.fullmatch(r'[0-9a-f]{40}', report['source_commit']), 'Invalid QSART execution source')
    commit = report['source_commit']
    require(report['source_sha256'].keys() == SOURCE_FILES, 'Unexpected QSART execution source files')
    for name, expected in report['source_sha256'].items():
        raw = (ROOT / name).read_bytes()
        committed = subprocess.check_output(['git', 'show', f'{commit}:{name}'], cwd=ROOT)
        require(raw == committed and hashlib.sha256(raw).hexdigest() == expected, 'QSART execution source bytes differ')
    root = root.resolve()
    blobs = release_blobs(source_git)
    files = {name: authenticated(root, name, blobs) for name in METADATA}
    metadata = {name: hashlib.sha256(data).hexdigest() for name, data in files.items()}
    require(report['metadata_sha256'] == metadata, 'Published metadata SHA256 values differ')
    grades = read_labels(files[GRADING], ('image_id', 'DR_Levels'), (range(5),))
    quality = read_labels(files[QUALITY], ('image_id', 'Overall quality', 'Artifact', 'Clarity', 'Field definition'),
                          ({0, 1}, {0, 1, 4, 6, 8, 10}, {1, 4, 6, 8, 10}, {1, 4, 6, 8, 10}))
    online, reference = image_rows(root, blobs, grades, quality)
    coverage = prior_manifest_coverage(reference)
    require(coverage == report['prior_manifest_coverage'], 'Published prior manifest bindings differ')
    labels = joint_annotations(online)
    internal, cross, excluded, population, overlap = screen(online, reference)
    new_witness = {'online': online, 'reference': reference, 'internal_pairs': internal,
                   'reference_pairs': cross, 'excluded_patients': excluded}
    require(new_witness == read_json(WITNESS), 'Independent row fingerprints, annotations, or candidate evidence differ')
    for field, value in (('population', population), ('labels', labels), ('overlap', overlap)):
        require(all(report[field].get(key) == item for key, item in value.items()), f'Published {field} differs')
    receipt = {
        'audit_status': 'passed', 'experiment_id': EXPERIMENT, 'source_release': RELEASE,
        'source_commit': commit, 'evaluation_sha256': sha256(evaluation), 'auditor_sha256': sha256(Path(__file__)),
        'private_witness_sha256': report['witness_sha256'], 'author_photo_blobs_verified': len(online) + len(reference),
        'author_metadata_blobs_verified': len(files), 'population': population,
        'prior_manifest_coverage': coverage,
        'internal_candidate_pairs_reconstructed': len(internal), 'reference_candidate_pairs_reconstructed': len(cross),
        'model_inference_performed': False,
    }
    with target.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(receipt, indent=2, allow_nan=False) + '\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--source-git', type=Path, required=True)
    args = parser.parse_args()
    run(args.root, args.source_git)
