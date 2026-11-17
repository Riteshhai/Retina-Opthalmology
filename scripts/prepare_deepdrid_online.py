"""Authenticate and screen the DeepDRiD online challenge before model inference."""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
from io import BytesIO
from itertools import combinations, product
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PIL import Image

from qsart.data.manifests import QUALITY_CODES, retinal_phash, sha256_file


ROOT = Path(__file__).resolve().parents[1]
STUDY = 'deepdrid_online_readiness_v1'
AUTHOR_COMMIT = '56d8af71319803150d0e7b90533e0918c7bf85ee'
AUTHOR_TREE = '8340aa4fa31819be0e6da05bb842804f40284bdf'
REGULAR = PurePosixPath('regular_fundus_images')
ONLINE = REGULAR / 'Online-Challenge1&2-Evaluation'
SOURCE_IMAGES = {ONLINE: 400, REGULAR / 'regular-fundus-training': 1200,
                 REGULAR / 'regular-fundus-validation': 400}
METADATA = ['LICENSE', str(ONLINE / 'Readme.docx'), str(ONLINE / 'Challenge1_labels.xlsx'),
            str(ONLINE / 'Challenge2_labels.xlsx')]
PRIOR_MANIFESTS = ('development_v1', 'confirmation_v1', 'pilot_manifest')
SOURCES = ('docs/research/DEEPDRID_ONLINE_READINESS_V1_PROTOCOL.md',
           'scripts/prepare_deepdrid_online.py', 'scripts/audit_deepdrid_online.py',
           'qsart/data/manifests.py', 'qsart/preprocessing/fundus.py')


def read_workbook(path, columns, domains):
    from openpyxl import load_workbook

    workbook = load_workbook(path, read_only=True, data_only=False)
    try:
        if workbook.sheetnames != ['Sheet1']:
            raise ValueError('Expected exactly one official label sheet')
        rows = workbook['Sheet1'].values
        if next(rows, None) != columns:
            raise ValueError('Unexpected official workbook header')
        labels = {}
        for identity, *values in rows:
            if (not isinstance(identity, str) or not re.fullmatch(r'[0-9]+_[lr][12]', identity)
                    or identity in labels or len(values) != len(domains)
                    or any(type(value) is not int or value not in domain
                           for value, domain in zip(values, domains, strict=True))):
                raise ValueError('Invalid or duplicate released workbook label')
            labels[identity] = tuple(values)
        return labels
    finally:
        workbook.close()


def join_online(online, grading, quality, *, expected_patients=100):
    keys = {row['image_id'] for row in online}
    if len(online) != len(keys) or keys != grading.keys() or keys != quality.keys():
        raise ValueError('The workbooks and online photos must join exactly by unique original stem')
    views = defaultdict(dict)
    for row in online:
        grade, = grading[row['image_id']]
        overall, artifact, clarity, field = quality[row['image_id']]
        eye, view = row['eye'], row['view']
        if row['image_id'] != f"{row['patient_id']}_{eye}{view}":
            raise ValueError('Online folder key, filename and view disagree')
        views[row['patient_id']][eye, view] = grade
        row.update(dr_grade=grade, overall_quality=overall, artifact=artifact,
                   clarity=clarity, field_definition=field)
    view_keys = set(product('lr', (1, 2)))
    if len(views) != expected_patients or any(patient.keys() != view_keys for patient in views.values()):
        raise ValueError('Expected four distinct views per online patient')
    if any(patient[eye, 1] != patient[eye, 2] for patient in views.values() for eye in 'lr'):
        raise ValueError('Released DR grades disagree within an eye')
    return views


def screen_overlap(online, reference):
    def extended(rows):
        return [(row, int(row['phash'], 16)) for row in sorted(rows, key=lambda row: row['path'])]

    online_extended, reference_extended = extended(online), extended(reference)
    shared_keys = {row['patient_id'] for row in online} & {row['patient_id'] for row in reference}
    excluded = set(shared_keys)
    internal_pairs, reference_pairs = [], []
    linked = set()
    counts = dict.fromkeys(('internal_exact_pairs', 'internal_cross_patient_pairs',
                          'internal_same_patient_same_eye_pairs', 'internal_same_patient_opposite_eye_pairs',
                          'internal_same_patient_opposite_eye_exact_pairs', 'reference_exact_pairs'), 0)
    for (a, a_hash), (b, b_hash) in combinations(online_extended, 2):
        exact = a['sha256'] == b['sha256']
        if not exact and (a_hash ^ b_hash).bit_count() > 2:
            continue
        internal_pairs.append({'first': a['path'], 'second': b['path']})
        counts['internal_exact_pairs'] += exact
        if a['patient_id'] != b['patient_id']:
            excluded.update((a['patient_id'], b['patient_id']))
            counts['internal_cross_patient_pairs'] += 1
        elif a['eye'] == b['eye']:
            counts['internal_same_patient_same_eye_pairs'] += 1
        else:
            counts['internal_same_patient_opposite_eye_pairs'] += 1
            counts['internal_same_patient_opposite_eye_exact_pairs'] += exact
            if exact:
                excluded.add(a['patient_id'])
    for (a, a_hash), (b, b_hash) in product(online_extended, reference_extended):
        exact = a['sha256'] == b['sha256']
        if exact or (a_hash ^ b_hash).bit_count() <= 2:
            reference_pairs.append({'first': a['path'], 'second': b['path']})
            counts['reference_exact_pairs'] += exact
            linked.add(a['patient_id'])
    excluded.update(linked)
    for row in online:
        row['eligible'] = row['patient_id'] not in excluded
    return internal_pairs, reference_pairs, sorted(excluded), {
        'reference_images': dict(Counter(row['split'] for row in reference)),
        'shared_source_patient_keys': len(shared_keys), 'internal_pairs': len(internal_pairs),
        'reference_pairs': len(reference_pairs), 'reference_patients': len(linked), **counts,
    }


def git_bytes(git_root, *args):
    return subprocess.check_output(['git', '-C', str(git_root), *args])


def source_tree(git_root):
    tree = git_bytes(git_root, 'rev-parse', f'{AUTHOR_COMMIT}^{{tree}}').decode().strip()
    if tree != AUTHOR_TREE:
        raise ValueError('Pinned author Git tree differs')
    paths = git_bytes(git_root, 'ls-tree', '-r', '-z', AUTHOR_COMMIT, '--', str(REGULAR), 'LICENSE')
    entries = {}
    for row in paths.decode('utf-8').split('\0')[:-1]:
        attrs, path = row.split('\t', 1)
        mode, kind, sha = attrs.split()
        if kind != 'blob' or mode != '100644' or path in entries:
            raise ValueError('Unexpected entry type in pinned author tree')
        entries[path] = sha
    return entries


def authenticated_bytes(root, name, entries):
    path = root / name
    if not path.resolve().is_relative_to(root):
        raise ValueError('Source asset is outside the author checkout')
    payload = path.read_bytes()
    object_id = hashlib.sha1(f'blob {len(payload)}\0'.encode() + payload, usedforsecurity=False).hexdigest()
    if object_id != entries[name]:
        raise ValueError('Local asset bytes differ from the pinned author Git tree')
    return payload


def inspect_images(root, entries):
    online, reference = [], []
    for folder, count in SOURCE_IMAGES.items():
        prefix = f'{folder}/Images/'
        expected = sorted(name for name in entries if name.startswith(prefix))
        local = sorted(path.relative_to(root).as_posix() for path in (root / prefix).rglob('*') if path.is_file())
        if len(expected) != count or expected != local:
            raise ValueError('Local and pinned author photograph inventories differ')
        for name in expected:
            path = PurePosixPath(name)
            identity, patient = path.stem, path.parent.name
            if len(path.parts) != 5 or path.suffix != '.jpg' or identity.rsplit('_', 1)[0] != patient:
                raise ValueError('Unexpected author photograph path or patient key')
            payload = authenticated_bytes(root, name, entries)
            with Image.open(BytesIO(payload)) as image:
                if image.format != 'JPEG' or image.mode != 'RGB' or getattr(image, 'n_frames', 1) != 1:
                    raise ValueError('Expected a single-frame RGB JPEG in the author photographs')
                image.load()
                row = {'path': name, 'image_id': identity, 'patient_id': patient,
                       'width': image.width, 'height': image.height,
                       'sha256': hashlib.sha256(payload).hexdigest(), 'phash': retinal_phash(image)}
            if folder == ONLINE:
                match = re.fullmatch(r'[0-9]+_([lr])([12])', identity)
                if match is None:
                    raise ValueError('Expected a conventional eye and view suffix in Set-C')
                online.append(dict(row, eye=match[1], view=int(match[2])))
            else:
                reference.append(dict(row, split=folder.name.rsplit('-', 1)[1]))
        print(f'Authenticated and fingerprinted {count} {folder.name} photographs', flush=True)
    return online, reference


def reconcile_prior_manifests(reference):
    authentic = {row['image_id']: row['sha256'] for row in reference}
    if len(authentic) != len(reference):
        raise ValueError('Prior source regular photographs have ambiguous stems')
    coverage = {}
    for name in PRIOR_MANIFESTS:
        path = ROOT / 'data/processed/deepdrid' / (name + '.csv')
        with path.open(encoding='utf-8-sig', newline='') as stream:
            rows = list(csv.DictReader(stream))
        ids = {row['image_id'] for row in rows}
        if not rows or len(ids) != len(rows) or any(authentic.get(row['image_id']) != row['sha256'] for row in rows):
            raise ValueError('A prior DeepDRiD manifest disagrees with authenticated prior split photographs')
        coverage[name] = {'images': len(rows), 'sha256': sha256_file(path)}
    return coverage


def label_counts(online, patients):
    counts = {key + '_counts': dict(sorted(Counter(str(row[key]) for row in online).items()))
              for key in ('overall_quality', 'artifact', 'clarity', 'field_definition')}
    counts['image_grade_counts'] = dict(sorted(Counter(str(row['dr_grade']) for row in online).items()))
    counts['eye_grade_counts'] = dict(sorted(Counter(str(patient[eye, 1])
                                                     for patient in patients.values() for eye in 'lr').items()))
    counts['patient_grade_counts'] = dict(sorted(Counter(str(max(patient.values()))
                                                         for patient in patients.values()).items()))
    return counts


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as out:
        out.write(json.dumps(payload, indent=2, allow_nan=False) + '\n')


def execute(root, source_git, source_commit):
    private = ROOT / 'artifacts' / STUDY / 'witness.json'
    public = ROOT / 'reports' / STUDY / 'evaluation.json'
    if private.exists() or public.exists():
        raise ValueError('Readiness outputs already exist; preserve them before considering a separate run')
    if (git_bytes(ROOT, 'rev-parse', 'HEAD').decode().strip() != source_commit
            or git_bytes(ROOT, 'status', '--porcelain')):
        raise ValueError('Use the declared clean QSART source commit')
    sources = {}
    for name in SOURCES:
        data = (ROOT / name).read_bytes()
        if data != git_bytes(ROOT, 'show', f'{source_commit}:{name}'):
            raise ValueError('Execution source differs from the pinned QSART Git bytes')
        sources[name] = hashlib.sha256(data).hexdigest()
    entries = source_tree(source_git)
    metadata = {name: hashlib.sha256(authenticated_bytes(root, name, entries)).hexdigest() for name in METADATA}
    grading = read_workbook(root / ONLINE / 'Challenge1_labels.xlsx', ('image_id', 'DR_Levels'), (range(5),))
    quality = read_workbook(root / ONLINE / 'Challenge2_labels.xlsx',
                            ('image_id', 'Overall quality', 'Artifact', 'Clarity', 'Field definition'),
                            (range(2), QUALITY_CODES['artifact'], QUALITY_CODES['clarity'],
                             QUALITY_CODES['field_definition']))
    online, reference = inspect_images(root, entries)
    patients = join_online(online, grading, quality)
    prior_manifest_coverage = reconcile_prior_manifests(reference)
    internal_pairs, reference_pairs, excluded, overlap = screen_overlap(online, reference)
    eligible = sum(row['eligible'] for row in online)
    if eligible != 4 * (len(patients) - len(excluded)):
        raise ValueError('Overlap exclusion did not preserve complete patient groups')
    write_json(private, {'online': online, 'reference': reference, 'internal_pairs': internal_pairs,
                         'reference_pairs': reference_pairs, 'excluded_patients': excluded})
    report = {
        'experiment_id': STUDY, 'dataset': 'DeepDRiD', 'partition': 'online_challenge_unscored',
        'scope': 'same_release_held_out_patient_data_readiness', 'source_commit': source_commit,
        'source_sha256': sources, 'source_release': {'commit': AUTHOR_COMMIT, 'tree': AUTHOR_TREE,
                                                     'license': 'CC-BY-SA-4.0'},
        'metadata_sha256': metadata, 'prior_manifest_coverage': prior_manifest_coverage,
        'population': {'images': len(online), 'patients': len(patients), 'eyes': 2 * len(patients),
                       'eligible_images': eligible, 'eligible_patients': eligible // 4,
                       'excluded_patients': len(excluded)},
        'labels': label_counts(online, patients), 'overlap': overlap,
        'model_inference_performed': False, 'witness_sha256': sha256_file(private),
    }
    write_json(public, report)
    print(json.dumps({'population': report['population'], 'overlap': overlap}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--source-git', type=Path, required=True)
    parser.add_argument('--source-commit', required=True)
    args = parser.parse_args()
    execute(args.root.resolve(), args.source_git.resolve(), args.source_commit)


if __name__ == '__main__':
    main()
