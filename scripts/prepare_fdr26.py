"""Verify the official FDR26 v3 external grading inputs without scoring models."""

import argparse
from collections import Counter, defaultdict
import hashlib
from io import BytesIO
import json
from pathlib import Path
import re
import subprocess
import sys
from zipfile import ZipFile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from PIL import Image

from qsart.data.aptos import fingerprint_components
from qsart.data.idrid import cross_dataset_candidates
from qsart.data.manifests import retinal_phash
from scripts.prepare_idrid import load_references


ROOT = Path(__file__).resolve().parents[1]
SLUG = 'fdr26_readiness_v1'
RAW = ROOT / 'data/raw/fdr26_v3'
PROTOCOL = 'docs/research/FDR26_READINESS_V1_PROTOCOL.md'
REFERENCES = 'configs/idrid_data_v1.json'
SOURCES = (PROTOCOL, 'scripts/prepare_fdr26.py', 'scripts/audit_fdr26_readiness.py',
           'scripts/prepare_idrid.py', REFERENCES, 'qsart/data/aptos.py', 'qsart/data/idrid.py',
           'qsart/data/manifests.py', 'qsart/preprocessing/fundus.py')
FILES = {
    'archive': ('Fundus_FDR26.zip', 1681871182, 'be97fb330cb06cadf3926de2ff8be7c9'),
    'workbook': ('metadata_classification_DR.xlsx', 129085, '128b725a17177d38e419993c423ad4b9'),
}
GRADE_TEXT = ('DR0 - Pas de RD', 'DR1 - Minime', 'DR2 - Moderee', 'DR3 - Severe', 'DR4 - Proliferante')
ID = r'[A-Z0-9]{8}[0-9]{2}'


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def source_bindings(commit):
    def git(*args):
        return subprocess.check_output(['git', *args], cwd=ROOT)

    if git('rev-parse', 'HEAD').decode().strip() != commit or git('status', '--porcelain'):
        raise ValueError('Use a clean checkout of the locked execution source')
    result = {}
    for name in SOURCES:
        data = (ROOT / name).read_bytes()
        if data != git('show', f'{commit}:{name}'):
            raise ValueError(f'Source differs from committed bytes: {name}')
        result[name] = hashlib.sha256(data).hexdigest()
    return result


def verify_release():
    record = json.loads((RAW / 'record.json').read_bytes())
    if (record['id'] != 23101200 or record['doi'] != '10.5281/zenodo.23101200'
            or record['metadata']['access_right'] != 'open'
            or record['metadata']['license']['id'] != 'cc-by-4.0'):
        raise ValueError('Expected official FDR26 v3 Zenodo record and license')
    files = {file['key']: file for file in record['files']}
    digests = {'record': sha(RAW / 'record.json'), 'reference_config': sha(ROOT / REFERENCES)}
    for kind, (name, size, checksum) in FILES.items():
        path = RAW / name
        if files[name]['size'] != size or files[name]['checksum'] != f'md5:{checksum}' or path.stat().st_size != size:
            raise ValueError('Published or local file metadata differ')
        md5, sha256 = hashlib.md5(usedforsecurity=False), hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(8 * 1024**2), b''):
                md5.update(block)
                sha256.update(block)
        if md5.hexdigest() != checksum:
            raise ValueError('File differs from the publisher MD5')
        digests[kind] = sha256.hexdigest()
    return digests


def read_labels(path, *, prefix_profile=None):
    from openpyxl import load_workbook

    book = load_workbook(path, read_only=True, data_only=False)
    try:
        if book.sheetnames != ['Toutes']:
            raise ValueError('Expected the official classification sheet')
        rows = iter(book['Toutes'].values)
        if next(rows) != ('image_id', 'classe', 'grade_DR', 'sexe', 'age', 'type_diabete'):
            raise ValueError('Unexpected workbook header')
        labels = {}
        sequences, metadata = defaultdict(list), defaultdict(set)
        for identity, label, text, sex, age, diabetes in rows:
            if (not isinstance(identity, str) or not re.fullmatch(ID, identity) or identity in labels
                    or type(label) is not int or label not in range(5) or GRADE_TEXT[label] != text):
                raise ValueError('Unexpected, inconsistent or duplicate released label')
            labels[identity] = label
            sequences[identity[:8]].append(int(identity[8:]))
            metadata[identity[:8]].add((sex, age, diabetes))
        if prefix_profile is not None:
            prefix_profile.update(prefixes=len(sequences),
                                  multi_image_prefixes=sum(len(values) > 1 for values in sequences.values()),
                                  contiguous_numbered_prefixes=sum(sorted(values) == list(range(1, len(values) + 1))
                                                                   for values in sequences.values()),
                                  metadata_consistent_prefixes=sum(len(values) == 1 for values in metadata.values()))
        return labels
    finally:
        book.close()


def inspect_archive(path, labels, *, expected_unmatched=0):
    rows = []
    with ZipFile(path) as archive:
        entries = {}
        for info in archive.infolist():
            if info.filename == 'images/' and info.is_dir():
                continue
            match = re.fullmatch(rf'images/({ID}(?:\(2\))?)\.jpg', info.filename)
            if not match or match[1] in entries:
                raise ValueError('Unexpected or duplicate archive member')
            entries[match[1]] = info
        orphan, missing = entries.keys() - labels.keys(), labels.keys() - entries.keys()
        if (len(orphan) != expected_unmatched or len(missing) != expected_unmatched
                or any(not identity.endswith('(2)') or identity[:-3] not in (entries.keys() & labels.keys())
                       for identity in orphan)):
            raise ValueError('Archive and workbook join differs from the expected exact matches and orphan')
        for identity, info in sorted(entries.items()):
            data = archive.read(info)
            with Image.open(BytesIO(data)) as photo:
                if photo.format != 'JPEG' or photo.mode != 'RGB' or getattr(photo, 'n_frames', 1) != 1:
                    raise ValueError('Expected an original single-frame RGB JPEG')
                photo.load()
                rows.append({'image_id': identity, 'member': info.filename, 'dr_grade': labels.get(identity),
                             'format': photo.format, 'width': photo.width, 'height': photo.height,
                             'sha256': hashlib.sha256(data).hexdigest(), 'phash': retinal_phash(photo)})
    return rows


def screen_overlap(rows, references):
    grade_by_id = {row['image_id']: row['dr_grade'] for row in rows}
    # The graph helper requires integer labels. Derive grade conflicts below
    # from the joined labels only. The surrogate never becomes a source grade.
    frame = pd.DataFrame(rows).assign(dataset_source='FDR26_v3',
                                     dr_grade=[0 if row['dr_grade'] is None else row['dr_grade'] for row in rows])
    groups, internal = fingerprint_components(frame, near_duplicate_distance=2)
    cross = cross_dataset_candidates(groups, references, distance=2)
    cross_groups = set(groups.loc[groups.image_id.isin(cross.external_image_id), 'split_group_id'])
    joined_grades, unmatched_groups = defaultdict(set), set()
    for row in groups.itertuples():
        grade = grade_by_id[row.image_id]
        if grade is None:
            unmatched_groups.add(row.split_group_id)
        else:
            joined_grades[row.split_group_id].add(grade)
    conflicting = {group for group, grades in joined_grades.items() if len(grades) > 1}
    internal_records = internal.to_dict('records')
    for pair in internal_records:
        a, b = (grade_by_id[pair[key]] for key in ('first_image_id', 'second_image_id'))
        pair['label_conflict'] = a is not None and b is not None and a != b
    first = groups.groupby('split_group_id').image_id.min()
    for row, group in zip(rows, groups.itertuples(), strict=True):
        if row['image_id'] != group.image_id:
            raise ValueError('Fingerprint graph changed archive row order')
        matched = group.split_group_id in cross_groups
        eligible = not (matched or group.split_group_id in conflicting or group.split_group_id in unmatched_groups)
        row.update(component=group.split_group_id, label_conflict=group.split_group_id in conflicting,
                   cross_candidate=matched, eligible=eligible,
                   representative=eligible and group.image_id == first[group.split_group_id])
    overlap = {
        'internal_pairs': len(internal), 'internal_exact_pairs': int(internal.byte_exact.sum()),
        'components': int(groups.split_group_id.nunique()),
        'conflicting_components': len(conflicting),
        'unmatched_component_excluded_images': sum(row['component'] in unmatched_groups for row in rows),
        'cross_pairs': len(cross), 'cross_exact_pairs': int(cross.byte_exact.sum()),
        'cross_images': int(cross.external_image_id.nunique()),
        'excluded_images': sum(not row['eligible'] for row in rows),
        'eligible_images': sum(row['eligible'] for row in rows),
        'representatives': sum(row['representative'] for row in rows),
        'reference_rows': len(references),
        'reference_pairs_by_dataset': {name: int(cross.reference_dataset.eq(name).sum())
                                       for name in sorted(references.dataset_source.unique())},
    }
    return internal_records, cross.to_dict('records'), overlap


def prefix_sensitivity(rows):
    components, prefixes = defaultdict(list), defaultdict(list)
    for index, row in enumerate(rows):
        components[row['component']].append(index)
        if row['dr_grade'] is not None:
            prefixes[row['image_id'][:8]].append(index)
    excluded = {index for index, row in enumerate(rows) if not row['eligible']}
    pending = list(excluded)
    visited_components, visited_prefixes = set(), set()
    while pending:
        row = rows[pending.pop()]
        prefix = row['image_id'][:8] if row['dr_grade'] is not None else None
        for key, groups, seen in ((row['component'], components, visited_components),
                                  (prefix, prefixes, visited_prefixes)):
            if key not in seen:
                seen.add(key)
                added = set(groups[key]) - excluded
                excluded.update(added)
                pending.extend(added)
    return {'prefix_sensitivity_excluded_images': len(excluded),
            'prefix_sensitivity_eligible_images': len(rows) - len(excluded),
            'prefix_sensitivity_representatives': sum(row['representative'] and index not in excluded
                                                      for index, row in enumerate(rows))}


def encode(value):
    return (json.dumps(value, indent=2, allow_nan=False) + '\n').encode('utf-8')


def run(commit):
    private, public = ROOT / 'artifacts' / SLUG, ROOT / 'reports' / SLUG
    if private.exists() or public.exists():
        raise ValueError('Preserve existing readiness artifacts')
    sources = source_bindings(commit)
    hashes = verify_release()
    prefixes = {}
    labels = read_labels(RAW / FILES['workbook'][0], prefix_profile=prefixes)
    if len(labels) != 3000:
        raise ValueError('Expected the 3,000-image classification release')
    config = json.loads((ROOT / REFERENCES).read_bytes())
    references, _ = load_references(config)
    rows = inspect_archive(RAW / FILES['archive'][0], labels, expected_unmatched=1)
    internal, cross, overlap = screen_overlap(rows, references)
    overlap.update(prefix_sensitivity(rows))
    joined = {row['image_id']: row['dr_grade'] for row in rows if row['dr_grade'] is not None}
    missing = sorted(labels.keys() - joined.keys())
    witness = encode({'rows': rows, 'unmatched_workbook_ids': missing,
                      'internal_candidates': internal, 'cross_candidates': cross})
    report = {
        'experiment_id': SLUG, 'dataset': 'FDR26', 'partition': 'external_candidate_unscored',
        'scope': 'external_grading_data_readiness', 'source_commit': commit, 'source_sha256': sources,
        'input_sha256': hashes, 'witness_sha256': hashlib.sha256(witness).hexdigest(),
        'inventory': {'images': len(rows), 'archive_images': len(rows), 'workbook_rows': len(labels),
                      'joined_images': len(joined), 'unmatched_archive_images': len(rows) - len(joined),
                      'unmatched_workbook_rows': len(missing),
                      'grade_counts': dict(Counter(str(value) for value in labels.values())),
                      'joined_grade_counts': dict(Counter(str(value) for value in joined.values())),
                      'formats': dict(Counter(row['format'] for row in rows)), 'filename_prefix_profile': prefixes},
        'overlap': overlap, 'grade_standard': 'ALFEDIAM', 'patient_linkage_available': False,
        'model_inference_performed': False,
    }
    private.mkdir(parents=True)
    public.mkdir(parents=True)
    with (private / 'witness.json').open('xb') as stream:
        stream.write(witness)
    with (public / 'evaluation.json').open('xb') as stream:
        stream.write(encode(report))
    print(json.dumps({'inventory': report['inventory'], 'overlap': overlap}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-commit', required=True)
    run(parser.parse_args().source_commit)
