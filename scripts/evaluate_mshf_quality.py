"""Evaluate the frozen overall-quality head on the official MSHF v2 CFP release."""

from __future__ import annotations

import argparse
from collections import Counter
from importlib import metadata
import hashlib
from io import BytesIO
import json
import math
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
from zipfile import ZipFile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from openpyxl import load_workbook
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score
import torch

from qsart.data.aptos import fingerprint_components
from qsart.data.idrid import cross_dataset_candidates
from qsart.data.manifests import retinal_phash, sha256_file
from qsart.models.checkpoint import load_checkpoint
from qsart.training.data import image_to_tensor
from scripts.prepare_idrid import load_references

ROOT = Path(__file__).resolve().parents[1]
SLUG = 'mshf_quality_v1'
PROTOCOL = 'docs/research/MSHF_QUALITY_V1_PROTOCOL.md'
REFERENCE_CONFIG = 'configs/idrid_data_v1.json'
REFERENCE_SHA = '11461a9a8a10b8554f3cf35a518a3650085714ac2ca439daecec22e96e16b027'
SELECTED = ROOT / 'checkpoints/development_v1/selected'
QUALITY_SHA = 'a600872ce8e2b256205350a5ef6c99b77d71f679d217c20c26810a3ed76eff43'
CALIBRATION_SHA = '674821e445490c209fab99e56b0e9915ac406ca4ec8694e5efdbed63e456a737'
ARCHIVE_ROOT = 'MSHF dataset 2.0'
ARCHIVE_SOURCE = {
    'article_id': 21507564, 'version': 2, 'file_id': 39485878,
    'doi': '10.6084/m9.figshare.21507564.v2', 'license': 'CC BY 4.0',
    'license_url': 'https://creativecommons.org/licenses/by/4.0/',
    'download_url': 'https://ndownloader.figshare.com/files/39485878',
    'bytes': 1136891149, 'publisher_md5': '58203e6c2e064dafc800f5b83887487b',
    'archive_sha256': '115def6513bc1516d77ceb4ac2e90a9fcf582e825604c08be9524f5fd6c4a219',
}
WORKBOOKS = {
    'individual': ('Individual_scores.xlsx.xlsx',
                   '299d331e7e252385fe413661d0b471a8342d45115cdab189a19eb21159ed8407'),
    'summary': ('MSHF_quality_scores.xlsx',
                '40481a6f207bcd8f2e0daa8bd4c1da6a5a41c3161a5b97fa5c74d499970381f7'),
}
PAPER_COUNTS = {'DR-XJU': (117, 118), 'DR-ZJU': (40, 147), 'Glaucoma': (50, 2), 'Healthy': (0, 26)}
GROUP_COUNTS = {group: sum(counts) for group, counts in PAPER_COUNTS.items()}
ENCODED_FORMATS = {'DR-XJU': 'JPEG', 'DR-ZJU': 'JPEG', 'Glaucoma': 'TIFF', 'Healthy': 'TIFF'}
SOURCES = (
    PROTOCOL, REFERENCE_CONFIG, 'scripts/evaluate_mshf_quality.py', 'scripts/prepare_idrid.py',
    'qsart/data/aptos.py', 'qsart/data/idrid.py', 'qsart/data/manifests.py',
    'qsart/models/checkpoint.py', 'qsart/models/networks.py', 'qsart/training/data.py',
    'qsart/preprocessing/fundus.py',
)
CPU_THREADS = 2
THRESHOLD = .5


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def source_hashes(commit):
    def git(*args):
        return subprocess.check_output(['git', *args], cwd=ROOT)

    if git('rev-parse', 'HEAD').decode().strip() != commit or git('status', '--porcelain'):
        raise ValueError('Execution requires the exact declared clean source commit')
    hashes = {}
    for name in SOURCES:
        value = (ROOT / name).read_bytes()
        if value != git('show', f'{commit}:{name}'):
            raise ValueError(f'Committed execution source differs: {name}')
        hashes[name] = hashlib.sha256(value).hexdigest()
    return hashes


def verify_archive(path):
    receipt_path = path.with_name('receipt.json')
    metadata_path = path.with_name('figshare_v2.json')
    receipt = read_json(receipt_path)
    metadata_sha = sha256_file(metadata_path)
    if (
        any(receipt.get(key) != value for key, value in ARCHIVE_SOURCE.items())
        or receipt.get('metadata_sha256') != metadata_sha
        or path.stat().st_size != ARCHIVE_SOURCE['bytes']
    ):
        raise ValueError('Archive size or official provenance receipt differs')
    deposit = read_json(metadata_path)
    (file,) = [entry for entry in deposit['files'] if entry['id'] == ARCHIVE_SOURCE['file_id']]
    if (
        deposit['id'] != ARCHIVE_SOURCE['article_id'] or deposit['version'] != ARCHIVE_SOURCE['version']
        or deposit['doi'] != ARCHIVE_SOURCE['doi'] or deposit['license']['name'] != ARCHIVE_SOURCE['license']
        or deposit['license']['url'] != ARCHIVE_SOURCE['license_url']
        or file['size'] != ARCHIVE_SOURCE['bytes'] or file['download_url'] != ARCHIVE_SOURCE['download_url']
        or file['supplied_md5'] != ARCHIVE_SOURCE['publisher_md5']
        or file['computed_md5'] != ARCHIVE_SOURCE['publisher_md5']
    ):
        raise ValueError('Official Figshare record differs from the receipt')
    md5 = hashlib.md5(usedforsecurity=False)
    sha = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            md5.update(block)
            sha.update(block)
    if md5.hexdigest() != ARCHIVE_SOURCE['publisher_md5'] or sha.hexdigest() != ARCHIVE_SOURCE['archive_sha256']:
        raise ValueError('Official archive byte digests did not verify')
    return {**ARCHIVE_SOURCE, 'metadata_sha256': metadata_sha, 'receipt_sha256': sha256_file(receipt_path)}


def original_members(archive):
    prefix = f'{ARCHIVE_ROOT}/Original/CFP/'
    members = []
    for info in archive.infolist():
        if info.is_dir() or not info.filename.startswith(prefix):
            continue
        relative = info.filename.removeprefix(prefix)
        parts = relative.split('/')
        if len(parts) <= 2 and parts[-1] == '.DS_Store':
            continue
        if len(parts) != 2 or parts[0] not in GROUP_COUNTS or not re.fullmatch(
            re.escape(parts[0]) + r'-[0-9]+\.jpg', parts[1],
        ):
            raise ValueError('Unexpected member under the official Original/CFP directory')
        members.append((info.filename, *parts))
    if (
        Counter(group for _, group, _ in members) != GROUP_COUNTS
        or len({filename for _, _, filename in members}) != len(members)
    ):
        raise ValueError('Expected exactly 500 uniquely named original CFP images in four author groups')
    return sorted(members)


def workbook_bytes(archive, kind):
    filename, digest = WORKBOOKS[kind]
    member = f'{ARCHIVE_ROOT}/{filename}'
    if archive.namelist().count(member) != 1:
        raise ValueError('Workbook path is missing or ambiguous')
    raw = archive.read(member)
    if hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError('Archived workbook bytes differ')
    return raw


def individual_ratings(raw, groups):
    workbook = load_workbook(BytesIO(raw), read_only=True, data_only=False)
    try:
        if workbook.sheetnames != ['Sheet1']:
            raise ValueError('Unexpected individual-rating worksheet layout')
        rows = workbook['Sheet1'].iter_rows(max_col=15, values_only=True)
        first, second = next(rows), next(rows)
        columns = (4, 9, 14)
        if first[0] != 'image name' or any(second[index] != 'Overall' for index in columns):
            raise ValueError('Unexpected individual overall-rating columns')
        ratings = {}
        for row in rows:
            name = row[0]
            if not isinstance(name, str) or not name.startswith(groups):
                continue
            values = [row[index] for index in columns]
            if name in ratings or any(type(value) is not int or value not in (0, 1) for value in values):
                raise ValueError('Duplicate filename or nonbinary individual quality rating')
            ratings[name] = values
        return ratings
    finally:
        workbook.close()


def summary_ratings(raw, members):
    formulas = load_workbook(BytesIO(raw), read_only=True, data_only=False)
    cached = load_workbook(BytesIO(raw), read_only=True, data_only=True)
    try:
        summary = {}
        for group in GROUP_COUNTS:
            names = {name for _, source, name in members if source == group}
            stems = {PurePosixPath(name).stem: name for name in names}
            start = 3 if group == 'DR-ZJU' else 2
            formula_sheet, cached_sheet = formulas[group], cached[group]
            formula_rows = formula_sheet.iter_rows(min_row=start, max_col=5, values_only=True)
            cached_rows = cached_sheet.iter_rows(min_row=start, max_col=5, values_only=True)
            for source_row, cached_row in zip(formula_rows, cached_rows, strict=True):
                source = source_row[0]
                if not isinstance(source, str) or not source.startswith(group + '-'):
                    continue
                exact = source in names
                match = source if exact else stems.get(source)
                value, formula = cached_row[4], source_row[4]
                if (
                    source != cached_row[0] or match is None or match in summary
                    or type(value) is not int or value not in (0, 1)
                    or (not exact and (group != 'DR-XJU' or PurePosixPath(source).suffix))
                ):
                    raise ValueError('Invalid or ambiguous cached summary workbook filename/label')
                if group == 'DR-ZJU':
                    # All three referenced columns are past the last populated column E.
                    if value != 0 or formula_sheet.max_column != 5 or not isinstance(formula, str) or not re.fullmatch(
                        r'=IF\(SUM\(\(J([0-9]+)\+O\1\+T\1\)\)>1,1,0\)', formula,
                    ):
                        raise ValueError('Expected the archived DR-ZJU formulas with empty inputs')
                elif type(formula) is not int or formula != value:
                    raise ValueError('Unexpected formula or changed cached summary label')
                summary[match] = {'summary_filename': source, 'summary_join': 'exact' if exact else 'stem',
                                  'summary_cached': value,
                                  'summary_formula': formula if isinstance(formula, str) else None}
        if set(summary) != {name for _, _, name in members} or sum(
            row['summary_join'] == 'stem' for row in summary.values()
        ) != 2:
            raise ValueError('Summary workbook did not join exactly 500 originals with two stem matches')
        return summary
    finally:
        formulas.close()
        cached.close()


def inspect_originals(archive):
    members = original_members(archive)
    ratings = individual_ratings(workbook_bytes(archive, 'individual'), tuple(group + '-' for group in GROUP_COUNTS))
    if set(ratings) != {name for _, _, name in members}:
        raise ValueError('Individual rating rows and original CFP filenames differ')
    summary = summary_ratings(workbook_bytes(archive, 'summary'), members)
    rows = []
    for member, group, filename in members:
        raw = archive.read(member)
        with Image.open(BytesIO(raw)) as image:
            encoded_format = image.format
            if encoded_format != ENCODED_FORMATS[group] or image.mode != 'RGB' or getattr(image, 'n_frames', 1) != 1:
                raise ValueError('Original CFP encoding differs from the single-frame RGB release')
            image.verify()
        with Image.open(BytesIO(raw)) as image:
            image.load()
            width, height, phash = image.width, image.height, retinal_phash(image)
        values = ratings[filename]
        rows.append({'member': member, 'group': group, 'filename': filename, 'width': width, 'height': height,
                     'format': encoded_format, 'sha256': hashlib.sha256(raw).hexdigest(),
                     'phash': phash, 'ratings': values,
                     'label': int(sum(values) >= 2), 'unanimous': len(set(values)) == 1, **summary[filename]})
    return rows


def inventory(rows):
    dimensions = Counter((row['width'], row['height']) for row in rows)
    groups = {}
    for group, (paper_poor, paper_good) in PAPER_COUNTS.items():
        part = [row for row in rows if row['group'] == group]
        good = sum(row['label'] for row in part)
        cached_good = sum(row['summary_cached'] for row in part)
        formula_rows = sum(row['summary_formula'] is not None for row in part)
        groups[group] = {
            'images': len(part), 'poor': len(part) - good, 'good': good,
            'unanimous': sum(row['unanimous'] for row in part),
            'summary_cached_poor': len(part) - cached_good, 'summary_cached_good': cached_good,
            'summary_disagreements': sum(row['label'] != row['summary_cached'] for row in part),
            'summary_stem_joins': sum(row['summary_join'] == 'stem' for row in part),
            'summary_formula_rows': formula_rows, 'summary_blank_formula_inputs': formula_rows,
            'paper_poor': paper_poor, 'paper_good': paper_good,
        }
    return {'images': len(rows), 'images_decoded': len(rows), 'groups': groups,
            'formats': dict(sorted(Counter(row['format'] for row in rows).items())),
            'dimensions': [{'width': width, 'height': height, 'images': n}
                           for (width, height), n in sorted(dimensions.items())]}


def overlap_audit(rows):
    config_path = ROOT / REFERENCE_CONFIG
    if sha256_file(config_path) != REFERENCE_SHA:
        raise ValueError('Frozen IDRiD reference configuration differs')
    references, bindings = load_references(read_json(config_path), ROOT)
    # The graph helper uses dr_grade only to detect conflicts in an integer label.
    frame = pd.DataFrame([{'image_id': row['member'], 'dataset_source': 'MSHF_v2', 'dr_grade': row['label'],
                           'sha256': row['sha256'], 'phash': row['phash']} for row in rows])
    components, internal = fingerprint_components(frame, near_duplicate_distance=2)
    cross = cross_dataset_candidates(components, references, distance=2)
    cross_components = set(components.loc[components.image_id.isin(cross.external_image_id), 'split_group_id'])
    first_members = components.groupby('split_group_id').image_id.min().to_dict()
    for row, component in zip(rows, components.itertuples(), strict=True):
        if row['member'] != component.image_id:
            raise ValueError('Fingerprint output changed the archived row order')
        cross_candidate = component.split_group_id in cross_components
        eligible = not (component.component_label_conflict or cross_candidate)
        row.update(component=component.split_group_id.replace('aptos_splitgroup_', 'mshf_component_', 1),
                   label_conflict=bool(component.component_label_conflict), cross_candidate=cross_candidate,
                   eligible=eligible, representative=eligible and first_members[component.split_group_id] == row['member'])
    aggregate = {
        'near_duplicate_distance': 2, 'fingerprint_method': 'central_green_dct_phash_v1',
        'internal_components': int(components.split_group_id.nunique()),
        'internal_repeat_components': int((components.groupby('split_group_id').size() > 1).sum()),
        'internal_candidate_pairs': len(internal), 'internal_exact_pairs': int(internal.byte_exact.sum()),
        'conflicting_components': int(components.loc[components.component_label_conflict, 'split_group_id'].nunique()),
        'cross_source_candidate_pairs': len(cross), 'cross_source_exact_pairs': int(cross.byte_exact.sum()),
        'cross_source_candidate_images': int(cross.external_image_id.nunique()),
        'cross_source_components': len(cross_components),
        'excluded_conflict_images': sum(row['label_conflict'] for row in rows),
        'excluded_cross_source_images': sum(row['cross_candidate'] for row in rows),
        'excluded_images': sum(not row['eligible'] for row in rows),
        'reference_datasets': {str(dataset): {
            'rows': len(part), 'unique_identities': int(part.image_id.nunique()),
            'candidate_pairs': int(cross.reference_dataset.eq(dataset).sum()),
            'exact_pairs': int((cross.reference_dataset.eq(dataset) & cross.byte_exact).sum()),
        } for dataset, part in references.groupby('dataset_source')},
    }
    return internal.to_dict('records'), cross.to_dict('records'), aggregate, bindings


def score(archive, rows):
    model_path, calibration_path = SELECTED / 'quality.pt', SELECTED / 'calibration.json'
    if sha256_file(model_path) != QUALITY_SHA or sha256_file(calibration_path) != CALIBRATION_SHA:
        raise ValueError('Selected model or DeepDRiD calibration bytes differ')
    torch.set_num_threads(CPU_THREADS)
    model, checkpoint = load_checkpoint(model_path)
    info = checkpoint['metadata']
    calibration = read_json(calibration_path)
    quality = calibration['quality']
    temperature = quality['heads']['quality_grade']['temperature']
    if (
        info['task'] != 'quality' or info.get('production_eligible', True) is not True
        or info['data_purpose'] != 'real_research' or info['heads'].get('quality_grade') != 2
        or info['poor_class_indices'] != [0] or calibration['data_purpose'] != info['data_purpose']
        or calibration['format_version'] != 1 or calibration['method'] != 'temperature_scaling'
        or quality['checkpoint_sha256'] != QUALITY_SHA or quality['split'] != 'calibration'
        or quality['model_version'] != info['model_version'] or quality['data_purpose'] != info['data_purpose']
        or quality['manifest_sha256'] != info['manifest_sha256']
        or quality['split_fingerprint'] != info['split_fingerprints']['calibration']
        or type(temperature) not in (float, int) or not math.isfinite(temperature) or temperature <= 0
    ):
        raise ValueError('Expected the selected binary quality head and frozen first-pass temperature')
    with torch.inference_mode():
        for row in rows:
            row.update(logits=None, raw_probabilities=None, calibrated_probabilities=None)
            if not row['eligible']:
                continue
            raw = archive.read(row['member'])
            if hashlib.sha256(raw).hexdigest() != row['sha256']:
                raise ValueError('Original archive member changed after overlap audit')
            with Image.open(BytesIO(raw)) as image:
                tensor = image_to_tensor(image.convert('RGB'), info['image_size'], info['normalization_mean'],
                                         info['normalization_std']).unsqueeze(0)
            logits = model(tensor)['quality_grade']
            if logits.shape != (1, 2) or logits.dtype != torch.float32 or not torch.isfinite(logits).all():
                raise ValueError('Invalid quality logits')
            row.update(logits=logits[0].tolist(), raw_probabilities=logits.softmax(1)[0].tolist(),
                       calibrated_probabilities=(logits / temperature).softmax(1)[0].tolist())
    return {'quality_checkpoint_sha256': QUALITY_SHA, 'calibration_sha256': CALIBRATION_SHA,
            'model_version': info['model_version'], 'temperature': temperature, 'cpu_threads': CPU_THREADS,
            'image_size': info['image_size'], 'poor_probability_threshold': THRESHOLD,
            'manifest_sha256': info['manifest_sha256'], 'selected_epoch': int(checkpoint['epoch']) + 1}


def probability_metrics(truth, probabilities):
    if not len(truth):
        return None
    p = np.asarray(probabilities, dtype=float)
    pred = (p[:, 0] <= THRESHOLD).astype(int)
    counts = np.bincount(2 * truth + pred, minlength=4).reshape(2, 2)
    poor, good = map(int, counts.sum(axis=1))
    sensitivity = float(counts[0, 0] / poor) if poor else None
    specificity = float(counts[1, 1] / good) if good else None
    confidence = p.max(axis=1)
    correct = pred == truth
    ece = 0.
    for index in range(10):
        lower, upper = index / 10, (index + 1) / 10
        mask = (confidence >= lower) & (confidence <= upper if index == 9 else confidence < upper)
        if mask.any():
            ece += float(mask.mean() * abs(correct[mask].mean() - confidence[mask].mean()))
    return {
        'confusion_matrix': counts.tolist(), 'poor_sensitivity': sensitivity, 'good_specificity': specificity,
        'balanced_accuracy': (sensitivity + specificity) / 2 if poor and good else None,
        'poor_auroc': float(roc_auc_score(truth == 0, p[:, 0])) if poor and good else None,
        'poor_average_precision': float(average_precision_score(truth == 0, p[:, 0])) if poor and good else None,
        'nll': float(-np.log(p[np.arange(len(truth)), truth].clip(1e-12)).mean()),
        'brier': float(((p - np.eye(2)[truth]) ** 2).sum(axis=1).mean()), 'ece': ece,
        'accuracy': float(correct.mean()),
    }


def metric_cell(rows):
    truth = np.asarray([row['label'] for row in rows], dtype=int)
    n, good = len(truth), int(truth.sum())
    return {'images': n, 'poor': n - good, 'good': good, 'always_good_accuracy': good / n if n else None,
            'raw': probability_metrics(truth, [row['raw_probabilities'] for row in rows]),
            'calibrated': probability_metrics(truth, [row['calibrated_probabilities'] for row in rows])}


def population_metrics(rows):
    main = [row for row in rows if row['eligible']]
    if not main:
        raise ValueError('No original CFP image remains eligible')
    cohorts = {'main': main, 'representative': [row for row in main if row['representative']],
               'unanimous': [row for row in main if row['unanimous']]}
    return {name: {'overall': metric_cell(part),
                   'groups': {group: metric_cell([row for row in part if row['group'] == group])
                              for group in GROUP_COUNTS}}
            for name, part in cohorts.items()}


def encode_json(data):
    return (json.dumps(data, indent=2, allow_nan=False) + '\n').encode('utf-8')


def execute(args):
    output = (ROOT / args.output).resolve()
    private = (ROOT / args.private).resolve()
    for path, base in ((output, ROOT / 'reports'), (private, ROOT / 'artifacts')):
        if path == base or not path.is_relative_to(base) or path.exists():
            raise ValueError('Expected new report and ignored artifact subdirectories')
    sources = source_hashes(args.source_commit)
    archive_path = (ROOT / args.archive).resolve()
    source = verify_archive(archive_path)
    with ZipFile(archive_path) as archive:
        rows = inspect_originals(archive)
        internal, cross, overlap, bindings = overlap_audit(rows)
        model = score(archive, rows)
    witness = encode_json({'rows': rows, 'internal_candidates': internal, 'cross_source_candidates': cross})
    report = encode_json({
        'schema_version': 1, 'experiment_id': SLUG, 'spec_version': '1.0',
        'dataset': 'MSHF', 'partition': 'external', 'scope': 'external_overall_quality_diagnostic',
        'source_commit': args.source_commit, 'source_sha256': sources, 'protocol_sha256': sources[PROTOCOL],
        'archive': source, 'workbook_sha256': {kind: value[1] for kind, value in WORKBOOKS.items()},
        'reference_manifest_bindings': [{key: value for key, value in item.items() if key != 'path'}
                                        for item in bindings],
        'witness_sha256': hashlib.sha256(witness).hexdigest(),
        'package_versions': {name: metadata.version(name) for name in
                             ('numpy', 'pandas', 'scipy', 'scikit-learn', 'openpyxl', 'Pillow', 'torch')},
        'inventory': inventory(rows), 'overlap': overlap, 'model': model,
        'populations': population_metrics(rows),
        'limitations': [
            'Original conventional color photographs and majority overall-quality ratings from MSHF v2.',
            'Source-specific grading criteria and unresolved patient linkage limit interpretation.',
            'Fingerprint components are not patient identities. Related images can remain undetected.',
            'The main population excludes whole components with a source-overlap candidate or label conflict.',
            'The quality endpoint does not establish DR grading, triage utility, or clinical performance.',
        ],
    })
    if any(token in report for token in (b'"member"', b'"filename"', b'"image_id"', b'"logits"')) or re.search(
        rb'"(?:[A-Za-z]:[\\/]|/|\\\\)', report,
    ):
        raise ValueError('Public report contains a private field or path')
    private.mkdir(parents=True)
    with (private / 'witness.json').open('xb') as stream:
        stream.write(witness)
    output.mkdir(parents=True)
    with (output / 'evaluation.json').open('xb') as stream:
        stream.write(report)
    print(json.dumps({'experiment_id': SLUG, 'images': len(rows), 'scored': sum(row['eligible'] for row in rows)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--archive', type=Path, default=Path('data/raw/mshf_v2/MSHF.zip'))
    parser.add_argument('--output', type=Path, default=Path('reports') / SLUG)
    parser.add_argument('--private', type=Path, default=Path('artifacts') / SLUG)
    execute(parser.parse_args())


if __name__ == '__main__':
    main()
