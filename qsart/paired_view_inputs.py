"""Bind and score the frozen DeepDRiD development pairs by physical view."""

from collections import defaultdict
from contextlib import ExitStack
import hashlib
from importlib.metadata import version
from io import BytesIO
import json
import math
from pathlib import Path
import re

from PIL import Image

from qsart.inference.engine import InferenceEngine
from qsart.models.checkpoint import sha256_file
from qsart.training.data import read_manifest


ROOT = Path(__file__).resolve().parents[1]
CPU_THREADS = 4
CHUNK_SIZE = 8
AUTHOR_COMMIT = '56d8af71319803150d0e7b90533e0918c7bf85ee'
PATIENT_COUNTS = {'train': 212, 'calibration': 42, 'validation': 42}
SCORED_SPLITS = ('train', 'validation')
VIEWS = ('l1', 'l2', 'r1', 'r2')
INPUTS = {
    'manifest': ('data/processed/deepdrid/development_v1.csv',
                 '45e9f2d34ed54a12056b57a808d03cfaf29ccc970d148e9eb107b79cf5ac3e0d'),
    'dataset_report': ('reports/development_v1/data_summary.json',
                       'a1745a8c555e15b6174f2a2400cb0bc9babca9e18eb1b4eb7db55c76500e7965'),
    'dr': ('checkpoints/development_v1/selected/dr.pt',
           'f530056c27f83dbff8bba4ebb481cf7a9d82289947c4eda655a69ed22685a527'),
    'quality': ('checkpoints/development_v1/selected/quality.pt',
                'a600872ce8e2b256205350a5ef6c99b77d71f679d217c20c26810a3ed76eff43'),
    'calibration': ('checkpoints/development_v1/selected/calibration.json',
                    '674821e445490c209fab99e56b0e9915ac406ca4ec8694e5efdbed63e456a737'),
    'model_card': ('checkpoints/development_v1/selected/model_card.json',
                   '43830695b8362aabd06df0d5dfa65b01086db9da2db3212001ac98b8b0e4710f'),
    'selection_lock': ('checkpoints/development_v1/selected/selection_lock.json',
                       'c2366e8ac8d369759d7431c6135bc593b5fa4623c4ce906a42cefa478928e3c2'),
}


def _bound_inputs():
    paths = {name: ROOT / path for name, (path, _) in INPUTS.items()}
    try:
        hashes = {name: sha256_file(path) for name, path in paths.items()}
    except OSError:
        raise ValueError('A required frozen paired-view input is unavailable') from None
    if hashes != {name: checksum for name, (_, checksum) in INPUTS.items()}:
        raise ValueError('A frozen paired-view input checksum differs')
    return paths, hashes


def _ordered_pairs(rows, expected_patients=PATIENT_COUNTS):
    patients = defaultdict(dict)
    released_ids = {}
    seen_hashes = set()
    for row in rows:
        patient, release_id = row['subject_id'], str(row['subject_id_original'])
        image_prefix, _, view = row['image_id'].rpartition('_')
        eye = row['eye']
        if (not isinstance(patient, str) or not patient or not release_id.isdecimal()
                or release_id != str(row['original_patient_id'])
                or image_prefix != release_id or view not in VIEWS
                or eye != ('left' if view.startswith('l') else 'right')
                or row['original_image_id'] != row['image_id']
                or Path(row['image_path']).stem != row['image_id']
                or row['dataset_source'] != 'DeepDRiD_regular_fundus'
                or row['official_split'] != 'official_train' or row['cohort_role'] != 'development_v1'
                or row['annotation_status'] != 'verified' or row['source_commit'] != AUTHOR_COMMIT
                or row['split'] not in expected_patients or type(row['dr_grade']) is not int
                or row['dr_grade'] not in range(5)
                or re.fullmatch('[0-9a-f]{64}', row['sha256']) is None
                or row['sha256'] in seen_hashes or view in patients[patient]
                or released_ids.setdefault(release_id, patient) != patient):
            raise ValueError('Invalid or duplicate DeepDRiD development photo identity')
        seen_hashes.add(row['sha256'])
        patients[patient][view] = row

    pairs = {split: [] for split in expected_patients}
    for patient, views in sorted(patients.items()):
        if (set(views) != set(VIEWS) or len({row['split'] for row in views.values()}) != 1
                or len({str(row['subject_id_original']) for row in views.values()}) != 1):
            raise ValueError('Each DeepDRiD patient needs four distinct views in one partition')
        partition = views['l1']['split']
        for eye, short in (('left', 'l'), ('right', 'r')):
            first, second = views[f'{short}1'], views[f'{short}2']
            if first['dr_grade'] != second['dr_grade']:
                raise ValueError('Released grades differ between views of one eye')
            pairs[partition].append((patient, eye, first, second))
    if any(len(pairs[split]) != 2 * count for split, count in expected_patients.items()):
        raise ValueError('Paired-view cohort counts differ from the frozen split')
    return pairs


def _photo_bytes(row):
    try:
        payload = Path(row['image_path']).read_bytes()
    except OSError:
        raise ValueError('A verified DeepDRiD development photograph is unavailable') from None
    if len(payload) != row['source_bytes'] or hashlib.sha256(payload).hexdigest() != row['sha256']:
        raise ValueError('A verified DeepDRiD development photograph differs from the frozen inventory')
    return payload


def _convert_prediction(prediction, model_version):
    probabilities = prediction['grade_probabilities']
    good_quality = 1. - prediction['quality']['poor_probability']
    if (prediction['model_version'] != model_version or prediction['data_purpose'] != 'real_research'
            or prediction['second_pass_activated'] is not False or len(probabilities) != 5
            or any(not 0. <= value <= 1. for value in probabilities)
            or not math.isclose(math.fsum(probabilities), 1., rel_tol=0, abs_tol=1e-6)
            or not 0. <= good_quality <= 1.):
        raise ValueError('Invalid genuine first-pass result for a DeepDRiD development photograph')
    return probabilities, good_quality


def _score_pairs(engine, model_version, pairs):
    scored = []
    for offset in range(0, len(pairs), CHUNK_SIZE // 2):
        chunk = pairs[offset:offset + CHUNK_SIZE // 2]
        photos = [row for _, _, first, second in chunk for row in (first, second)]
        with ExitStack() as stack:
            images = []
            for row in photos:
                photo = stack.enter_context(Image.open(BytesIO(_photo_bytes(row))))
                if (photo.format != 'JPEG' or photo.mode != 'RGB' or getattr(photo, 'n_frames', 1) != 1
                        or photo.size != (row['width'], row['height'])):
                    raise ValueError('A frozen development photo does not match its decoded dimensions or format')
                photo.load()
                images.append(photo)
            predictions = engine.predict_images(images, second_pass_budget=0., second_pass_policy='none')
        scores = [_convert_prediction(row, model_version) for row in predictions]
        for (patient, eye, first, second), (p_first, quality), (p_second, _) in zip(
                chunk, scores[::2], scores[1::2], strict=True):
            scored.append({'patient': patient, 'eye': eye, 'grade': first['dr_grade'],
                           'first': p_first, 'second': p_second, 'first_quality': quality,
                           'first_id': first['image_id'], 'second_id': second['image_id'],
                           'first_sha256': first['sha256'], 'second_sha256': second['sha256']})
    return scored


def load_and_score():
    """Return train and selection observations for private use plus public source provenance."""
    paths, hashes = _bound_inputs()
    report = json.loads(paths['dataset_report'].read_bytes())['development_v1']
    card = json.loads(paths['model_card'].read_bytes())
    if (report['manifest_sha256'] != hashes['manifest']
            or any((report['partitions'][name]['images'], report['partitions'][name]['subjects']) != (n * 4, n)
                   for name, n in PATIENT_COUNTS.items())
            or card['research_only'] is not True
            or card['dr_checkpoint_sha256'] != hashes['dr'] or card['quality_checkpoint_sha256'] != hashes['quality']
            or card['calibration_sha256'] != hashes['calibration']
            or card['selection_lock_sha256'] != hashes['selection_lock']):
        raise ValueError('Frozen paired-view model or partition provenance differs')
    pairs = _ordered_pairs(read_manifest(paths['manifest']).to_dict('records'))
    for split in SCORED_SPLITS:
        for _, _, first, second in pairs[split]:
            _photo_bytes(first)
            _photo_bytes(second)

    engine = InferenceEngine.load(paths['dr'], paths['quality'], paths['calibration'], cpu_threads=CPU_THREADS)
    info = engine.info()
    if (info['data_purpose'] != 'real_research' or info['device'] != 'cpu' or info['cpu_threads'] != CPU_THREADS
            or info['model_version'] != card['model_version'] or info['quality_model_version'] != card['quality_model_version']
            or any(info['training_provenance'][task]['manifest_sha256'] != hashes['manifest']
                   for task in ('dr', 'quality'))):
        raise ValueError('Unexpected frozen engine for paired-view development scoring')
    fit, selection = (_score_pairs(engine, info['model_version'], pairs[split]) for split in SCORED_SPLITS)
    provenance = {
        'input_sha256': hashes, 'source_release_commit': AUTHOR_COMMIT,
        'cohort': 'DeepDRiD regular Set-A development',
        'partitions': {name: {'photos': count * 4, 'eyes': count * 2, 'patients': count,
                             'scored': name in SCORED_SPLITS} for name, count in PATIENT_COUNTS.items()},
        'model_version': info['model_version'], 'quality_model_version': info['quality_model_version'],
        'probability_path': 'calibrated_first_pass_per_physical_photo',
        'device': info['device'], 'cpu_threads': CPU_THREADS, 'photo_chunk_size': CHUNK_SIZE,
        'second_pass_inference': False, 'first_view_suffix': 1, 'second_view_suffix': 2,
        'package_versions': {package: version(package) for package in ('torch', 'torchvision', 'pillow', 'numpy')},
    }
    return fit, selection, provenance
