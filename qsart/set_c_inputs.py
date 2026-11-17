"""Verify the frozen DeepDRiD Set-C inputs and obtain first-pass scores."""

from collections import defaultdict
from contextlib import ExitStack
import hashlib
from importlib.metadata import version
from io import BytesIO
from itertools import product
import json
import math
from pathlib import Path
import re

from PIL import Image

from qsart.inference.conformal_sidecar import ConformalSidecar, content_hash
from qsart.inference.engine import InferenceEngine
from qsart.models.checkpoint import sha256_file


ROOT = Path(__file__).resolve().parents[1]
CPU_THREADS = 4
CHUNK_SIZE = 8
ALPHA = .10
AUTHOR_RELEASE = {
    'commit': '56d8af71319803150d0e7b90533e0918c7bf85ee',
    'tree': '8340aa4fa31819be0e6da05bb842804f40284bdf',
    'license': 'CC-BY-SA-4.0',
}
SOURCE_FOLDER = 'regular_fundus_images/Online-Challenge1&2-Evaluation/Images'
POPULATION = {'images': 400, 'patients': 100, 'eyes': 200,
              'eligible_images': 396, 'eligible_patients': 99, 'excluded_patients': 1}
INPUTS = {
    'readiness_report': ('reports/deepdrid_online_readiness_v1/evaluation.json',
                         '57e77bef5d18fd154d6b0b2db80c2a6753d9a6b463037bb6b48f652c5f031d9d'),
    'readiness_audit': ('reports/deepdrid_online_readiness_v1/independent_audit.json',
                        '8c0ec473ad9853c20dec41a3d3a5392777930d2b6429b8fcaf14af2823cbe7f7'),
    'readiness_witness': ('artifacts/deepdrid_online_readiness_v1/witness.json',
                          'b2d176eaf8277806c16274648cc26dc52f5bf7021de1878ebbec6f3fa4d031b8'),
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
    'conformal': ('checkpoints/spec_v1/conformal_r1.json',
                  '15a98f6bfe93f81156c5592701995d08796dd889e3a5e43bcbde7ab5aa61c221'),
}


def _bound_inputs():
    paths = {key: ROOT / name for key, (name, _) in INPUTS.items()}
    hashes = {key: digest for key, (_, digest) in INPUTS.items()}
    try:
        actual = {key: sha256_file(path) for key, path in paths.items()}
    except OSError:
        raise ValueError('A required local Set-C research artifact is unavailable') from None
    if actual != hashes:
        raise ValueError('Frozen Set-C research artifact checksum mismatch')
    return paths, hashes


def _eligible_rows(readiness, audit, witness, hashes):
    # The historical audit stays in its readiness-only scope after this new study.
    if (readiness['experiment_id'] != 'deepdrid_online_readiness_v1'
            or readiness['partition'] != 'online_challenge_unscored'
            or readiness['scope'] != 'same_release_held_out_patient_data_readiness'
            or readiness['model_inference_performed'] is not False
            or readiness['witness_sha256'] != hashes['readiness_witness']
            or audit['audit_status'] != 'passed'
            or audit['experiment_id'] != readiness['experiment_id']
            or audit['model_inference_performed'] is not False
            or audit['evaluation_sha256'] != hashes['readiness_report']
            or audit['private_witness_sha256'] != hashes['readiness_witness']
            or audit['source_commit'] != readiness['source_commit']
            or any(record['source_release'] != AUTHOR_RELEASE or record['population'] != POPULATION
                   for record in (readiness, audit))):
        raise ValueError('Set-C readiness and independent audit bindings differ')

    groups = defaultdict(dict)
    eligible = []
    expected_views = set(product('lr', (1, 2)))
    for row in witness['online']:
        patient, eye, view = row['patient_id'], row['eye'], row['view']
        identity = f'{patient}_{eye}{view}'
        if (re.fullmatch(r'[0-9]+', patient) is None or type(view) is not int
                or (eye, view) not in expected_views or (eye, view) in groups[patient]
                or row['image_id'] != identity or row['path'] != f'{SOURCE_FOLDER}/{patient}/{identity}.jpg'
                or re.fullmatch(r'[0-9a-f]{64}', row['sha256']) is None or type(row['eligible']) is not bool
                or type(row['dr_grade']) is not int or row['dr_grade'] not in range(5)
                or type(row['overall_quality']) is not int or row['overall_quality'] not in (0, 1)):
            raise ValueError('Frozen Set-C roster, grading or quality fields are invalid')
        groups[patient][eye, view] = row
        if row['eligible']:
            eligible.append(row)

    excluded, referable = set(), 0
    for views in groups.values():
        if views.keys() != expected_views or len({row['eligible'] for row in views.values()}) != 1:
            raise ValueError('Set-C requires four complete, consistently masked views per patient')
        if any(views[eye, 1]['dr_grade'] != views[eye, 2]['dr_grade'] for eye in 'lr'):
            raise ValueError('Released grades differ between views of the same eye')
        if views['l', 1]['eligible']:
            referable += max(views['l', 1]['dr_grade'], views['r', 1]['dr_grade']) >= 2
        else:
            excluded.add(views['l', 1]['patient_id'])
    expected_excluded = witness['excluded_patients']
    if (len(witness['online']) != POPULATION['images'] or len(groups) != POPULATION['patients']
            or len(eligible) != POPULATION['eligible_images'] or len(excluded) != POPULATION['excluded_patients']
            or len(expected_excluded) != len(excluded) or set(expected_excluded) != excluded or referable != 48):
        raise ValueError('Set-C whole-patient eligibility differs from the frozen readiness audit')
    return eligible


def _original_bytes(author_root, row):
    try:
        path = (author_root / row['path']).resolve(strict=True)
        if not path.is_relative_to(author_root) or not path.is_file():
            raise ValueError('A Set-C photograph is outside the author package')
        payload = path.read_bytes()
    except OSError:
        raise ValueError('A frozen eligible Set-C photograph is unavailable') from None
    if hashlib.sha256(payload).hexdigest() != row['sha256']:
        raise ValueError('Frozen eligible Set-C photograph checksum mismatch')
    return payload


def _probabilities(values):
    return (len(values) == 5 and all(math.isfinite(value) and 0 <= value <= 1 for value in values)
            and math.isclose(math.fsum(values), 1., abs_tol=1e-5))


def _scored_row(row, prediction, model_version):
    p = prediction['grade_probabilities']
    raw = prediction['uncalibrated']['grade_probabilities']
    entropy = prediction['uncertainty']['entropy']
    good_quality = 1. - prediction['quality']['poor_probability']
    conformal = prediction['conformal']
    ordinal, lac = conformal['prediction_set'], conformal['baseline_lac_set']
    if (prediction['data_purpose'] != 'real_research' or prediction['model_version'] != model_version
            or prediction['second_pass_activated'] is not False or conformal['path'] != 'first_pass'
            or not _probabilities(p) or not _probabilities(raw) or not 0 <= good_quality <= 1
            or not 0 <= entropy <= 1 or any(len(grades) != len(set(grades))
                                            or any(type(grade) is not int or grade not in range(5) for grade in grades)
                                            for grades in (ordinal, lac))):
        raise ValueError('Invalid genuine first-pass Set-C model output')
    return {'id': row['image_id'], 'patient': row['patient_id'], 'eye': row['eye'], 'view': row['view'],
            'grade': row['dr_grade'], 'quality_label': row['overall_quality'], 'sha256': row['sha256'],
            'p': p, 'raw': raw, 'good_quality': good_quality, 'entropy': entropy, 'ordinal': ordinal, 'lac': lac}


def load_and_score(author_root):
    """Return private image rows in witness order and path-free provenance for aggregate reports."""
    paths, hashes = _bound_inputs()
    try:
        readiness, audit, witness, card = (
            json.loads(paths[key].read_bytes()) for key in ('readiness_report', 'readiness_audit',
                                                           'readiness_witness', 'model_card'))
        author_root = Path(author_root).resolve(strict=True)
    except OSError:
        raise ValueError('A bound input or the author package is unavailable') from None
    eligible = _eligible_rows(readiness, audit, witness, hashes)
    for row in eligible:
        _original_bytes(author_root, row)

    engine = InferenceEngine.load(paths['dr'], paths['quality'], paths['calibration'], cpu_threads=CPU_THREADS)
    sidecar = ConformalSidecar(paths['conformal'], dr_checkpoint=paths['dr'], quality_checkpoint=paths['quality'],
                              calibration=paths['calibration'])
    info, bundle = engine.info(), sidecar.bundle
    if (info['data_purpose'] != 'real_research' or info['device'] != 'cpu' or info['cpu_threads'] != CPU_THREADS
            or card['research_only'] is not True or card['model_version'] != info['model_version']
            or card['quality_model_version'] != info['quality_model_version']
            or card['dr_checkpoint_sha256'] != hashes['dr'] or card['quality_checkpoint_sha256'] != hashes['quality']
            or card['calibration_sha256'] != hashes['calibration'] or card['selection_lock_sha256'] != hashes['selection_lock']
            or card['calibration_content_sha256'] != content_hash(engine.calibration)
            or bundle['model_version'] != info['model_version'] or bundle['patient_groups_disjoint'] is not True
            or bundle['manifest_sha256'] != readiness['prior_manifest_coverage']['development_v1']['sha256']
            or (bundle['temperature_images'], bundle['temperature_patients']) != (84, 21)
            or (bundle['conformal_images'], bundle['conformal_patients']) != (84, 21)):
        raise ValueError('Frozen Set-C grading, quality or conformal calibration provenance differs')

    images = []
    for start in range(0, len(eligible), CHUNK_SIZE):
        chunk = eligible[start:start + CHUNK_SIZE]
        with ExitStack() as stack:
            decoded = []
            for row in chunk:
                photo = stack.enter_context(Image.open(BytesIO(_original_bytes(author_root, row))))
                if photo.format != 'JPEG' or photo.mode != 'RGB' or getattr(photo, 'n_frames', 1) != 1:
                    raise ValueError('Frozen Set-C inputs must be single-frame RGB JPEGs')
                photo.load()
                decoded.append(photo)
            predictions = engine.predict_images(decoded, second_pass_budget=0., second_pass_policy='none',
                                                include_uncalibrated=True)
        for row, prediction in zip(chunk, sidecar.enrich(predictions, alpha=ALPHA), strict=True):
            images.append(_scored_row(row, prediction, info['model_version']))
    return images, {
        'artifact_sha256': hashes, 'source_release': AUTHOR_RELEASE.copy(),
        'model_version': info['model_version'], 'quality_model_version': info['quality_model_version'],
        'device': info['device'], 'cpu_threads': CPU_THREADS, 'inference_chunk_size': CHUNK_SIZE,
        'probability_path': 'first_pass', 'conformal_alpha': ALPHA,
        'calibration': {'development_manifest_sha256': bundle['manifest_sha256'],
                        'temperature_fit_split_sha256': bundle['temperature_fit_split_sha256'],
                        'conformal_split_sha256': bundle['split_sha256'],
                        'temperature_patients': bundle['temperature_patients'],
                        'conformal_patients': bundle['conformal_patients'],
                        'patient_groups_disjoint': True},
        'package_versions': {package: version(package) for package in ('torch', 'torchvision', 'pillow', 'numpy')},
    }
