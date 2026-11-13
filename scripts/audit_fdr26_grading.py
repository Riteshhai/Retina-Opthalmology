"""Independently reconstruct the frozen FDR26 grading comparisons and replay samples."""

from collections import defaultdict
import hashlib
from io import BytesIO
import json
import math
from pathlib import Path
import re
import subprocess
import sys
from zipfile import ZipFile

import numpy as np
from PIL import Image
from scipy.stats import rankdata


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = 'fdr26_grading_v1'
OUTPUT = ROOT / 'reports' / EXPERIMENT
WITNESS = ROOT / 'artifacts' / EXPERIMENT / 'witness.json'
SOURCE_FILES = {
    'docs/research/FDR26_GRADING_V1_PROTOCOL.md', 'scripts/evaluate_fdr26_grading.py',
    'scripts/audit_fdr26_grading.py', 'qsart/models/checkpoint.py', 'qsart/models/networks.py',
    'qsart/training/data.py', 'qsart/preprocessing/fundus.py', 'qsart/evaluation.py',
}
INPUTS = {
    'readiness_report': ('reports/fdr26_readiness_v1/evaluation.json',
                         'b9692d9db75ff0f8113f348e944ca62d64e699335f773c2c900852d513c1e2b9'),
    'readiness_audit': ('reports/fdr26_readiness_v1/independent_audit.json',
                        '4d16a03e781f501a1aa52fee9cf6958605e86599be483c83e1413f02e636fae5'),
    'readiness_witness': ('artifacts/fdr26_readiness_v1/witness.json',
                          '556e47ad165f76e5586c2936423b2a933985ff266c69135706c9c8f9ba8192b2'),
    'archive': ('data/raw/fdr26_v3/Fundus_FDR26.zip',
                '6c5517d60f24cbd310a1ab0607b8c90dbb4b88b61aee57840fc2e823b753dadb'),
    'checkpoint': ('checkpoints/development_v1/selected/dr.pt',
                   'f530056c27f83dbff8bba4ebb481cf7a9d82289947c4eda655a69ed22685a527'),
    'calibration': ('checkpoints/development_v1/selected/calibration.json',
                    '674821e445490c209fab99e56b0e9915ac406ca4ec8694e5efdbed63e456a737'),
}
MODEL = {'version': 'development-v1-dr-mobilenet_v3_small-seed2026-epoch15', 'image_size': 224,
         'temperature': 1.4621494531656656, 'cpu_threads': 2}
COUNTS = {'primary': 2849, 'representative': 2742, 'prefix': 2761, 'prefix_representative': 2673}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def match(expected, actual, label):
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and expected.keys() <= actual.keys(), f'{label} lacks required fields')
        for key, value in expected.items():
            match(value, actual[key], f'{label}.{key}')
    elif isinstance(expected, float):
        require(type(actual) in (int, float) and math.isclose(expected, actual, rel_tol=0, abs_tol=2e-12),
                f'{label} differs')
    else:
        require(expected == actual, f'{label} differs')


def verify_bindings(report):
    commit, sources = report['source_commit'], report['source_sha256']
    require(re.fullmatch(r'[0-9a-f]{40}', commit) and sources.keys() == SOURCE_FILES,
            'Expected locked execution source and complete dependency bindings')
    for path, expected in sources.items():
        raw = (ROOT / path).read_bytes()
        committed = subprocess.check_output(['git', 'show', f'{commit}:{path}'], cwd=ROOT)
        require(raw == committed and hashlib.sha256(raw).hexdigest() == expected, f'Locked source differs: {path}')
    paths = {name: ROOT / path for name, (path, _) in INPUTS.items()}
    hashes = {name: digest(path) for name, path in paths.items()}
    require(hashes == {name: expected for name, (_, expected) in INPUTS.items()}
            and hashes == report['input_sha256'], 'Readiness, archive or model artifact bytes differ')
    readiness, audit = (read(paths[key]) for key in ('readiness_report', 'readiness_audit'))
    require(audit['audit_status'] == 'passed' and audit['evaluation_sha256'] == hashes['readiness_report']
            and audit['private_witness_sha256'] == readiness['witness_sha256'] == hashes['readiness_witness']
            and readiness['input_sha256']['archive'] == hashes['archive']
            and readiness['model_inference_performed'] is audit['model_inference_performed'] is False,
            'The readiness audit is not bound to the unscored input artifacts')
    calibration = read(paths['calibration'])['dr']
    require(calibration['checkpoint_sha256'] == hashes['checkpoint'] and calibration['split'] == 'calibration'
            and calibration['model_version'] == MODEL['version'] and calibration['data_purpose'] == 'real_research'
            and calibration['heads']['dr_grade']['temperature'] == MODEL['temperature'],
            'Expected the frozen DR temperature and checkpoint provenance')
    match(MODEL, report['model'], 'model')
    return paths, readiness, calibration


def reconstruct_populations(readiness, witness):
    records = witness['rows']
    by_id = {row['image_id']: row for row in records}
    require(len(records) == len(by_id) == readiness['inventory']['archive_images'] == 3000,
            'Readiness archive population differs')
    primary = [row for row in records if row['eligible']]
    representative = [row for row in primary if row['representative']]
    require(all(type(row['dr_grade']) is int and 0 <= row['dr_grade'] <= 4 for row in primary)
            and len({row['component'] for row in primary}) == len(representative),
            'Invalid primary labels or fingerprint representatives')

    edges = defaultdict(set)
    for pair in witness['internal_candidates']:
        left, right = pair['first_image_id'], pair['second_image_id']
        require(left in by_id and right in by_id, 'Readiness pair references an unknown archive row')
        edges[left].add(right)
        edges[right].add(left)
    prefix_groups, prefix_key = defaultdict(set), {}
    for identity, row in by_id.items():
        if row['dr_grade'] is not None:
            require(re.fullmatch(r'[A-Z0-9]{8}[0-9]{2}', identity), 'Invalid joined filename prefix')
            prefix_key[identity] = identity[:8]
            prefix_groups[identity[:8]].add(identity)
    excluded = {identity for identity, row in by_id.items() if not row['eligible']}
    pending = list(excluded)
    while pending:
        identity = pending.pop()
        linked = edges[identity] | prefix_groups.pop(prefix_key.get(identity), set())
        new = linked - excluded
        excluded.update(new)
        pending.extend(new)
    populations = {
        'primary': primary, 'representative': representative,
        'prefix': [row for row in primary if row['image_id'] not in excluded],
        'prefix_representative': [row for row in representative if row['image_id'] not in excluded],
    }
    overlap = readiness['overlap']
    counts = {name: len(rows) for name, rows in populations.items()}
    require(counts == COUNTS and counts == {
        'primary': overlap['eligible_images'], 'representative': overlap['representatives'],
        'prefix': overlap['prefix_sensitivity_eligible_images'],
        'prefix_representative': overlap['prefix_sensitivity_representatives'],
    } and len(excluded) == overlap['prefix_sensitivity_excluded_images'], 'Population reconstruction differs')
    return populations


def verify_scores(witness, primary):
    require(witness.keys() == {'rows'}, 'Unexpected grading witness fields')
    rows = witness['rows']
    require([row['image_id'] for row in rows] == [row['image_id'] for row in primary],
            'Grading rows do not match the exact eligible photo order')
    require(all(row.keys() == {'image_id', 'logits', 'raw', 'calibrated'} for row in rows),
            'Unexpected grading row fields')
    logits = np.array([row['logits'] for row in rows], dtype=np.float32)
    require(logits.shape == (len(rows), 5) and np.isfinite(logits).all(), 'Invalid saved five-grade logits')
    scores, deltas = {}, {}
    for key, temperature in (('raw', 1.0), ('calibrated', MODEL['temperature'])):
        scaled = logits / np.float32(temperature)
        exps = np.exp(scaled - scaled.max(axis=1, keepdims=True))
        expected = exps / exps.sum(axis=1, keepdims=True)
        saved = np.array([row[key] for row in rows], dtype=float)
        require(saved.shape == expected.shape and np.isfinite(saved).all() and (saved >= 0).all()
                and (saved <= 1).all() and np.allclose(saved.sum(axis=1), 1.0, rtol=0, atol=2e-7),
                'Invalid saved probabilities')
        deltas[key] = float(np.max(np.abs(saved - expected)))
        require(deltas[key] <= 2e-7, 'Saved probabilities differ from logits and frozen temperature')
        scores[key] = saved
    require(np.array_equal(logits.argmax(1), scores['raw'].argmax(1))
            and np.array_equal(scores['raw'].argmax(1), scores['calibrated'].argmax(1)),
            'Positive temperature scaling changed a grade decision')
    return rows, scores, deltas


def binary_metrics(truth, risk):
    actual = truth >= 2
    predicted = risk >= .5
    matrix = np.zeros((2, 2), dtype=int)
    np.add.at(matrix, (actual.astype(int), predicted.astype(int)), 1)
    positives = int(actual.sum())
    negatives = len(actual) - positives
    ranks = rankdata(risk, method='average')
    auc = (ranks[actual].sum() - positives * (positives + 1) / 2) / (positives * negatives)
    order = np.argsort(-risk, kind='stable')
    ordered = risk[order]
    ends = np.flatnonzero(np.r_[ordered[:-1] != ordered[1:], True])
    cumulative = np.cumsum(actual[order])[ends]
    average_precision = np.dot(np.diff(cumulative, prepend=0), cumulative / (ends + 1)) / positives
    return {
        'auroc': float(auc), 'average_precision': float(average_precision), 'confusion': matrix.tolist(),
        'sensitivity': float(matrix[1, 1] / positives), 'specificity': float(matrix[0, 0] / negatives),
        'always_positive_accuracy': positives / len(actual),
    }


def grading_metrics(truth, probabilities):
    prediction = probabilities.argmax(axis=1)
    matrix = np.zeros((5, 5), dtype=int)
    np.add.at(matrix, (truth, prediction), 1)
    support, called, correct = matrix.sum(1), matrix.sum(0), matrix.diagonal()
    weights = np.square(np.arange(5)[:, None] - np.arange(5)[None, :])
    disagreement = float((weights * matrix).sum())
    expected_disagreement = float((weights * np.outer(support, called)).sum() / len(truth))
    confidence = probabilities.max(1)
    bins = np.searchsorted(np.arange(1, 10) / 10.0, confidence, side='right')
    residual = np.bincount(bins, weights=(truth == prediction) - confidence, minlength=10)
    true_probability = probabilities[np.arange(len(truth)), truth]
    one_hot = truth[:, None] == np.arange(5)
    return {
        'accuracy': float(correct.sum() / len(truth)), 'macro_f1': float((2 * correct / (support + called)).mean()),
        'quadratic_kappa': 1 - disagreement / expected_disagreement,
        'mean_absolute_grade_error': float(np.abs(truth - prediction).mean()),
        'undergrade_by_two_or_more': int((truth - prediction >= 2).sum()), 'confusion': matrix.tolist(),
        'recall_by_grade': {str(grade): float(value) for grade, value in enumerate(correct / support)},
        'nll': float(-np.log(np.clip(true_probability, 1e-12, 1.)).mean()),
        'brier': float(np.square(probabilities - one_hot).sum() / len(truth)),
        'ece': float(np.abs(residual).sum() / len(truth)),
        'binary': binary_metrics(truth, probabilities[:, 2:].sum(1)),
    }


def verify_aggregates(report, populations, scores):
    positions = {row['image_id']: index for index, row in enumerate(populations['primary'])}
    require(report.keys() == populations.keys(), 'Grading report population keys differ')
    for name, rows in populations.items():
        truth = np.array([row['dr_grade'] for row in rows], dtype=int)
        subset = [positions[row['image_id']] for row in rows]
        counts = np.bincount(truth, minlength=5)
        require((counts > 0).all(), 'Each audited FDR26 population needs all five source grades')
        expected = {'images': len(rows), 'grade_counts': {str(grade): int(n) for grade, n in enumerate(counts)}}
        expected.update({key: grading_metrics(truth, probability[subset]) for key, probability in scores.items()})
        match(expected, report[name], name)


def replay(paths, calibration, primary, saved):
    sys.path.insert(0, str(ROOT))
    import torch
    from qsart.models.checkpoint import load_checkpoint
    from qsart.training.data import image_to_tensor

    torch.set_num_threads(MODEL['cpu_threads'])
    model, checkpoint = load_checkpoint(paths['checkpoint'])
    metadata = checkpoint['metadata']
    require(metadata['task'] == 'dr' and metadata['heads'] == {'dr_grade': 5}
            and metadata['trained'] is True and metadata['data_purpose'] == 'real_research'
            and metadata.get('production_eligible', True) is True and metadata['model_version'] == MODEL['version']
            and metadata['image_size'] == MODEL['image_size'] and metadata['manifest_sha256'] == calibration['manifest_sha256']
            and metadata['split_fingerprints']['calibration'] == calibration['split_fingerprint'],
            'Checkpoint does not match the frozen first-pass grading protocol')
    by_id = {row['image_id']: row for row in saved}
    result = {}
    with ZipFile(paths['archive']) as bundle, torch.inference_mode():
        for grade in range(5):
            samples = sorted((row for row in primary if row['dr_grade'] == grade), key=lambda row: row['image_id'])[:2]
            require(len(samples) == 2, 'Two eligible replay samples per grade are required')
            deltas = []
            for row in samples:
                payload = bundle.read(row['member'])
                require(hashlib.sha256(payload).hexdigest() == row['sha256'], 'Replay JPEG digest differs')
                with Image.open(BytesIO(payload)) as image:
                    require(image.format == 'JPEG' and image.mode == 'RGB' and image.size == (row['width'], row['height']),
                            'Replay JPEG format or dimensions differ')
                    image.load()
                    tensor = image_to_tensor(image.convert('RGB'), metadata['image_size'],
                                             metadata['normalization_mean'], metadata['normalization_std'])
                actual = model(tensor.unsqueeze(0))['dr_grade'][0].numpy().astype(float)
                expected = np.array(by_id[row['image_id']]['logits'], dtype=float)
                delta = float(np.max(np.abs(actual - expected)))
                require(delta <= 1e-6 and actual.argmax() == expected.argmax(), 'Replay logits or grade decision differ')
                deltas.append(delta)
            result[str(grade)] = {'images': len(samples), 'max_absolute_logit_delta': max(deltas)}
    return result


def run():
    target = OUTPUT / 'independent_audit.json'
    require(not target.exists(), 'Preserve the existing grading audit receipt')
    evaluation = OUTPUT / 'evaluation.json'
    report = read(evaluation)
    witness_sha = digest(WITNESS)
    require(report['experiment_id'] == EXPERIMENT and report['witness_sha256'] == witness_sha,
            'Grading report identity or private witness hash differs')
    require(report['dataset'] == 'FDR26' and report['partition'] == 'external'
            and report['scope'] == 'external_alfediam_grade_code_diagnostic'
            and report['reference_grade_standard'] == 'ALFEDIAM' and report['binary_reference_minimum_grade'] == 2
            and report['binary_probability_threshold'] == .5 and report['patient_linkage_available'] is False,
            'Grading report scope or binary proxy differs')
    paths, readiness, calibration = verify_bindings(report)
    populations = reconstruct_populations(readiness, read(paths['readiness_witness']))
    rows, scores, probability_delta = verify_scores(read(WITNESS), populations['primary'])
    verify_aggregates(report['populations'], populations, scores)
    replay_result = replay(paths, calibration, populations['primary'], rows)
    receipt = {
        'audit_status': 'passed', 'experiment_id': EXPERIMENT, 'source_commit': report['source_commit'],
        'evaluation_sha256': digest(evaluation), 'auditor_sha256': digest(Path(__file__)),
        'private_witness_sha256': witness_sha, 'input_sha256': report['input_sha256'],
        'population_images': {name: len(part) for name, part in populations.items()},
        'population_probability_views_reconstructed': len(populations) * len(scores),
        'scored_images_verified': len(rows), 'max_absolute_probability_delta': probability_delta,
        'replay_images': sum(row['images'] for row in replay_result.values()), 'replay_by_grade': replay_result,
    }
    with target.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(receipt, indent=2, allow_nan=False) + '\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    run()
