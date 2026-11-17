"""Score the frozen DeepDRiD grading head on audited FDR26 photos."""

import argparse
from collections import defaultdict, deque
import hashlib
from io import BytesIO
import json
import math
from pathlib import Path
import subprocess
import sys
from zipfile import ZipFile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score
import torch

from qsart.evaluation import grade_metrics
from qsart.models.checkpoint import load_checkpoint, sha256_file
from qsart.training.data import image_to_tensor


ROOT = Path(__file__).resolve().parents[1]
SLUG = 'fdr26_grading_v1'
PROTOCOL = 'docs/research/FDR26_GRADING_V1_PROTOCOL.md'
SOURCES = (PROTOCOL, 'scripts/evaluate_fdr26_grading.py', 'scripts/audit_fdr26_grading.py',
           'qsart/models/checkpoint.py', 'qsart/models/networks.py', 'qsart/training/data.py',
           'qsart/preprocessing/fundus.py', 'qsart/evaluation.py')
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
COUNTS = {'primary': 2849, 'representative': 2742, 'prefix': 2761, 'prefix_representative': 2673}


def lock_source(commit):
    def git(*args):
        return subprocess.check_output(['git', *args], cwd=ROOT)

    if git('rev-parse', 'HEAD').decode().strip() != commit or git('status', '--porcelain'):
        raise ValueError('Run from the exact clean, committed source')
    sources = {}
    for name in SOURCES:
        data = (ROOT / name).read_bytes()
        if data != git('show', f'{commit}:{name}'):
            raise ValueError(f'Source bytes differ from the lock: {name}')
        sources[name] = hashlib.sha256(data).hexdigest()
    return sources


def inputs():
    paths = {name: ROOT / value[0] for name, value in INPUTS.items()}
    hashes = {name: sha256_file(path) for name, path in paths.items()}
    if any(hashes[name] != value[1] for name, value in INPUTS.items()):
        raise ValueError('Frozen readiness, release or model bytes differ')
    readiness, audit, witness = (json.loads(paths[name].read_bytes()) for name in
                                 ('readiness_report', 'readiness_audit', 'readiness_witness'))
    if (
        audit['audit_status'] != 'passed' or readiness['model_inference_performed']
        or audit['evaluation_sha256'] != hashes['readiness_report']
        or readiness['witness_sha256'] != audit['private_witness_sha256']
        or audit['private_witness_sha256'] != hashes['readiness_witness']
        or readiness['input_sha256']['archive'] != hashes['archive']
        or readiness['grade_standard'] != 'ALFEDIAM' or readiness['patient_linkage_available']
    ):
        raise ValueError('Readiness report and audit disagree')
    return paths, hashes, witness['rows']


def populations(rows):
    components, prefixes = defaultdict(list), defaultdict(list)
    queue = deque()
    visited = set()
    for i, row in enumerate(rows):
        components[row['component']].append(i)
        if row['dr_grade'] is not None:
            prefixes[row['image_id'][:8]].append(i)
        if not row['eligible']:
            queue.append(i)
            visited.add(i)
    while queue:
        row = rows[queue.popleft()]
        neighbors = components.pop(row['component'], [])
        if row['dr_grade'] is not None:
            neighbors = neighbors + prefixes.pop(row['image_id'][:8], [])
        for i in neighbors:
            if i not in visited:
                visited.add(i)
                queue.append(i)
    primary = [row for row in rows if row['eligible']]
    representative = [row for row in primary if row['representative']]
    prefix = [row for i, row in enumerate(rows) if i not in visited]
    prefix_rep = [row for row in prefix if row['representative']]
    result = dict(zip(COUNTS, (primary, representative, prefix, prefix_rep), strict=True))
    if (len(rows) != 3000 or len({row['image_id'] for row in rows}) != len(rows)
            or {name: len(part) for name, part in result.items()} != COUNTS
            or any(type(row['dr_grade']) is not int or row['dr_grade'] not in range(5)
                   or row['cross_candidate'] or row['label_conflict'] for row in primary)):
        raise ValueError('Evaluation populations differ from the readiness audit')
    return result


def frozen_model(paths):
    torch.set_num_threads(2)
    model, checkpoint = load_checkpoint(paths['checkpoint'])
    metadata = checkpoint['metadata']
    cal = json.loads(paths['calibration'].read_bytes())
    dr = cal['dr']
    temperature = dr['heads']['dr_grade']['temperature']
    if (metadata['data_purpose'] != 'real_research' or metadata['task'] != 'dr'
            or metadata['heads'] != {'dr_grade': 5} or metadata['image_size'] != 224
            or metadata.get('production_eligible', True) is not True
            or cal['data_purpose'] != 'real_research' or cal['method'] != 'temperature_scaling'
            or dr['checkpoint_sha256'] != INPUTS['checkpoint'][1] or dr['split'] != 'calibration'
            or dr['manifest_sha256'] != metadata['manifest_sha256']
            or dr['model_version'] != metadata['model_version']
            or dr['split_fingerprint'] != metadata['split_fingerprints']['calibration']
            or not math.isfinite(temperature) or temperature <= 0):
        raise ValueError('The frozen DeepDRiD model and temperature do not match')
    return model, metadata, temperature


def score(paths, primary):
    model, metadata, temperature = frozen_model(paths)
    saved = {}
    with ZipFile(paths['archive']) as archive, torch.inference_mode():
        for i, row in enumerate(primary, 1):
            image_bytes = archive.read(row['member'])
            if hashlib.sha256(image_bytes).hexdigest() != row['sha256']:
                raise ValueError('Archived image differs from audited bytes')
            with Image.open(BytesIO(image_bytes)) as image:
                tensor = image_to_tensor(image.convert('RGB'), metadata['image_size'],
                                         metadata['normalization_mean'], metadata['normalization_std'])
            logits = model(tensor.unsqueeze(0))['dr_grade']
            if logits.shape != (1, 5) or logits.dtype != torch.float32 or not torch.isfinite(logits).all():
                raise ValueError('Invalid frozen grading output')
            identity = row['image_id']
            saved[identity] = {'image_id': identity, 'logits': logits[0].tolist(),
                               'raw': logits.softmax(1)[0].tolist(),
                               'calibrated': (logits / temperature).softmax(1)[0].tolist()}
            if i % 500 == 0:
                print(f'Scored {i}/{len(primary)} photos', flush=True)
    return saved, {'version': metadata['model_version'], 'temperature': temperature,
                   'image_size': metadata['image_size'], 'cpu_threads': 2}


def metrics(y, p):
    native = grade_metrics(y, p)
    error = y - p.argmax(axis=1)
    truth = y >= 2
    risk = p[:, 2:].sum(axis=1)
    confusion = confusion_matrix(truth, risk >= .5, labels=[False, True])
    tn, fp, fn, tp = map(int, confusion.ravel())
    return {
        'accuracy': native['accuracy'], 'macro_f1': native['macro_f1'],
        'quadratic_kappa': native['quadratic_weighted_kappa'], 'confusion': native['confusion_matrix'],
        'recall_by_grade': {str(row['grade']): row['recall'] for row in native['per_class']},
        'mean_absolute_grade_error': float(np.abs(error).mean()),
        'undergrade_by_two_or_more': int((error >= 2).sum()),
        'nll': float(-np.log(p[np.arange(len(y)), y].clip(1e-12)).mean()),
        'brier': native['brier'], 'ece': native['ece'],
        'binary': {'auroc': float(roc_auc_score(truth, risk)),
                   'average_precision': float(average_precision_score(truth, risk)),
                   'sensitivity': tp / (tp + fn), 'specificity': tn / (tn + fp),
                   'confusion': confusion.tolist(), 'always_positive_accuracy': float(truth.mean())},
    }


def evaluate(part, saved):
    y = np.array([row['dr_grade'] for row in part], dtype=int)
    return {'images': len(part), 'grade_counts': {str(grade): int((y == grade).sum()) for grade in range(5)},
            **{key: metrics(y, np.array([saved[row['image_id']][key] for row in part], dtype=float))
               for key in ('raw', 'calibrated')}}


def encode(value):
    return (json.dumps(value, indent=2, allow_nan=False) + '\n').encode('utf-8')


def run(commit):
    private, public = ROOT / 'artifacts' / SLUG, ROOT / 'reports' / SLUG
    if private.exists() or public.exists():
        raise ValueError('Preserve existing grading outputs')
    sources = lock_source(commit)
    paths, hashes, readiness = inputs()
    parts = populations(readiness)
    scores, model = score(paths, parts['primary'])
    witness = encode({'rows': list(scores.values())})
    report = {
        'experiment_id': SLUG, 'dataset': 'FDR26', 'partition': 'external',
        'scope': 'external_alfediam_grade_code_diagnostic', 'source_commit': commit, 'source_sha256': sources,
        'input_sha256': hashes, 'witness_sha256': hashlib.sha256(witness).hexdigest(), 'model': model,
        'reference_grade_standard': 'ALFEDIAM', 'binary_reference_minimum_grade': 2,
        'binary_probability_threshold': .5, 'patient_linkage_available': False,
        'populations': {name: evaluate(part, scores) for name, part in parts.items()},
    }
    private.mkdir(parents=True)
    public.mkdir(parents=True)
    with (private / 'witness.json').open('xb') as stream:
        stream.write(witness)
    with (public / 'evaluation.json').open('xb') as stream:
        stream.write(encode(report))
    print(json.dumps({'scored': len(scores), 'populations': {name: len(rows) for name, rows in parts.items()}}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-commit', required=True)
    run(parser.parse_args().source_commit)
