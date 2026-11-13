"""Reconstruct the paired FDR26 comparison and replay the frozen APTOS predictor."""

import hashlib
from io import BytesIO
import json
from pathlib import Path
import re
import subprocess
import sys
from zipfile import ZipFile

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.audit_fdr26_grading import (digest, grading_metrics, match, read, reconstruct_populations,
                                         require, verify_scores)


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = 'fdr26_aptos_v1'
PUBLIC = ROOT / 'reports' / EXPERIMENT
WITNESS = ROOT / 'artifacts' / EXPERIMENT / 'witness.json'
READINESS_REPORT = 'reports/fdr26_readiness_v1/evaluation.json'
READINESS_AUDIT = 'reports/fdr26_readiness_v1/independent_audit.json'
READINESS_WITNESS = 'artifacts/fdr26_readiness_v1/witness.json'
DR_REPORT = 'reports/fdr26_grading_v1/evaluation.json'
DR_AUDIT = 'reports/fdr26_grading_v1/independent_audit.json'
DR_WITNESS = 'artifacts/fdr26_grading_v1/witness.json'
APTOS_REPORT = 'reports/aptos_finetune_v1/evaluation.json'
IDRID_CONFIG = 'configs/idrid_external_v1.json'
ARCHIVE = 'data/raw/fdr26_v3/Fundus_FDR26.zip'
CHECKPOINT = 'checkpoints/aptos_finetune_v1/seed2026/best.pt'
APTOS_SELECTION = 'artifacts/aptos_finetune_v1/scores/seed2026/validation_scores.csv'
APTOS_RUNTIME = 'artifacts/aptos_finetune_v1/runtime_development.csv'
APTOS_CACHE = ('artifacts/aptos_finetune_v1/inputs/'
               'canonical_uint8_2e091d56319a2ca27ee87e4e7e75f9e4c6913fe93bb986ea531ed93f82f78ec3.pt')
INPUT_HASHES = {
    READINESS_REPORT: 'b9692d9db75ff0f8113f348e944ca62d64e699335f773c2c900852d513c1e2b9',
    READINESS_AUDIT: '4d16a03e781f501a1aa52fee9cf6958605e86599be483c83e1413f02e636fae5',
    READINESS_WITNESS: '556e47ad165f76e5586c2936423b2a933985ff266c69135706c9c8f9ba8192b2',
    DR_REPORT: '52ddaf00408c988601d11dcb6161e1aab6e2553ed769881521616b994a395455',
    DR_AUDIT: 'ec090dcf8de5bfa41b45d451fb656f12f6c5f7c3caec268b68c04cc67dcb28da',
    DR_WITNESS: 'f7871753b433a797a2c8c164b6916862957a772dda9135dbdd551deba04f249b',
    APTOS_REPORT: '8c6f50e93db412fc845600a4e368094c46fce041109806e37819dcd6bb70c5f4',
    IDRID_CONFIG: '63473f75e7404a6bed811b6abbec1fd6eb1099730b23dd40dd01a603ce433189',
    ARCHIVE: '6c5517d60f24cbd310a1ab0607b8c90dbb4b88b61aee57840fc2e823b753dadb',
    CHECKPOINT: '53cdad8295aca6941431a3c321815c10ec3c9fdd5544abc0d8d83094863ead1c',
    APTOS_SELECTION: 'e4deeee308ebcaf0663d05305e5493c7f21d7ad47182dedcc07462ff122b4c04',
    APTOS_RUNTIME: 'dff7122d9cb496f3610e5dca0fab6acdafb6e1309f548f299287c5041cee963d',
    APTOS_CACHE: '2ac8de257140cea3cbcd8f10d3140f984f63416c7daf1e2af2e6a74e94df510d',
}
SOURCE_FILES = {
    'docs/research/FDR26_APTOS_V1_PROTOCOL.md', 'scripts/evaluate_fdr26_aptos.py', 'scripts/audit_fdr26_aptos.py',
    IDRID_CONFIG, 'scripts/audit_fdr26_grading.py', 'scripts/evaluate_fdr26_grading.py', 'scripts/evaluate_idrid.py',
    'qsart/data/external_evaluation.py', 'qsart/training/finetune.py', 'qsart/preprocessing/fundus.py',
    'qsart/models/checkpoint.py', 'qsart/models/networks.py', 'qsart/evaluation.py', 'qsart/training/data.py',
}
MODEL = {'version': 'aptos-finetune-v1-efficientnet_b0-seed2026-epoch4', 'temperature': 1.7353179819929263}
BATCH_SIZE = 16
THREADS = 4
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def verify_bindings(report):
    commit, sources = report['source_commit'], report['source_sha256']
    require(re.fullmatch(r'[0-9a-f]{40}', commit) and sources.keys() == SOURCE_FILES,
            'Expected locked paired source bindings')
    for name, expected in sources.items():
        raw = (ROOT / name).read_bytes()
        committed = subprocess.check_output(['git', 'show', f'{commit}:{name}'], cwd=ROOT)
        require(raw == committed and hashlib.sha256(raw).hexdigest() == expected, f'Locked source differs: {name}')
    require(report['input_sha256'] == INPUT_HASHES and all(digest(ROOT / name) == sha for name, sha in INPUT_HASHES.items()),
            'Archived inputs differ from the locked comparison')
    readiness, readiness_audit, prior, prior_audit = (
        read(ROOT / name) for name in (READINESS_REPORT, READINESS_AUDIT, DR_REPORT, DR_AUDIT)
    )
    require(readiness_audit['audit_status'] == prior_audit['audit_status'] == 'passed'
            and readiness_audit['evaluation_sha256'] == INPUT_HASHES[READINESS_REPORT]
            and readiness_audit['private_witness_sha256'] == readiness['witness_sha256'] == INPUT_HASHES[READINESS_WITNESS]
            and readiness['input_sha256']['archive'] == INPUT_HASHES[ARCHIVE]
            and prior_audit['evaluation_sha256'] == INPUT_HASHES[DR_REPORT]
            and prior_audit['private_witness_sha256'] == prior['witness_sha256'] == INPUT_HASHES[DR_WITNESS]
            and prior['input_sha256']['readiness_witness'] == INPUT_HASHES[READINESS_WITNESS],
            'Published readiness and DeepDRiD audits do not bind the comparison inputs')
    source, config = read(ROOT / APTOS_REPORT), read(ROOT / IDRID_CONFIG)
    primary = [model for model in config['models'] if model['model_id'] == config['primary_model']]
    selected = [run for run in source['runs'] if run['training_seed'] == source['summary']['primary_training_seed']]
    require(len(primary) == len(selected) == 1, 'Expected one originally declared APTOS primary')
    binding, run = primary[0], selected[0]
    require(binding['training_seed'] == run['training_seed'] == 2026 and run['selected_epoch'] == 4
            and binding['checkpoint'] == CHECKPOINT
            and binding['checkpoint_sha256'] == run['checkpoint_sha256'] == INPUT_HASHES[CHECKPOINT]
            and binding['temperature'] == run['calibration']['temperature'] == MODEL['temperature']
            and binding['selection_scores'] == APTOS_SELECTION
            and binding['selection_scores_sha256'] == INPUT_HASHES[APTOS_SELECTION]
            and config['preflight_cache'] == APTOS_CACHE and config['preflight_runtime_manifest'] == APTOS_RUNTIME
            and config['input_sha256'][APTOS_CACHE] == INPUT_HASHES[APTOS_CACHE]
            and config['input_sha256'][APTOS_RUNTIME] == INPUT_HASHES[APTOS_RUNTIME]
            and config['batch_size'] == BATCH_SIZE and config['cpu_threads'] == THREADS
            and run['model_version'] == source['model_version'] == MODEL['version'],
            'APTOS source report or earlier primary designation differs')
    match(MODEL, report['model'], 'APTOS model')
    preflight = report['source_replay']
    require(preflight['images'] == 10 and preflight['argmax_predictions_match'] is True
            and preflight['tolerance'] == 1e-5 and 0 <= preflight['raw_probability_max_abs_delta'] <= 1e-5,
            'Recorded source development preflight did not pass')
    return readiness, prior


def verify_aptos_scores(witness, primary):
    require(witness.keys() == {'rows'}, 'Unexpected APTOS witness fields')
    rows = witness['rows']
    require([row['image_id'] for row in rows] == [row['image_id'] for row in primary],
            'APTOS scores do not follow the frozen primary photo order')
    require(all(row.keys() == {'image_id', 'logits', 'raw', 'calibrated'} for row in rows),
            'Unexpected APTOS score row fields')
    logits = np.array([row['logits'] for row in rows], dtype=float)
    require(logits.shape == (len(primary), 5) and np.isfinite(logits).all()
            and np.array_equal(logits, logits.astype(np.float32)), 'Invalid float32 APTOS logits')
    scores, deltas = {}, {}
    for key, temperature in (('raw', 1.0), ('calibrated', MODEL['temperature'])):
        scaled = logits / temperature
        exp = np.exp(scaled - scaled.max(axis=1, keepdims=True))
        expected = exp / exp.sum(axis=1, keepdims=True)
        saved = np.array([row[key] for row in rows], dtype=float)
        require(saved.shape == expected.shape and np.isfinite(saved).all() and (saved >= 0).all()
                and (saved <= 1).all() and np.allclose(saved.sum(axis=1), 1.0, rtol=0, atol=2e-12),
                'Invalid saved APTOS probabilities')
        deltas[key] = float(np.max(np.abs(saved - expected)))
        require(deltas[key] <= 2e-12 and np.array_equal(saved.argmax(1), logits.argmax(1)),
                'APTOS probabilities differ from the frozen logits and temperature')
        scores[key] = saved
    return logits, scores, deltas


def verify_aggregates(report, prior, populations, aptos, deepdrid):
    require(report.keys() == prior.keys() == populations.keys(), 'Paired report populations differ')
    positions = {row['image_id']: index for index, row in enumerate(populations['primary'])}
    cells = 0
    for name, rows in populations.items():
        truth = np.array([row['dr_grade'] for row in rows], dtype=int)
        subset = [positions[row['image_id']] for row in rows]
        counts = {str(grade): int(n) for grade, n in enumerate(np.bincount(truth, minlength=5))}
        aptos_correct = aptos['calibrated'][subset].argmax(1) == truth
        deepdrid_correct = deepdrid['calibrated'][subset].argmax(1) == truth
        joint = np.bincount(2 * aptos_correct.astype(int) + deepdrid_correct, minlength=4)
        paired = dict(zip(('neither', 'deepdrid_only', 'aptos_only', 'both_correct'), map(int, joint), strict=True))
        old = {'images': len(rows), 'grade_counts': counts,
               **{key: grading_metrics(truth, probabilities[subset]) for key, probabilities in deepdrid.items()}}
        match(old, prior[name], f'archived DeepDRiD {name}')
        require(report[name]['deepdrid'] == {key: prior[name][key] for key in ('raw', 'calibrated')},
                'DeepDRiD metrics differ from the published aggregate')
        match({'images': len(rows), 'grade_counts': counts, 'paired': paired,
               'aptos': {key: grading_metrics(truth, probabilities[subset]) for key, probabilities in aptos.items()}},
              report[name], name)
        cells += len(paired)
    return cells


def replay(primary, logits):
    import torch
    from qsart.models.checkpoint import load_checkpoint
    from qsart.preprocessing import preprocess_image

    torch.set_num_threads(THREADS)
    model, checkpoint = load_checkpoint(ROOT / CHECKPOINT)
    metadata = checkpoint['metadata']
    require(metadata['model_version'] == MODEL['version'] and checkpoint['epoch'] == 3
            and metadata['seed'] == 2026 and metadata['data_purpose'] == 'real_research'
            and metadata['task'] == 'dr' and metadata['heads'] == {'dr_grade': 5}
            and metadata['image_size'] == 224 and metadata['normalization_mean'] == MEAN
            and metadata['normalization_std'] == STD and metadata['backbone'] == 'efficientnet_b0',
            'Replayed APTOS checkpoint metadata differs')
    rgb = []
    with ZipFile(ROOT / ARCHIVE) as bundle:
        for row in primary[:BATCH_SIZE]:
            payload = bundle.read(row['member'])
            require(hashlib.sha256(payload).hexdigest() == row['sha256'], 'Replay photo SHA256 differs')
            with Image.open(BytesIO(payload)) as image:
                require(image.format == 'JPEG' and image.mode == 'RGB'
                        and image.size == (row['width'], row['height']), 'Replay JPEG format or dimensions differ')
                rgb.append(np.array(preprocess_image(image, image_size=224, crop=True), dtype=np.float32))
    batch = torch.from_numpy(np.stack(rgb) / 255.0).permute(0, 3, 1, 2)
    batch = (batch - torch.tensor(MEAN)[None, :, None, None]) / torch.tensor(STD)[None, :, None, None]
    model.to(memory_format=torch.channels_last).eval()
    with torch.inference_mode():
        actual = model(batch.contiguous(memory_format=torch.channels_last))['dr_grade'].numpy()[:10]
    expected = logits[:10]
    maximum = float(np.abs(actual - expected).max())
    require(maximum <= 1e-6 and np.array_equal(actual.argmax(1), expected.argmax(1)), 'APTOS batch replay differs')
    return {'replay_images': len(actual), 'replay_batch_size': len(batch), 'max_absolute_logit_delta': maximum}


def run():
    target = PUBLIC / 'independent_audit.json'
    require(not target.exists(), 'Preserve the existing paired audit receipt')
    evaluation = PUBLIC / 'evaluation.json'
    report = read(evaluation)
    witness_sha = digest(WITNESS)
    require(report['experiment_id'] == EXPERIMENT and report['witness_sha256'] == witness_sha,
            'Paired report identity or APTOS witness digest differs')
    require(report['dataset'] == 'FDR26' and report['partition'] == 'external'
            and report['scope'] == 'exploratory_paired_frozen_grading_on_reused_fdr26', 'Paired report scope differs')
    readiness, prior = verify_bindings(report)
    populations = reconstruct_populations(readiness, read(ROOT / READINESS_WITNESS))
    primary = populations['primary']
    _, deepdrid, deepdrid_delta = verify_scores(read(ROOT / DR_WITNESS), primary)
    logits, aptos, aptos_delta = verify_aptos_scores(read(WITNESS), primary)
    cells = verify_aggregates(report['populations'], prior['populations'], populations, aptos, deepdrid)
    replay_result = replay(primary, logits)
    receipt = {
        'audit_status': 'passed', 'experiment_id': EXPERIMENT, 'source_commit': report['source_commit'],
        'evaluation_sha256': digest(evaluation), 'auditor_sha256': digest(Path(__file__)),
        'private_witness_sha256': witness_sha, 'input_sha256': report['input_sha256'],
        'population_images': {name: len(rows) for name, rows in populations.items()},
        'scores_verified_per_model': len(primary), 'paired_correctness_cells_verified': cells,
        'population_probability_views_reconstructed': len(populations) * (len(aptos) + len(deepdrid)),
        'max_absolute_probability_delta': {'aptos': aptos_delta, 'deepdrid': deepdrid_delta},
        **replay_result,
    }
    with target.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(receipt, indent=2, allow_nan=False) + '\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    run()
