"""Run the previously designated primary APTOS classifier on audited FDR26 photos."""

import argparse
import hashlib
from io import BytesIO
import json
from pathlib import Path
import subprocess
import sys
from zipfile import ZipFile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from PIL import Image
import torch

from qsart.data.external_evaluation import infer_logits
from qsart.models.checkpoint import load_checkpoint, sha256_file
from qsart.preprocessing.fundus import preprocess_image
from qsart.training.data import MEAN, STD
from qsart.training.finetune import _named_tensor_hash
from scripts.evaluate_fdr26_grading import INPUTS, encode, evaluate, populations
from scripts.evaluate_idrid import preflight


ROOT = Path(__file__).resolve().parents[1]
SLUG = 'fdr26_aptos_v1'
CONFIG = 'configs/idrid_external_v1.json'
PROTOCOL = 'docs/research/FDR26_APTOS_V1_PROTOCOL.md'
SOURCES = (PROTOCOL, CONFIG, 'scripts/evaluate_fdr26_aptos.py', 'scripts/audit_fdr26_aptos.py',
           'scripts/evaluate_fdr26_grading.py', 'scripts/audit_fdr26_grading.py', 'scripts/evaluate_idrid.py',
           'qsart/data/external_evaluation.py', 'qsart/models/checkpoint.py', 'qsart/models/networks.py',
           'qsart/training/finetune.py', 'qsart/training/data.py', 'qsart/preprocessing/fundus.py', 'qsart/evaluation.py')
DEEP_REPORT = 'reports/fdr26_grading_v1/evaluation.json'
DEEP_AUDIT = 'reports/fdr26_grading_v1/independent_audit.json'
DEEP_WITNESS = 'artifacts/fdr26_grading_v1/witness.json'
DEEP_BINDINGS = {
    DEEP_REPORT: '52ddaf00408c988601d11dcb6161e1aab6e2553ed769881521616b994a395455',
    DEEP_AUDIT: 'ec090dcf8de5bfa41b45d451fb656f12f6c5f7c3caec268b68c04cc67dcb28da',
    DEEP_WITNESS: 'f7871753b433a797a2c8c164b6916862957a772dda9135dbdd551deba04f249b',
}


def read(path):
    return json.loads((ROOT / path).read_bytes())


def lock(commit):
    def git(*args):
        return subprocess.check_output(['git', *args], cwd=ROOT)

    if git('rev-parse', 'HEAD').decode().strip() != commit or git('status', '--porcelain'):
        raise ValueError('Run from the clean locked source')
    bindings = {}
    for path in SOURCES:
        raw = (ROOT / path).read_bytes()
        if raw != git('show', f'{commit}:{path}'):
            raise ValueError(f'Changed execution source: {path}')
        bindings[path] = hashlib.sha256(raw).hexdigest()
    return bindings


def bound_inputs():
    expected_config = '63473f75e7404a6bed811b6abbec1fd6eb1099730b23dd40dd01a603ce433189'
    if sha256_file(ROOT / CONFIG) != expected_config:
        raise ValueError('Historical IDRiD configuration differs')
    config = read(CONFIG)
    (model,) = [row for row in config['models'] if row['model_id'] == config['primary_model']]
    if (model['model_id'] != 'finetune_seed2026' or model['training_seed'] != 2026
            or config['batch_size'] != 16 or config['cpu_threads'] != 4 or config['image_size'] != 224):
        raise ValueError('Unexpected historical source primary or inference path')
    paths = {CONFIG: expected_config, **DEEP_BINDINGS,
             **dict(INPUTS[k] for k in ('archive', 'readiness_report', 'readiness_audit', 'readiness_witness')),
             model['checkpoint']: model['checkpoint_sha256'],
             model['selection_scores']: model['selection_scores_sha256']}
    for key in (model['source_report'], config['preflight_cache'], config['preflight_runtime_manifest']):
        paths[key] = config['input_sha256'][key]
    if any(sha256_file(ROOT / path) != expected for path, expected in paths.items()):
        raise ValueError('A frozen comparison input differs')
    return paths, config, model


def score(rows, config, source):
    torch.set_num_threads(config['cpu_threads'])
    model, checkpoint = load_checkpoint(ROOT / source['checkpoint'])
    metadata = checkpoint['metadata']
    (source_run,) = [item for item in read(source['source_report'])['runs'] if item['training_seed'] == 2026]
    temperature = source['temperature']
    if (metadata['backbone'] != 'efficientnet_b0' or metadata['heads'] != {'dr_grade': 5}
            or metadata['image_size'] != 224 or metadata['data_purpose'] != 'real_research'
            or metadata['normalization_mean'] != MEAN or metadata['normalization_std'] != STD
            or checkpoint['epoch'] != 3 or source_run['calibration']['temperature'] != temperature):
        raise ValueError('Frozen APTOS predictor or temperature differs')
    state = _named_tensor_hash(model.state_dict().items())
    runtime = pd.read_csv(ROOT / config['preflight_runtime_manifest'])
    cache = torch.load(ROOT / config['preflight_cache'], map_location='cpu', weights_only=True)
    regression = preflight(config, source, model, cache, runtime)
    print(f"APTOS source replay passed on {regression['images']} photos", flush=True)
    del cache, runtime
    scored = {}
    with ZipFile(ROOT / INPUTS['archive'][0]) as archive:
        for start in range(0, len(rows), 16):
            group = rows[start:start + 16]
            pixels = []
            for row in group:
                payload = archive.read(row['member'])
                if hashlib.sha256(payload).hexdigest() != row['sha256']:
                    raise ValueError('Archived photo differs from its audited hash')
                with Image.open(BytesIO(payload)) as image:
                    pixels.append(np.asarray(preprocess_image(image, image_size=224, crop=True)).copy())
            logits = infer_logits(model, torch.from_numpy(np.stack(pixels)), batch_size=16)
            raw = logits.double().softmax(1)
            calibrated = (logits.double() / temperature).softmax(1)
            for row, logit, first, second in zip(group, logits, raw, calibrated, strict=True):
                identity = row['image_id']
                scored[identity] = {'image_id': identity, 'logits': logit.tolist(),
                                    'raw': first.tolist(), 'calibrated': second.tolist()}
            if start % 512 == 0 and start:
                print(f'Scored {len(scored)}/{len(rows)} photos', flush=True)
    if _named_tensor_hash(model.state_dict().items()) != state:
        raise ValueError('Frozen APTOS predictor state changed')
    return scored, {'version': metadata['model_version'], 'temperature': temperature}, regression


def paired(part, aptos, deep):
    y = np.array([row['dr_grade'] for row in part])
    correctness = []
    for model in (aptos, deep):
        p = np.array([model[row['image_id']]['calibrated'] for row in part])
        correctness.append(y == p.argmax(axis=1))
    a, d = correctness
    return {'both_correct': int((a & d).sum()), 'aptos_only': int((a & ~d).sum()),
            'deepdrid_only': int((~a & d).sum()), 'neither': int((~a & ~d).sum())}


def run(commit):
    private, public = ROOT / 'artifacts' / SLUG, ROOT / 'reports' / SLUG
    if private.exists() or public.exists():
        raise ValueError('Preserve existing output')
    sources = lock(commit)
    inputs, config, source = bound_inputs()
    parts = populations(read(INPUTS['readiness_witness'][0])['rows'])
    deep_report, deep_audit, deep = read(DEEP_REPORT), read(DEEP_AUDIT), read(DEEP_WITNESS)['rows']
    if (deep_audit['audit_status'] != 'passed' or deep_audit['evaluation_sha256'] != inputs[DEEP_REPORT]
            or deep_report['witness_sha256'] != inputs[DEEP_WITNESS]
            or [row['image_id'] for row in deep] != [row['image_id'] for row in parts['primary']]):
        raise ValueError('Saved DeepDRiD outcomes differ from the audited cohort')
    deep = {row['image_id']: row for row in deep}
    for name, rows in parts.items():
        if evaluate(rows, deep) != deep_report['populations'][name]:
            raise ValueError('Saved DeepDRiD metrics did not replay')
    aptos, model, regression = score(parts['primary'], config, source)
    populations_report = {}
    for name, rows in parts.items():
        baseline = deep_report['populations'][name]
        new = evaluate(rows, aptos)
        populations_report[name] = {
            'images': new['images'], 'grade_counts': new['grade_counts'],
            'aptos': {key: new[key] for key in ('raw', 'calibrated')},
            'deepdrid': {key: baseline[key] for key in ('raw', 'calibrated')},
            'paired': paired(rows, aptos, deep),
        }
    witness = encode({'rows': list(aptos.values())})
    report = {
        'experiment_id': SLUG, 'dataset': 'FDR26', 'partition': 'external',
        'scope': 'exploratory_paired_frozen_grading_on_reused_fdr26',
        'source_commit': commit, 'source_sha256': sources, 'input_sha256': inputs,
        'witness_sha256': hashlib.sha256(witness).hexdigest(), 'model': model, 'source_replay': regression,
        'populations': populations_report,
    }
    private.mkdir(parents=True)
    public.mkdir(parents=True)
    with (private / 'witness.json').open('xb') as stream:
        stream.write(witness)
    with (public / 'evaluation.json').open('xb') as stream:
        stream.write(encode(report))
    print(json.dumps({'scored': len(aptos), 'population_counts': {key: len(part) for key, part in parts.items()}}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-commit', required=True)
    run(parser.parse_args().source_commit)
