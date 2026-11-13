"""Execute the predeclared selection-only, fixed model-input robustness study."""
from collections import Counter
from importlib import metadata
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, features

from qsart.evaluation import grade_metrics
from qsart.evaluation_robustness import (CONDITIONS, CONDITION_PARAMETERS, apply_degradation, canonical_rgb,
                                        development_selection, paired_patient_bootstrap, score_canonical)
from qsart.inference.conformal_sidecar import ConformalSidecar
from qsart.inference.engine import InferenceEngine
from qsart.models.checkpoint import sha256_file
from qsart.quality_baselines import binary_quality_metrics
from qsart.training.data import read_manifest
from qsart.triage_v2 import pointwise_action
from qsart.uncertainty.conformal import coverage_metrics


MANIFEST_SHA = '45e9f2d34ed54a12056b57a808d03cfaf29ccc970d148e9eb107b79cf5ac3e0d'
ARTIFACT_SHA = {
    'dr': 'f530056c27f83dbff8bba4ebb481cf7a9d82289947c4eda655a69ed22685a527',
    'quality': 'a600872ce8e2b256205350a5ef6c99b77d71f679d217c20c26810a3ed76eff43',
    'calibration': '674821e445490c209fab99e56b0e9915ac406ca4ec8694e5efdbed63e456a737',
    'conformal': '15a98f6bfe93f81156c5592701995d08796dd889e3a5e43bcbde7ab5aa61c221',
}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8', newline='\n')


def rgb_sha(image):
    return hashlib.sha256(np.asarray(image, dtype=np.uint8).tobytes()).hexdigest()


def observation_vectors(rows, truth):
    probabilities = np.array([row['grade_probabilities'] for row in rows])
    return {
        'accuracy': (probabilities.argmax(1) == truth).astype(float),
        'nll': -np.log(probabilities[np.arange(len(truth)), truth].clip(1e-12)),
        'entropy': np.array([row['uncertainty']['entropy'] for row in rows]),
        'poor_quality_probability': np.array([row['quality']['poor_probability'] for row in rows]),
        'lac_image_coverage': np.array([int(label) in row['conformal']['baseline_lac_set']
                                      for label, row in zip(truth, rows, strict=True)], dtype=float),
        'lac_mean_set_size': np.array([len(row['conformal']['baseline_lac_set']) for row in rows], dtype=float),
        'referable_prediction': (probabilities[:, 2:].sum(1) >= .5).astype(float),
    }


def condition_metrics(rows, truth, quality_truth, groups, *, clean=False):
    p = np.array([row['grade_probabilities'] for row in rows])
    vectors = observation_vectors(rows, truth)
    positive = truth >= 2
    predicted = vectors['referable_prediction'].astype(bool)
    referable = {
        'threshold': .5, 'positive_images': int(positive.sum()),
        'negative_images': int((~positive).sum()),
        'false_negatives': int((positive & ~predicted).sum()),
        'sensitivity': float(predicted[positive].mean()) if positive.any() else None,
        'specificity': float((~predicted[~positive]).mean()) if (~positive).any() else None,
    }
    conformal = {method: coverage_metrics([row['conformal'][field] for row in rows], truth, groups=groups)
                 for method, field in (('lac', 'baseline_lac_set'), ('ordinal_greedy', 'prediction_set'))}
    actions = Counter(pointwise_action(row['grade_probabilities'], 1-row['quality']['poor_probability'],
                        row['uncertainty']['entropy'], conformal_set=row['conformal_set'])[0].value for row in rows)
    strata = []
    for natural in (0, 1):
        for grade in range(5):
            mask = (quality_truth == natural) & (truth == grade)
            strata.append({'original_natural_quality': 'poor' if natural == 0 else 'good',
                'grade': grade, 'images': int(mask.sum()), 'patients': len(np.unique(groups[mask])),
                'accuracy': float(vectors['accuracy'][mask].mean()) if mask.any() else None,
                'mean_entropy': float(vectors['entropy'][mask].mean()) if mask.any() else None,
                'mean_poor_quality_probability': float(vectors['poor_quality_probability'][mask].mean())
                    if mask.any() else None})
    result = {'grading': {**grade_metrics(truth, p), 'nll': float(vectors['nll'].mean())},
              'referable': referable, 'mean_entropy': float(vectors['entropy'].mean()),
              'mean_poor_quality_probability': float(vectors['poor_quality_probability'].mean()),
              'conformal': conformal, 'pointwise_guarded_action_counts': dict(actions),
              'original_natural_quality_by_severity': strata}
    if clean:
        result['natural_quality_classification'] = binary_quality_metrics(quality_truth,
            np.array([row['quality']['probabilities'] for row in rows]))
    return result, vectors


def make_figures(report, output):
    labels = ['Clean', 'Blur 1', 'Blur 2', 'Exposure .5', 'Exposure 1.5', 'JPEG 50', 'JPEG 20']
    conditions = report['conditions']
    paths = [('grading', 'accuracy'), ('referable', 'sensitivity'), ('mean_poor_quality_probability',),
             ('mean_entropy',), ('conformal', 'lac', 'marginal_coverage'), ('conformal', 'lac', 'mean_set_size')]
    titles = ['DR accuracy', 'Referable sensitivity (risk >= .5)', 'Mean predicted poor-quality probability',
              'Mean normalized DR entropy', 'LAC image coverage (alpha .10)', 'LAC mean set size']
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for axis, path, title in zip(axes.flat, paths, titles, strict=True):
        values = []
        for name in CONDITIONS:
            value = conditions[name]
            for key in path:
                value = value[key]
            values.append(value)
        axis.bar(labels, values, color=['#35546c']+['#468e9c']*6)
        axis.set_title(title, fontsize=11)
        axis.tick_params(axis='x', labelrotation=40, labelsize=9)
        axis.set_ylim(0, 5 if path[-1] == 'mean_set_size' else 1)
        axis.grid(axis='y', alpha=.2)
    fig.suptitle('QSART fixed model-input stress study | 168 selection images / 42 patients\n'
                 'Development diagnostics; original disease labels; fixed model/calibration', fontsize=14)
    fig.savefig(output/'degradation_summary.png', dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    series = [('Gaussian blur radius (input pixels)', ('clean', 'blur_1', 'blur_2'), ('0', '1', '2')),
              ('sRGB intensity multiplier', ('exposure_0_5', 'clean', 'exposure_1_5'), ('0.5', '1.0', '1.5')),
              ('JPEG condition', ('clean', 'jpeg_50', 'jpeg_20'), ('No JPEG', 'Q50', 'Q20'))]
    for axis, (title, names, ticks) in zip(axes, series, strict=True):
        axis.plot(ticks, [conditions[name]['mean_entropy'] for name in names], 'o-', label='DR entropy')
        axis.plot(ticks, [conditions[name]['mean_poor_quality_probability'] for name in names],
                  's-', label='Predicted poor quality')
        axis.set_xlabel(title)
        axis.set_ylim(0, 1)
        axis.grid(alpha=.2)
        axis.legend(fontsize=8)
    fig.suptitle('Fixed-model response to controlled input degradation | development selection only')
    fig.savefig(output/'degradation_curves.png', dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)
    for axis, natural in zip(axes, ('poor', 'good'), strict=True):
        values = np.full((len(CONDITIONS), 5), np.nan)
        counts = np.zeros_like(values, dtype=int)
        for i, name in enumerate(CONDITIONS):
            for row in conditions[name]['original_natural_quality_by_severity']:
                if row['original_natural_quality'] == natural:
                    values[i, row['grade']] = row['accuracy'] if row['accuracy'] is not None else np.nan
                    counts[i, row['grade']] = row['images']
        artist = axis.imshow(np.ma.masked_invalid(values), vmin=0, vmax=1, cmap='Blues', aspect='auto')
        axis.set_xticks(range(5), [f'Grade {i}' for i in range(5)])
        axis.set_yticks(range(len(labels)), labels)
        axis.set_title(f'Original natural quality: {natural}')
        for i in range(len(labels)):
            for j in range(5):
                label = 'Absent' if not counts[i, j] else f'{values[i, j]:.2f}\nn={counts[i, j]}'
                axis.text(j, i, label, ha='center', va='center', fontsize=8,
                          color='white' if values[i, j] > .6 else 'black')
        fig.colorbar(artist, ax=axis, label='DR accuracy')
    fig.suptitle('Quality × severity strata use original natural annotations\n'
                 'Corrupted images have no new quality ground-truth labels')
    fig.savefig(output/'natural_quality_severity.png', dpi=160)
    plt.close(fig)


def run(args):
    started = time.perf_counter()
    if args.output.exists() or args.private.exists():
        raise ValueError('Refusing existing experiment directories; choose fresh public/private paths')
    if 'artifacts' not in args.private.resolve().parts or 'reports' not in args.output.resolve().parts:
        raise ValueError('Private cache must stay under artifacts; public output must stay under reports')
    if sha256_file(args.manifest) != MANIFEST_SHA:
        raise ValueError('Frozen development manifest checksum differs from predeclared protocol')
    frame = read_manifest(args.manifest)
    if set(frame.split) != {'train', 'validation', 'calibration'}:
        raise ValueError('Only the fixed development manifest is allowed')
    selected = development_selection(frame)
    if len(selected) != 168 or selected.subject_id.nunique() != 42:
        raise ValueError('Frozen selection population differs from 168 images / 42 patients')
    paths = {'dr': args.selected/'dr.pt', 'quality': args.selected/'quality.pt',
             'calibration': args.selected/'calibration.json', 'conformal': args.conformal}
    actual_sha = {name: sha256_file(path) for name, path in paths.items()}
    if actual_sha != ARTIFACT_SHA:
        raise ValueError('A predeclared model/calibration/conformal artifact differs')
    engine = InferenceEngine(paths['dr'], paths['quality'], paths['calibration'])
    cp = ConformalSidecar(paths['conformal'], dr_checkpoint=paths['dr'], quality_checkpoint=paths['quality'],
                          calibration=paths['calibration'])
    if cp.bundle['manifest_sha256'] != MANIFEST_SHA:
        raise ValueError('Conformal manifest identity differs')
    truth = selected.dr_grade.to_numpy(dtype=int)
    quality_truth = selected.quality_grade.to_numpy(dtype=int)
    groups = selected.subject_id.to_numpy()
    ordinary, canonical = [], []
    for start in range(0, len(selected), 8):
        originals = []
        for row in selected.iloc[start:start+8].itertuples():
            with Image.open(row.image_path) as image:
                original = image.convert('RGB').copy()
            originals.append(original)
            canonical.append(canonical_rgb(original, 224))
        ordinary.extend(engine.predict_images(originals, second_pass_policy='none', include_uncalibrated=True))
    original_defect_heads = tuple(ordinary[0]['quality']['defects'])
    report = {'schema_version': 1, 'experiment_id': 'controlled_degradation_v1', 'spec_version': '1.0',
              'declaration_date': '2026-10-22', 'dataset': 'DeepDRiD', 'partition': 'validation',
              'scope': 'development_model_input_stress', 'images': len(selected), 'patients': len(np.unique(groups)),
              'model_version': engine.model_version, 'artifact_sha256': actual_sha, 'manifest_sha256': MANIFEST_SHA,
              'protocol_sha256': sha256_file(args.protocol),
              'execution_source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
              'source_sha256': {str(path): sha256_file(path) for path in (
                  Path('scripts/run_controlled_degradation.py'), Path('qsart/evaluation_robustness.py'),
                  Path('qsart/preprocessing/fundus.py'), Path('qsart/training/data.py'),
                  Path('qsart/inference/engine.py'), Path('qsart/inference/conformal_sidecar.py'),
                  Path('qsart/uncertainty/conformal.py'), Path('qsart/evaluation.py'), Path('qsart/triage_v2.py'))},
              'package_versions': {name: metadata.version(name) for name in
                  ('numpy', 'pandas', 'Pillow', 'torch', 'torchvision', 'scikit-learn', 'matplotlib')},
              'jpeg_codec': features.version_codec('jpg'),
              'preprocessing': 'Original crop/letterbox/resize exactly once; perturb224RGB; direct normalization.',
              'condition_parameters': CONDITION_PARAMETERS,
              'conditions': {}, 'paired_changes': {},
              'limitations': [
                  'Fixed development selection set informed model choice; not an independent confirmation.',
                  'Synthetic model-input perturbations are separate from naturally annotated quality defects.',
                  'Original disease labels describe underlying disease; corruption can hide diagnostic evidence.',
                  'No recalibration, threshold tuning, new model selection or conditional second pass.',
                  'Shifted conformal coverage is empirical; full five-grade sets provide no useful resolution.',
                  'Bootstrap intervals are paired patient-cluster descriptions without multiplicity adjustment.',
                  'Pointwise recommendations do not certify capacity feasibility or real deferral outcomes.',
                  'No external/camera-domain generalization, prospective utility or clinical safety is established.']}
    cache = {'scope': report['scope'], 'partition': 'validation', 'original_subject_ids': groups.tolist(),
             'original_grades': truth.tolist(), 'original_natural_quality': quality_truth.tolist(),
             'canonical_rgb_sha256': [rgb_sha(image) for image in canonical], 'conditions': {}}
    clean_vectors = None
    for condition in CONDITIONS:
        images = [apply_degradation(image, condition) for image in canonical]
        rows = []
        for start in range(0, len(images), 8):
            rows.extend(score_canonical(engine, images[start:start+8]))
        if condition == 'clean':
            errors = []
            for new, old in zip(rows, ordinary, strict=True):
                pairs = [(new['grade_probabilities'], old['grade_probabilities']),
                         (new['quality']['probabilities'], old['quality']['probabilities']),
                         (new['uncalibrated']['grade_probabilities'], old['uncalibrated']['grade_probabilities'])]
                pairs.extend((new['quality']['defects'][head], old['quality']['defects'][head])
                             for head in original_defect_heads)
                errors.extend(float(np.max(np.abs(np.asarray(a)-np.asarray(b)))) for a, b in pairs)
            maximum = max(errors)
            if maximum > 1e-6:
                raise ValueError(f'Clean research/ordinary inference mismatch: {maximum}')
            report['clean_equivalence'] = {'images': len(rows), 'maximum_absolute_probability_error': maximum,
                'tolerance': 1e-6, 'heads': ['dr_grade', 'quality_grade', *original_defect_heads],
                'raw_dr_probabilities_also_checked': True}
        cp.enrich(rows)
        metrics, vectors = condition_metrics(rows, truth, quality_truth, groups, clean=condition == 'clean')
        report['conditions'][condition] = metrics
        if condition == 'clean':
            clean_vectors = vectors
        else:
            changes = {key: paired_patient_bootstrap(clean_vectors[key], vectors[key], groups)
                       for key in ('accuracy', 'nll', 'entropy', 'poor_quality_probability',
                                   'lac_image_coverage', 'lac_mean_set_size')}
            mask = truth >= 2
            changes['referable_sensitivity'] = paired_patient_bootstrap(
                clean_vectors['referable_prediction'][mask], vectors['referable_prediction'][mask], groups[mask])
            report['paired_changes'][condition] = changes
        cache['conditions'][condition] = {'rgb_sha256': [rgb_sha(image) for image in images], 'rows': rows}
        print(json.dumps({'condition': condition, 'images': len(rows), 'accuracy': metrics['grading']['accuracy'],
                          'poor_probability': metrics['mean_poor_quality_probability']}), flush=True)
    args.output.mkdir(parents=True)
    args.private.mkdir(parents=True)
    cache_path = args.private/'scores.json'
    write_json(cache_path, cache)
    report['private_scores_sha256'] = sha256_file(cache_path)
    report['elapsed_seconds'] = time.perf_counter()-started
    serialized = json.dumps(report, allow_nan=False)
    if any(value in serialized for value in ('subject_id', 'image_path', str(Path.cwd()), 'C:\\Users\\')):
        raise ValueError('Public aggregate contains a private field/path')
    write_json(args.output/'evaluation.json', report)
    make_figures(report, args.output)
    print(json.dumps({'status': 'executed', 'images': len(selected), 'patients': len(np.unique(groups)),
                      'conditions': len(CONDITIONS), 'elapsed_seconds': report['elapsed_seconds'],
                      'clean_equivalence': report['clean_equivalence']}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=Path('data/processed/deepdrid/development_v1.csv'))
    parser.add_argument('--selected', type=Path, default=Path('checkpoints/development_v1/selected'))
    parser.add_argument('--conformal', type=Path, default=Path('checkpoints/spec_v1/conformal_r1.json'))
    parser.add_argument('--protocol', type=Path, default=Path('docs/research/ROBUSTNESS_V1_PROTOCOL.md'))
    parser.add_argument('--output', type=Path, default=Path('reports/robustness_v1'))
    parser.add_argument('--private', type=Path, default=Path('artifacts/robustness_v1'))
    run(parser.parse_args())


if __name__ == '__main__':
    main()
