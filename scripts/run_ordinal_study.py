"""Fixed development-only ordinal randomization/calibration-unit diagnostics."""
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
from torch.utils.data import DataLoader
import torch

from qsart.inference.conformal_sidecar import ConformalSidecar, temperature_probabilities
from qsart.models.checkpoint import load_checkpoint, sha256_file
from qsart.training.data import ManifestDataset, read_manifest, split_fingerprint
from qsart.triage_v2 import Action, pointwise_action
from qsart.uncertainty.conformal import (coverage_metrics, fit_conformal, nonconformity_scores, predict_sets)
from qsart.uncertainty.randomized_ordinal import fit_randomized, predict_randomized, randomized_scores


MANIFEST_SHA = '45e9f2d34ed54a12056b57a808d03cfaf29ccc970d148e9eb107b79cf5ac3e0d'
ARTIFACT_SHA = {
    'dr': 'f530056c27f83dbff8bba4ebb481cf7a9d82289947c4eda655a69ed22685a527',
    'quality': 'a600872ce8e2b256205350a5ef6c99b77d71f679d217c20c26810a3ed76eff43',
    'calibration': '674821e445490c209fab99e56b0e9915ac406ca4ec8694e5efdbed63e456a737',
    'conformal': '15a98f6bfe93f81156c5592701995d08796dd889e3a5e43bcbde7ab5aa61c221',
}
SEEDS = (2026, 2027, 2028, 2029, 2030)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8', newline='\n')


def score_distribution(values):
    values = np.asarray(values, dtype=float)
    return {'n': len(values), 'minimum': float(values.min()), 'median': float(np.median(values)),
            'p90': float(np.quantile(values, .9)), 'maximum': float(values.max()),
            'exact_score_one_count': int((values == 1.).sum())}


def patient_maxima(values, groups):
    return np.array([np.max(values[groups == group]) for group in np.unique(groups)])


def greedy_order(row):
    lower = upper = int(np.argmax(row))
    order = [lower]
    while len(order) < len(row):
        if lower == 0:
            upper += 1; order.append(upper)
        elif upper == len(row)-1:
            lower -= 1; order.append(lower)
        elif row[upper+1] >= row[lower-1]:
            upper += 1; order.append(upper)
        else:
            lower -= 1; order.append(lower)
    return order


def summarize_sets(sets, raw_sets, labels, groups, cached_rows):
    result = coverage_metrics(sets, labels, groups=groups)
    result['full_size_fraction'] = float(np.mean([len(values) == 5 for values in sets]))
    result['raw_empty_count_before_modal_fallback'] = sum(not values for values in raw_sets)
    result['sets_wholly_below_grade2'] = sum(bool(values) and max(values) < 2 for values in sets)
    actions = [pointwise_action(row['grade_probabilities'], 1-row['quality']['poor_probability'],
                    row['uncertainty']['entropy'], conformal_set=values)[0]
               for row, values in zip(cached_rows, sets, strict=True)]
    result['guarded_pointwise_action_counts'] = dict(Counter(action.value for action in actions))
    result['true_referable_auto_count'] = int(sum(action == Action.AUTO_NONREFERABLE and grade >= 2
                                                 for action, grade in zip(actions, labels, strict=True)))
    result['capacity_feasibility_evaluated'] = False
    return result


def make_plots(report, output):
    names = ('lac', 'ordinal_greedy', 'ordinal_greedy_randomized')
    labels = ('LAC', 'Ordinal full mass', 'Ordinal randomized')
    for metric, title, filename, maximum in (
            ('mean_set_size', 'Mean prediction-set size', 'set_sizes.png', 5.),
            ('patient_all_views_coverage', 'Empirical patient all-views coverage', 'patient_coverage.png', 1.)):
        fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
        for i, alpha in enumerate((.1, .05)):
            for j, unit in enumerate(('patient_max', 'image')):
                axis = axes[i, j]
                for x, name in enumerate(names):
                    rows = [r for r in report['comparisons'] if
                            r['method'] == name and r['unit'] == unit and r['alpha'] == alpha]
                    values = np.array([r['metrics'][metric] for r in rows])
                    axis.bar(x, values.mean(), color='#35546c' if x < 2 else '#468e9c', width=.6)
                    axis.scatter(np.repeat(x, len(values)), values, color='#d58b28', s=22, zorder=3)
                    if len(values) > 1:
                        axis.vlines(x, values.min(), values.max(), color='#d58b28', linewidth=2)
                axis.set_xticks(range(3), labels, rotation=12)
                axis.set_ylim(0, maximum)
                axis.set_title(f'alpha {alpha:.2f} | '+('Patient maximum' if unit == 'patient_max'
                                                       else 'Pooled images: diagnostic only'))
                axis.grid(axis='y', alpha=.2)
        fig.suptitle(title+' | 168 selection images / 42 patients\n'
                     'Dots show every randomization seed; not training-seed or cohort intervals')
        fig.savefig(output/filename, dpi=160)
        plt.close(fig)


def run(args):
    started = time.perf_counter()
    root = Path(__file__).resolve().parents[1]
    for path, parent in ((args.private, root/'artifacts'), (args.output, root/'reports')):
        if not path.resolve().is_relative_to(parent.resolve()) or path.resolve() == parent.resolve():
            raise ValueError('Experiment paths must remain inside project-local artifacts/reports subdirectories')
        if path.exists():
            raise ValueError('Refusing an existing experiment output directory')
    if sha256_file(args.manifest) != MANIFEST_SHA:
        raise ValueError('Frozen development manifest differs')
    frame = read_manifest(args.manifest)
    if set(frame.split) != {'train', 'validation', 'calibration'}:
        raise ValueError('Only the declared development manifest is permitted')
    selected = frame.loc[frame.split == 'validation'].reset_index(drop=True)
    calibration = frame.loc[frame.split == 'calibration'].copy()
    if len(selected) != 168 or selected.subject_id.nunique() != 42:
        raise ValueError('Frozen selection count differs')
    patients = sorted(calibration.subject_id.unique(), key=lambda value: (
        hashlib.sha256(f'spec-v1-conformal-20261020:{value}'.encode()).hexdigest(), str(value)))
    temp = calibration[calibration.subject_id.isin(patients[:21])].reset_index(drop=True)
    cp_frame = calibration[calibration.subject_id.isin(patients[21:])].reset_index(drop=True)
    if len(patients) != 42 or len(cp_frame) != 84 or len(temp) != 84:
        raise ValueError('Declared independent calibration subdivision differs')
    paths = {'dr': args.selected/'dr.pt', 'quality': args.selected/'quality.pt',
             'calibration': args.selected/'calibration.json', 'conformal': args.conformal}
    actual_sha = {name: sha256_file(path) for name, path in paths.items()}
    if actual_sha != ARTIFACT_SHA:
        raise ValueError('Frozen artifact identity differs')
    sidecar = ConformalSidecar(paths['conformal'], dr_checkpoint=paths['dr'], quality_checkpoint=paths['quality'],
                              calibration=paths['calibration'])
    bundle = sidecar.bundle
    if (bundle['split_sha256'] != split_fingerprint(cp_frame)
            or bundle['temperature_fit_split_sha256'] != split_fingerprint(temp)
            or bundle['manifest_sha256'] != MANIFEST_SHA):
        raise ValueError('Existing conformal/temperature split binding differs')
    for row in list(cp_frame.itertuples())+list(selected.itertuples()):
        if sha256_file(row.image_path) != row.sha256:
            raise ValueError('Actual development image bytes differ from frozen manifest')
    robust_report = json.loads(args.selection_report.read_text(encoding='utf-8'))
    if (robust_report['private_scores_sha256'] != sha256_file(args.selection_cache)
            or robust_report['artifact_sha256'] != actual_sha
            or robust_report['manifest_sha256'] != MANIFEST_SHA):
        raise ValueError('Selection-cache model/manifest/hash binding differs')
    cache = json.loads(args.selection_cache.read_text(encoding='utf-8'))
    truth = selected.dr_grade.to_numpy(dtype=int)
    groups = selected.subject_id.to_numpy()
    if (cache['original_grades'] != truth.tolist()
            or list(map(str, cache['original_subject_ids'])) != list(map(str, groups))):
        raise ValueError('Selection-cache label/group/order differs')
    rows = cache['conditions']['clean']['rows']
    if len(rows) != len(selected) or any(row['second_pass_activated'] for row in rows):
        raise ValueError('Only the complete clean first-pass selection cache is allowed')
    torch.set_num_threads(2)
    model, checkpoint = load_checkpoint(paths['dr'])
    model_metadata = checkpoint['metadata']
    dataset = ManifestDataset(cp_frame, {'dr_grade': 5}, model_metadata['image_size'],
        mean=model_metadata['normalization_mean'], std=model_metadata['normalization_std'])
    cp_raw = []
    with torch.inference_mode():
        for images, _ in DataLoader(dataset, batch_size=16, num_workers=0):
            cp_raw.extend(model(images)['dr_grade'].softmax(1).tolist())
    temperature = bundle['independent_temperature']['first_pass']
    p_cal = temperature_probabilities(cp_raw, temperature)
    p_eval = temperature_probabilities([row['uncalibrated']['grade_probabilities'] for row in rows], temperature)
    y_cal = cp_frame.dr_grade.to_numpy(dtype=int)
    g_cal = cp_frame.subject_id.to_numpy()
    true_scores = nonconformity_scores(p_cal, method='ordinal_greedy')[np.arange(len(y_cal)), y_cal]
    diagnostics = {'image_true_label_scores': score_distribution(true_scores),
        'patient_max_true_label_scores': score_distribution(patient_maxima(true_scores, g_cal)),
        'true_grade_is_last_in_greedy_order_count': sum(int(label) == greedy_order(p)[-1]
                                                     for p, label in zip(p_cal, y_cal, strict=True)),
        'calibration_images': 84, 'calibration_patients': 21}
    source_names = ('scripts/run_ordinal_study.py', 'qsart/uncertainty/randomized_ordinal.py',
                    'qsart/uncertainty/conformal.py', 'qsart/inference/conformal_sidecar.py',
                    'qsart/training/data.py', 'qsart/preprocessing/fundus.py', 'qsart/triage_v2.py')
    report = {'schema_version': 1, 'experiment_id': 'ordinal_randomization_v1', 'spec_version': '1.0',
        'dataset': 'DeepDRiD', 'partition': 'validation', 'scope': 'development_ordinal_diagnostics',
        'images': 168, 'patients': 42, 'model_version': model_metadata['model_version'],
        'artifact_sha256': actual_sha, 'manifest_sha256': MANIFEST_SHA,
        'protocol_sha256': sha256_file(args.protocol), 'selection_cache_sha256': sha256_file(args.selection_cache),
        'selection_report_sha256': sha256_file(args.selection_report),
        'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'source_sha256': {name: sha256_file(name) for name in source_names},
        'package_versions': {name: metadata.version(name) for name in ('numpy', 'torch', 'torchvision', 'pandas')},
        'temperature_fitted_in_this_study': False, 'independent_temperature': temperature,
        'randomization_seeds': list(SEEDS), 'randomization_generator': 'NumPy default_rng(seed), PCG64',
        'deterministic_score_diagnostics': diagnostics, 'baseline_reproduction': [], 'comparisons': [],
        'randomization_summaries': [], 'limitations': [
            'Method exploration follows inspected development failures; not fresh confirmation.',
            'Selection data informed model choice; no formal or shifted-data coverage guarantee.',
            'Pooled-image diagnostics contain correlated views and change the patient-all-views target.',
            'Five randomization seeds are not five training seeds or independent cohorts.',
            'No seed/method/alpha is selected for deployment; original sidecar remains unchanged.',
            'Pointwise actions do not establish cohort capacities, specialist outcomes or clinical utility.',
            'This score-construction study is not exact APTOS/backbone/base-paper reproduction.']}
    private = {'calibration_raw_probabilities': cp_raw, 'calibration_grades': y_cal.tolist(),
        'calibration_subject_ids': g_cal.tolist(), 'selection_grades': truth.tolist(),
        'selection_subject_ids': groups.tolist(), 'uniforms': {}, 'fits_and_sets': []}
    uniforms = {}
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        uniforms[seed] = (rng.random(84), rng.random(168))
        private['uniforms'][str(seed)] = {'calibration': uniforms[seed][0].tolist(),
                                        'selection': uniforms[seed][1].tolist()}
    for alpha in (.1, .05):
        for unit in ('patient_max', 'image'):
            for method, seed in [('lac', None), ('ordinal_greedy', None)]+[
                    ('ordinal_greedy_randomized', seed) for seed in SEEDS]:
                identifiers = g_cal if unit == 'patient_max' else None
                binding = {'manifest_sha256': MANIFEST_SHA, 'artifact_sha256': actual_sha,
                           'temperature_sha256': bundle['temperature_sha256'],
                           'protocol_sha256': report['protocol_sha256'], 'probability_path': 'first_pass',
                           'randomization_seed': seed}
                if seed is None:
                    artifact = fit_conformal(p_cal, y_cal, alpha=alpha, method=method, groups=identifiers,
                                            metadata=binding)
                    scores = nonconformity_scores(p_cal, method=method)[np.arange(len(y_cal)), y_cal]
                    sets = predict_sets(p_eval, artifact)
                    raw = predict_sets(p_eval, {**artifact, 'empty_policy': 'allow'})
                    if unit == 'patient_max':
                        old = bundle['models']['first_pass'][str(alpha)][method]
                        difference = abs(artifact['qhat']-old['qhat'])
                        matched = sets == predict_sets(p_eval, old)
                        if alpha == .1:
                            field = 'baseline_lac_set' if method == 'lac' else 'prediction_set'
                            matched = matched and sets == [row['conformal'][field] for row in rows]
                        if difference > 1e-6 or not matched:
                            raise ValueError('Existing patient-max baseline could not be reproduced')
                        report['baseline_reproduction'].append({'method': method, 'alpha': alpha,
                            'threshold_absolute_error': difference, 'selection_sets_match': matched,
                            'rank': artifact['rank']})
                else:
                    u_cal, u_eval = uniforms[seed]
                    artifact = fit_randomized(p_cal, y_cal, u_cal, alpha=alpha, groups=identifiers,
                                             metadata=binding)
                    scores = randomized_scores(p_cal, u_cal)[np.arange(len(y_cal)), y_cal]
                    sets = predict_randomized(p_eval, artifact, u_eval)
                    raw = predict_randomized(p_eval, artifact, u_eval, empty_policy='allow')
                result = {'method': method, 'unit': unit, 'alpha': alpha, 'randomization_seed': seed,
                          'threshold': artifact['qhat'], 'rank': artifact['rank'],
                          'calibration_units': artifact['n_calibration_units'],
                          'calibration_score_distribution': score_distribution(
                              patient_maxima(scores, g_cal) if unit == 'patient_max' else scores),
                          'metrics': summarize_sets(sets, raw, truth, groups, rows)}
                report['comparisons'].append(result)
                private['fits_and_sets'].append({'method': method, 'unit': unit, 'alpha': alpha,
                                                'seed': seed, 'artifact': artifact, 'sets': sets, 'raw_sets': raw})
                print(json.dumps({key: result[key] for key in ('method', 'unit', 'alpha', 'randomization_seed')}
                                 | {'mean_size': result['metrics']['mean_set_size']}), flush=True)
    for alpha in (.1, .05):
        for unit in ('patient_max', 'image'):
            matches = [r for r in report['comparisons'] if r['method'] == 'ordinal_greedy_randomized'
                       and r['unit'] == unit and r['alpha'] == alpha]
            summary = {'method': 'ordinal_greedy_randomized', 'alpha': alpha, 'unit': unit, 'seeds': list(SEEDS)}
            for key in ('marginal_coverage', 'patient_all_views_coverage', 'mean_set_size', 'full_size_fraction',
                        'sets_wholly_below_grade2'):
                values = np.array([r['metrics'][key] for r in matches])
                summary[key] = {'mean': float(values.mean()), 'minimum': float(values.min()), 'maximum': float(values.max())}
            report['randomization_summaries'].append(summary)
    args.private.mkdir(parents=True)
    write_json(args.private/'scores_and_sets.json', private)
    report['private_output_sha256'] = sha256_file(args.private/'scores_and_sets.json')
    report['elapsed_seconds'] = time.perf_counter()-started
    public = json.dumps(report, allow_nan=False)
    if any(token in public for token in ('subject_id', 'image_path', str(root), 'C:\\Users\\')):
        raise ValueError('Public report contains a private field/path')
    write_json(args.output/'evaluation.json', report)
    make_plots(report, args.output)
    print(json.dumps({'comparisons': len(report['comparisons']), 'diagnostics': diagnostics,
                      'randomization_summaries': report['randomization_summaries']}, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, default=Path('data/processed/deepdrid/development_v1.csv'))
    p.add_argument('--selected', type=Path, default=Path('checkpoints/development_v1/selected'))
    p.add_argument('--conformal', type=Path, default=Path('checkpoints/spec_v1/conformal_r1.json'))
    p.add_argument('--selection-cache', type=Path, default=Path('artifacts/robustness_v1/scores.json'))
    p.add_argument('--selection-report', type=Path, default=Path('reports/robustness_v1/evaluation.json'))
    p.add_argument('--protocol', type=Path, default=Path('docs/research/ORDINAL_STUDY_V1_PROTOCOL.md'))
    p.add_argument('--private', type=Path, default=Path('artifacts/ordinal_study_v1'))
    p.add_argument('--output', type=Path, default=Path('reports/ordinal_study_v1'))
    run(p.parse_args())


if __name__ == '__main__':
    main()
