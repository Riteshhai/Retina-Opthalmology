"""Evaluate frozen artifacts once on the separately reserved internal confirmation cohort."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from qsart.evaluation import evaluate_budgets, grade_metrics
from qsart.inference.engine import InferenceEngine
from qsart.models.checkpoint import sha256_file
from qsart.quality_baselines import binary_quality_metrics, feature_matrix, predict_statistical_quality
from qsart.research_statistics import OUTCOME_NAMES, cluster_mean_intervals, policy_outcomes
from qsart.training.data import read_manifest
from qsart.triage import METHODS, BudgetInfeasible, PolicyConfig, allocate, screening_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--development', type=Path, default=Path('data/processed/deepdrid/development_v1.csv'))
    parser.add_argument('--confirmation', type=Path, default=Path('data/processed/deepdrid/confirmation_v1.csv'))
    parser.add_argument('--selected', type=Path, default=Path('checkpoints/development_v1/selected'))
    parser.add_argument('--output', type=Path, default=Path('reports/confirmation_v1'))
    parser.add_argument('--batch-size', type=int, default=8)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error('Positive inference batch size required')
    if (args.output/'evaluation.json').exists():
        parser.error('Confirmation report already exists; use frozen artifacts for further analysis')
    lock = json.loads((args.selected/'selection_lock.json').read_text(encoding='utf-8'))
    checks = {'development_manifest_sha256': args.development,
              'confirmation_manifest_sha256': args.confirmation,
              'dr_checkpoint_sha256': args.selected/'dr.pt',
              'quality_checkpoint_sha256': args.selected/'quality.pt',
              'calibration_sha256': args.selected/'calibration.json',
              'triage_source_sha256': Path('qsart/triage.py'),
              'evaluation_source_sha256': Path(__file__),
              'protocol_sha256': Path('docs/research/DEVELOPMENT_V1_PROTOCOL.md'),
              'classical_quality_sha256': args.selected.parent/'classical_quality.json'}
    for name, path in checks.items():
        if sha256_file(path) != lock[name]:
            parser.error(f'Locked artifact changed before confirmation: {name}')
    for path, expected in lock['evaluation_dependency_sha256'].items():
        if sha256_file(path) != expected:
            parser.error(f'Locked evaluation dependency changed: {path}')
    frame, development = read_manifest(args.confirmation), read_manifest(args.development)
    if not frame.split.eq('holdout').all():
        parser.error('Confirmation must be an entirely reserved holdout cohort')
    engine = InferenceEngine(args.selected/'dr.pt', args.selected/'quality.pt', args.selected/'calibration.json')
    classical = json.loads((args.selected.parent/'classical_quality.json').read_text(encoding='utf-8'))
    private = Path('artifacts/confirmation_v1'); private.mkdir(parents=True, exist_ok=True)
    predictions, classical_probabilities = [], []
    for start in range(0, len(frame), args.batch_size):
        part = frame.iloc[start:start+args.batch_size]
        images = []
        for path in part.image_path:
            with Image.open(path) as image:
                images.append(image.convert('RGB').copy())
        predictions.extend(engine.predict_images(images, include_uncalibrated=True))
        classical_probabilities.extend(predict_statistical_quality(feature_matrix(part), classical).tolist())
        (private/'predictions.partial.json').write_text(json.dumps({
            'selection_lock_sha256': sha256_file(args.selected/'selection_lock.json'),
            'completed_images': len(predictions), 'results': predictions,
            'classical_probabilities': classical_probabilities}), encoding='utf-8')
        print(f'Frozen confirmation inference: {len(predictions)}/{len(frame)} images', flush=True)
    p = np.array([r['grade_probabilities'] for r in predictions])
    raw = np.array([r['uncalibrated']['grade_probabilities'] for r in predictions])
    quality_p = np.array([r['quality']['probabilities'] for r in predictions])
    raw_q = np.array([r['uncalibrated']['quality_probabilities'] for r in predictions])
    q = quality_p[:, 1]
    u = np.array([r['uncertainty']['entropy'] for r in predictions])
    grades, quality = frame.dr_grade.to_numpy(dtype=int), frame.quality_grade.to_numpy(dtype=int)
    ids = frame.subject_id.to_numpy()
    prior = np.bincount(development.loc[development.split.eq('train'), 'dr_grade'].astype(int), minlength=5)
    prior = np.tile(prior/prior.sum(), (len(frame), 1))
    report = {'experiment': 'confirmation_v1', 'date': '2026-10-20',
              'scope': 'Fresh internal DeepDRiD same-domain confirmation; not external or clinical validation',
              'images': len(frame), 'patients': frame.subject_id.nunique(),
              'selection_lock_sha256': sha256_file(args.selected/'selection_lock.json'),
              'confirmation_manifest_sha256': sha256_file(args.confirmation),
              'model_info': engine.info(), 'grading': grade_metrics(grades, p),
              'grading_uncalibrated': grade_metrics(grades, raw),
              'training_prior_grading': grade_metrics(grades, prior),
              'accuracy_ci95': cluster_mean_intervals((p.argmax(1) == grades).astype(float), ids)[0],
              'quality': binary_quality_metrics(quality, quality_p),
              'quality_uncalibrated': binary_quality_metrics(quality, raw_q),
              'classical_quality': binary_quality_metrics(quality, classical_probabilities),
              'budgets': evaluate_budgets(grades, p, q, u),
              'defect_heads': {}, 'paired_policy_comparisons': [],
              'limitations': ['Single-site/domain internal evaluation; external transport is unmeasured.',
                              'Recapture and specialist review remain unresolved recommendations.',
                              'No confirmation-based model or policy tuning was performed.',
                              'Bootstrap intervals are exploratory and not corrected for multiple comparisons.']}
    for head, mapping in engine.quality_metadata.get('label_mappings', {}).items():
        observed = frame[head].to_numpy(dtype=int)
        probabilities = np.array([r['quality']['defects'][head] for r in predictions])
        report['defect_heads'][head] = {
            'label_mapping': mapping, 'n': len(observed),
            'accuracy': float(accuracy_score(observed, probabilities.argmax(1))),
            'macro_f1': float(f1_score(observed, probabilities.argmax(1), labels=np.arange(mapping['classes']),
                                     average='macro', zero_division=0)),
            'confusion_matrix': confusion_matrix(observed, probabilities.argmax(1),
                                                  labels=np.arange(mapping['classes'])).tolist()}
    for review in (.2, .4, .6):
        for recapture in (.1, .2, .4):
            config = PolicyConfig(review, recapture)
            routes = {}
            for method in METHODS:
                try:
                    routes[method] = allocate(p, q, u, config, method)
                except BudgetInfeasible:
                    continue
            if 'dual_budget' not in routes:
                continue
            reference = screening_metrics(grades, p, routes['dual_budget'], config)
            a = policy_outcomes(grades, p, routes['dual_budget'])
            for method, actions in routes.items():
                if method == 'dual_budget':
                    continue
                baseline = screening_metrics(grades, p, actions, config)
                matched = all(abs(reference[k]-baseline[k]) < 1e-9 for k in ('review_rate', 'recapture_rate'))
                comparison = {'review_budget': review, 'recapture_budget': recapture,
                              'baseline': method, 'actual_workloads_matched': matched}
                if matched:
                    delta = a-policy_outcomes(grades, p, actions)
                    intervals = cluster_mean_intervals(delta, ids)
                    comparison['proposed_minus_baseline_ci95'] = dict(zip(OUTCOME_NAMES, intervals))
                    comparison['interpretation'] = 'Conditional on original cohort assignments; lower errors/delays favored.'
                else:
                    comparison['interpretation'] = 'No matched-workload superiority comparison is made.'
                report['paired_policy_comparisons'].append(comparison)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'evaluation.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    pd.DataFrame(report['budgets']).to_csv(args.output/'budget_comparisons.csv', index=False)
    counts = np.zeros((5, 2), dtype=int); errors = counts.copy()
    for grade, label, predicted in zip(grades, quality, p.argmax(1)):
        counts[grade, label] += 1; errors[grade, label] += grade != predicted
    (args.output/'severity_quality.json').write_text(json.dumps({
        'counts': counts.tolist(), 'errors': errors.tolist(), 'quality_is_natural_label': True}, indent=2))
    for name, matrix in [('grading_confusion', report['grading']['confusion_matrix']),
                          ('quality_confusion', report['quality']['confusion_matrix'])]:
        matrix = np.asarray(matrix)
        fig, ax = plt.subplots(figsize=(6, 5)); ax.imshow(matrix, cmap='Blues')
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(j, i, matrix[i, j], ha='center', va='center')
        ax.set(xlabel='Predicted class', ylabel='Reference class', title=name.replace('_', ' ').title())
        fig.tight_layout(); fig.savefig(args.output/f'{name}.png', dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 5))
    rates = np.divide(errors, counts, out=np.full((5, 2), np.nan), where=counts > 0)
    plot = ax.imshow(rates, vmin=0, vmax=1, cmap='Reds')
    ax.set(xticks=[0, 1], xticklabels=['Poor', 'Good'], ylabel='Reference DR grade',
           title='Natural quality × severity: grading error rate')
    for i in range(5):
        for j in range(2):
            ax.text(j, i, f'n={counts[i,j]}', ha='center', va='center')
    fig.colorbar(plot, ax=ax); fig.tight_layout(); fig.savefig(args.output/'severity_quality.png', dpi=180)
    plt.close(fig)
    print(json.dumps({key: report[key] for key in ('images', 'patients', 'grading', 'quality', 'classical_quality')}, indent=2))


if __name__ == '__main__':
    main()
