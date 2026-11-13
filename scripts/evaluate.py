"""Locked pilot evaluation. Output aggregates; per-image scores remain local/ignored."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pandas as pd

from qsart.evaluation import ablations, evaluate_budgets, grade_metrics, grouped_accuracy_ci
from qsart.inference.engine import InferenceEngine
from qsart.models.checkpoint import sha256_file
from qsart.training.data import read_manifest
from qsart.triage import BudgetInfeasible, PolicyConfig, allocate, screening_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--split', choices=['validation', 'holdout'], default='holdout')
    parser.add_argument('--dr-checkpoint', default='checkpoints/dr/best.pt')
    parser.add_argument('--quality-checkpoint', default='checkpoints/quality/best.pt')
    parser.add_argument('--calibration', default='checkpoints/calibration.json')
    parser.add_argument('--output', type=Path, default=Path('reports/pilot'))
    parser.add_argument('--seed', type=int, default=2026)
    args = parser.parse_args()
    engine = InferenceEngine.load(args.dr_checkpoint, args.quality_checkpoint, args.calibration)
    frame = read_manifest(args.manifest)
    frame = frame[frame.split == args.split].copy()
    if frame.empty:
        parser.error('Requested evaluation partition is empty')
    # Checkpoints are already frozen. This command never updates model/policy parameters.
    images = []
    for path in frame.image_path:
        with Image.open(path) as im:
            images.append(im.convert('RGB').copy())
    rows = engine.predict_images(images, second_pass_budget=0.)
    calibration = deepcopy(engine.calibration)
    for task in ('dr', 'quality'):
        for head in engine.calibration[task]['heads'].values():
            head['temperature'] = 1.
            head['flip_tta_temperature'] = 1.
    uncalibrated = engine.predict_images(images, second_pass_budget=0.)
    engine.calibration = calibration
    p = np.array([r['grade_probabilities'] for r in rows])
    q = np.array([1-r['quality']['poor_probability'] for r in rows])
    u = np.array([r['uncertainty']['entropy'] for r in rows])
    grades = frame.dr_grade.to_numpy(dtype=int)
    budgets = evaluate_budgets(grades, p, q, u)
    ablation_rows = ablations(grades, p, q, u)
    raw_p = np.array([r['grade_probabilities'] for r in uncalibrated])
    raw_q = np.array([1-r['quality']['poor_probability'] for r in uncalibrated])
    raw_u = np.array([r['uncertainty']['entropy'] for r in uncalibrated])
    try:
        actions = allocate(raw_p, raw_q, raw_u, PolicyConfig(.4, .2))
        ablation_rows.append({'ablation': 'without_calibration', 'status': 'evaluated',
                             **screening_metrics(grades, raw_p, actions, PolicyConfig(.4, .2))})
    except BudgetInfeasible:
        ablation_rows.append({'ablation': 'without_calibration', 'status': 'infeasible'})
    quality_labels = frame.quality_grade.to_numpy(dtype=int)
    quality_predictions = np.array([r['quality']['predicted_class'] for r in rows])
    counts, errors = np.zeros((5, 2), dtype=int), np.zeros((5, 2), dtype=int)
    for grade, quality, predicted in zip(grades, quality_labels, p.argmax(1)):
        counts[grade, quality] += 1
        errors[grade, quality] += grade != predicted
    try:
        revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    # Source digest identifies the exact uncommitted implementation used by this run.
    source_digest = hashlib.sha256()
    for source in sorted(Path('qsart').rglob('*.py')):
        source_digest.update(source.as_posix().encode()+source.read_bytes())
    report = {
        'experiment': 'EXP01_EXP04_EXP05_EXP08_pilot', 'data_purpose': 'real_image_pilot',
        'limitation': 'Tiny pilot trained from random initialization; not clinical validation or a publication result.',
        'dataset': 'DeepDRiD regular fundus', 'split': args.split,
        'manifest_sha256': sha256_file(args.manifest), 'checkpoint_sha256': sha256_file(args.dr_checkpoint),
        'quality_checkpoint_sha256': sha256_file(args.quality_checkpoint),
        'calibration_sha256': sha256_file(args.calibration), 'git_revision': revision,
        'source_sha256': source_digest.hexdigest(), 'seed': args.seed,
        'model_info': engine.info(), 'grading': grade_metrics(grades, p),
        'grading_uncalibrated': grade_metrics(grades, raw_p),
        'accuracy_ci95': grouped_accuracy_ci(grades, p, frame.subject_id.to_numpy(), args.seed),
        'quality': {'n': len(frame), 'accuracy': float((quality_predictions == quality_labels).mean()),
                    'confusion_matrix': [[int(((quality_labels == a) & (quality_predictions == b)).sum())
                                          for b in range(2)] for a in range(2)],
                    'natural_quality_codes': {'0': 'not good enough', '1': 'good enough'}},
        'severity_quality': {'sample_counts': counts.tolist(), 'error_counts': errors.tolist()},
        'budgets': budgets, 'ablations': ablation_rows,
        'interpretation': ('No policy thresholds optimized on this held-out population. Cohort ranking uses scores '
                           'and requested capacities only. Actual workloads can differ; compare matched operating '
                           'points before any superiority claim. Deferral does not imply correct care.')}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'evaluation.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    pd.DataFrame(budgets).to_csv(args.output/'budget_comparisons.csv', index=False)
    pd.DataFrame(report['ablations']).to_csv(args.output/'ablations.csv', index=False)
    private = Path('artifacts')/args.output.name
    private.mkdir(parents=True, exist_ok=True)
    (private/'predictions.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(report['grading']['confusion_matrix'], cmap='Blues')
    ax.set(xlabel='Predicted grade', ylabel='Reference grade', title='Real-image pilot confusion matrix')
    for i in range(5):
        for j in range(5):
            ax.text(j, i, report['grading']['confusion_matrix'][i][j], ha='center', va='center')
    fig.tight_layout(); fig.savefig(args.output/'confusion_matrix.png', dpi=180); plt.close(fig)
    rates = np.divide(errors, counts, out=np.full((5, 2), np.nan), where=counts > 0)
    fig, ax = plt.subplots(figsize=(6, 5))
    plot = ax.imshow(rates, cmap='Reds', vmin=0, vmax=1)
    ax.set(xticks=[0, 1], xticklabels=['Not good enough', 'Good enough'],
           ylabel='Reference DR grade', title='Pilot natural quality × severity error rate')
    for i in range(5):
        for j in range(2):
            ax.text(j, i, f'n={counts[i,j]}', ha='center', va='center')
    fig.colorbar(plot, ax=ax); fig.tight_layout()
    fig.savefig(args.output/'severity_quality.png', dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 5))
    b = report['grading']['reliability_bins']
    populated = [x for x in b if x['count']]
    ax.plot([0, 1], [0, 1], '--', color='gray')
    ax.plot([x['confidence'] for x in populated], [x['accuracy'] for x in populated], 'o-')
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel='Confidence', ylabel='Accuracy', title='Held-out pilot reliability')
    fig.tight_layout(); fig.savefig(args.output/'reliability.png', dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 5))
    for method in sorted({r['method'] for r in budgets}):
        points = [r for r in budgets if r['method'] == method and r['status'] == 'evaluated']
        ax.scatter([r['coverage'] for r in points], [r['weighted_auto_error_per_image'] for r in points],
                   label=method, alpha=.7)
    ax.set(xlabel='Automated coverage', ylabel='Weighted automated grade error / all images',
           title='Pilot budget operating points (exploratory costs)')
    ax.legend(fontsize=7); fig.tight_layout()
    fig.savefig(args.output/'risk_coverage.png', dpi=180); plt.close(fig)
    print(json.dumps({'n': len(frame), 'grading': report['grading'], 'output': str(args.output)}, indent=2))


if __name__ == '__main__':
    main()
