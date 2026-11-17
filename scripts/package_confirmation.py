"""Package frozen aggregate evidence; never fit or select models from confirmation."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from qsart.models.checkpoint import sha256_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selected', type=Path, default=Path('checkpoints/development_v1/selected'))
    parser.add_argument('--report', type=Path, default=Path('reports/confirmation_v1/evaluation.json'))
    args = parser.parse_args()
    lock_path = args.selected/'selection_lock.json'
    lock = json.loads(lock_path.read_text(encoding='utf-8'))
    report = json.loads(args.report.read_text(encoding='utf-8'))
    if (lock.get('experiment') != 'development_v1' or report.get('experiment') != 'confirmation_v1'
            or report['images'] != 340 or report['patients'] != 85):
        parser.error('This packaging command describes the declared development-v1 cohort only')
    if report['selection_lock_sha256'] != sha256_file(lock_path):
        parser.error('Confirmation report does not match the selection lock')
    for name, filename in [('dr_checkpoint_sha256', 'dr.pt'),
                           ('quality_checkpoint_sha256', 'quality.pt'),
                           ('calibration_sha256', 'calibration.json')]:
        if sha256_file(args.selected/filename) != lock[name]:
            parser.error(f'Locked artifact changed: {filename}')
    card = {
        'date': report['date'], 'research_only': True,
        'model_version': report['model_info']['model_version'],
        'quality_model_version': report['model_info']['quality_model_version'],
        **{key: lock[key] for key in ('dr_checkpoint_sha256', 'quality_checkpoint_sha256',
                                      'calibration_sha256', 'protocol_sha256')},
        'selection_lock_sha256': sha256_file(lock_path),
        'confirmation_report_sha256': sha256_file(args.report),
        'calibration_content_sha256': hashlib.sha256(json.dumps(
            json.loads((args.selected/'calibration.json').read_text(encoding='utf-8')),
            sort_keys=True, separators=(',', ':')).encode()).hexdigest(),
        'development': {'fitting_images': 848, 'fitting_patients': 212,
                        'selection_images': 168, 'selection_patients': 42,
                        'calibration_images': 168, 'calibration_patients': 42,
                        'initialization': 'Official torchvision ImageNet; frozen backbone',
                        'selection_rule': 'Selection macro-F1, then lowest unweighted NLL'},
        'confirmation': {
            'scope': report['scope'], 'images': report['images'], 'patients': report['patients'],
            'dr_accuracy': report['grading']['accuracy'],
            'dr_accuracy_patient_bootstrap_ci95': report['accuracy_ci95'],
            'dr_macro_f1': report['grading']['macro_f1'],
            'dr_quadratic_weighted_kappa': report['grading']['quadratic_weighted_kappa'],
            'referable_auroc': report['grading']['referable_auroc'],
            'dr_ece': report['grading']['ece'],
            'overall_quality': report['quality'],
            'classical_quality_comparator': report['classical_quality'],
            'defect_heads': {name: {key: row[key] for key in ('accuracy', 'macro_f1', 'n')}
                             for name, row in report['defect_heads'].items()}},
        'limitations': report['limitations'] + [
            'Clarity, field-definition and artifact predictions have limited accuracy.',
            'The artifact head is worse than a constant majority-class accuracy comparator.',
            'The old pilot used a different cohort/protocol; no paired improvement estimate is made.',
            'Calibration-fit diagnostics are not independent confirmation performance.',
            'Policy confidence intervals are exploratory and conditional on fixed cohort assignments.',
            'Not for patient care; prospective utility and clinical authorization are unestablished.']}
    payload = json.dumps(card, indent=2, allow_nan=False)
    (args.selected/'model_card.json').write_text(payload, encoding='utf-8')
    public_card = Path('reports/development_v1/model_card.json')
    public_card.parent.mkdir(parents=True, exist_ok=True)
    public_card.write_text(payload, encoding='utf-8')

    # Show every feasible operating point; do not select a favorable budget.
    methods = list(dict.fromkeys(row['method'] for row in report['budgets']))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for method in methods:
        rows = [row for row in report['budgets']
                if row['method'] == method and row['status'] == 'evaluated']
        for ax, key in zip(axes, ['missed_referable_auto', 'severe_deferred_to_recapture']):
            ax.scatter([1-row['coverage'] for row in rows],
                       [1000*row[key]/row['n'] for row in rows], label=method, alpha=.75)
    for ax, title in zip(axes, ['Referable disease in automated negatives', 'Severe disease sent to recapture']):
        ax.set(xlabel='Total realized deferral fraction', ylabel='Cases per 1,000 all inputs',
               title=title, xlim=(-.02, 1.02))
        ax.grid(alpha=.2)
    axes[1].legend(fontsize=8)
    fig.suptitle('Fixed nine-budget confirmation grid: ceilings and realized action use differ')
    fig.tight_layout()
    fig.savefig(args.report.parent/'workload_outcomes.png', dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 5))
    for key, label in [('grading_uncalibrated', 'Raw'), ('grading', 'Temperature scaled')]:
        bins = [row for row in report[key]['reliability_bins'] if row['count']]
        ax.plot([row['confidence'] for row in bins], [row['accuracy'] for row in bins],
                marker='o', label=label)
    ax.plot([0, 1], [0, 1], '--', color='gray')
    ax.set(xlabel='Mean top-class confidence', ylabel='Observed accuracy',
           title='DR confirmation reliability: fixed 10 confidence bins',
           xlim=(0, 1), ylim=(0, 1))
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.report.parent/'dr_reliability.png', dpi=180)
    plt.close(fig)
    print(json.dumps({'model_card': str(public_card), 'model_version': card['model_version'],
                      'policy_scenarios': len(report['budgets']),
                      'infeasible': sum(row['status'] == 'infeasible' for row in report['budgets'])}))


if __name__ == '__main__':
    main()
