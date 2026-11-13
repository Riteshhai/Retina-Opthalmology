"""Derived scientific figures from the completed APTOS schedule extension."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    folder = ROOT/'reports/aptos_optimization_v2'
    source = folder/'evaluation.json'
    original = source.read_bytes()
    report = json.loads(original)
    digest = hashlib.sha256(original).hexdigest()
    paths = [folder/'training_and_grading.png', folder/'conformal_diagnostics.png']
    if any(path.exists() for path in paths):
        raise ValueError('Preserve existing derived figures')
    run = report['runs'][0]
    history = run['history']
    epochs = [row['epoch'] for row in history]
    colors = ['#315f84', '#a76d35', '#358a7b']
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for axis, field, title in zip(axes[0], ['macro_f1', 'accuracy'], ['Selection macro-F1', 'Selection accuracy'], strict=True):
        axis.plot(epochs, [row['validation_'+field] for row in history], 'o-', color=colors[2], label='Extended primary trajectory')
        axis.axhline(report['baseline']['selection_metrics'][field], linestyle='--', color=colors[0], label='Frozen-feature baseline')
        axis.axhline(report['prior_primary']['selection_metrics'][field], linestyle=':', color=colors[1], label='Four-epoch primary')
        axis.axvline(run['selected_epoch'], color='#526070', alpha=.4, label='Selected trained epoch')
        axis.set(title=title, xlabel='Completed epoch', ylim=(0, 1.04), xticks=epochs)
        axis.legend(fontsize=8, loc='lower right')
        axis.yaxis.set_major_formatter(PercentFormatter(1))
    sources = [report['baseline']['selection_metrics'], report['prior_primary']['selection_metrics'], report['grading_before_temperature']]
    labels = ['Frozen baseline', 'Four-epoch primary', 'Extended primary']
    for i, (metrics, label) in enumerate(zip(sources, labels, strict=True)):
        axes[1, 0].bar(np.arange(5)+(i-1)*.24, [row['recall'] for row in metrics['per_class']], width=.23, color=colors[i], label=label)
    axes[1, 0].set(title='Recall against supplied source grades', xticks=range(5), xlabel='DR grade', ylim=(0, 1.04))
    axes[1, 0].yaxis.set_major_formatter(PercentFormatter(1))
    axes[1, 0].legend(fontsize=8, loc='upper right')
    for field, color, label in [('backbone_learning_rate', colors[0], 'Backbone LR'), ('head_learning_rate', colors[1], 'Head LR')]:
        axes[1, 1].plot(epochs, [row[field] for row in history], 'o-', color=color, label=label)
    axes[1, 1].set(title='Predeclared 15-epoch cosine schedule', xlabel='Completed epoch', xticks=epochs, ylabel='Learning rate', yscale='log')
    axes[1, 1].legend(fontsize=8)
    for axis in axes.flat:
        axis.grid(axis='y', alpha=.18)
        axis.set_axisbelow(True)
    fig.suptitle('APTOS source-development extension · fixed primary training seed 2026\n'
        f"Completed {report['summary']['epochs_executed']}/15 maximum epochs; selected epoch {run['selected_epoch']}; patience four", fontsize=14)
    fig.text(.5, .066, 'Same selected warm start and reused selection cohort. One extended seed; no new seed-variance or fresh-test estimate.', ha='center', fontsize=9)
    fig.text(.5, .041, 'All severe-grade changes retained. No IDRiD or reserved-test inference; production models remain unchanged.', ha='center', fontsize=9)
    fig.text(.5, .016, f'Aggregate SHA256: {digest[:24]}…', ha='center', fontsize=9, color='#526070')
    fig.subplots_adjust(top=.9, bottom=.15, hspace=.34, wspace=.23)
    fig.savefig(paths[0], dpi=160, facecolor='white')
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for column, alpha in enumerate([.1, .05]):
        rows = [row for row in report['comparisons'] if row['alpha'] == alpha]
        names = [('LAC' if row['method'] == 'lac' else 'Ordinal\nfixed' if row['randomization_seed'] is None
                  else 'Random\n'+str(row['randomization_seed'])) for row in rows]
        for row_index, metric, title in [(0, 'marginal_coverage', 'Empirical inclusion'), (1, 'mean_set_size', 'Mean set size')]:
            axis = axes[row_index, column]
            raw = [row['raw']['all_selection_images'][metric] for row in rows]
            fallback = [row['argmax_fallback']['all_selection_images'][metric] for row in rows]
            axis.bar(range(7), fallback, color=[colors[0], colors[1]]+[colors[2]]*5, alpha=.8)
            axis.plot(range(7), raw, 'o', color='#152e43', label='Raw inversion')
            axis.set(title=f'Alpha {alpha:.2f} · {title}', xticks=range(7), xticklabels=names)
            axis.tick_params(axis='x', labelsize=8)
            if row_index == 0:
                axis.set_ylim(0, 1.08)
                axis.yaxis.set_major_formatter(PercentFormatter(1))
                axis.axhline(1-alpha, linestyle='--', color='#b1484c')
            else:
                axis.set_ylim(0, 5.3)
            axis.grid(axis='y', alpha=.18)
            axis.set_axisbelow(True)
    fig.suptitle(f"APTOS extension · classifier training seed 2026, selected epoch {run['selected_epoch']}\n"
        'All five conformal randomization seeds and both alpha levels', fontsize=14)
    fig.text(.5, .08, 'Bars: common argmax fallback. Dots: raw inversion. All 521 selection images; representative diagnostics remain in the report.', ha='center', fontsize=9)
    fig.text(.5, .052, 'Dashed target guides are not coverage guarantees. Reused selection/calibration groups; fingerprint units are not patients.', ha='center', fontsize=9)
    fig.text(.5, .024, f'No clinical, external or triage-utility claim. Aggregate SHA256: {digest[:24]}…', ha='center', fontsize=9, color='#526070')
    fig.subplots_adjust(top=.9, bottom=.18, hspace=.38, wspace=.23)
    fig.savefig(paths[1], dpi=160, facecolor='white')
    plt.close(fig)
    assert source.read_bytes() == original
    print(json.dumps({'source_sha256': digest, 'figures': [path.name for path in paths]}, indent=2))


if __name__ == '__main__':
    main()
