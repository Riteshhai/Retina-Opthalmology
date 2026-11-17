"""Plot only saved, genuine APTOS aggregates; never infer or choose a new model."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np


NAMES = {'efficientnet_b0': 'EfficientNet-B0', 'mobilenet_v3_small': 'MobileNetV3-small'}
METHODS = [('lac', None), ('ordinal_greedy', None)] + [
    ('ordinal_greedy_randomized', seed) for seed in range(2026, 2031)]
COLORS = ['#385f88', '#a56f3b'] + ['#40988e'] * 5


def _validate(report):
    if (report.get('dataset') != 'APTOS_2019' or report.get('experiment_id') != 'aptos_development_v1' or
            report.get('scope') != 'development_selection_and_independent_calibration'):
        raise ValueError('Expected completed APTOS development v1 aggregates')
    if {row['backbone'] for row in report['candidates']} != set(NAMES) or len(report['candidates']) != 2:
        raise ValueError('Both genuine candidates must remain visible')
    if len(report['comparisons']) != 14:
        raise ValueError('All fourteen fixed conformal comparisons must remain visible')
    expected = {(method, seed, alpha) for method, seed in METHODS for alpha in (.1, .05)}
    actual = {(row['method'], row['randomization_seed'], row['alpha']) for row in report['comparisons']}
    if actual != expected:
        raise ValueError('Conformal methods, seeds or alpha values differ from the declaration')
    if report['population'].get('reserved_test_inference_performed') is not False:
        raise ValueError('These figures describe development-only data')


def candidate_figure(report, output, digest):
    candidates = report['candidates']
    selected = report['summary']['selected_backbone']
    labels = [NAMES[row['backbone']] for row in candidates]
    colors = ['#31756c' if row['backbone'] == selected else '#a5b6c8' for row in candidates]
    fig = plt.figure(figsize=(14, 8.9), facecolor='white')
    grid = fig.add_gridspec(2, 3, height_ratios=(1, 1.65), hspace=.5, wspace=.44)
    for column, (metric, title) in enumerate((('accuracy', 'Selection accuracy'),
                                             ('macro_f1', 'Selection macro-F1'),
                                             ('quadratic_weighted_kappa', 'Selection quadratic kappa'))):
        axis = fig.add_subplot(grid[0, column])
        values = [row['selection_metrics'][metric] for row in candidates]
        axis.bar(range(len(candidates)), values, width=.56, color=colors)
        for index, value in enumerate(values):
            formatted = f'{value:.2%}' if metric == 'accuracy' else f'{value:.4f}'
            axis.text(index, value+.018, formatted, ha='center', va='bottom', fontsize=11)
        axis.set_xticks(range(len(candidates)), labels, fontsize=9)
        axis.set_ylim(min(0, min(values)*1.1), 1.03)
        axis.set_title(title, fontsize=12, pad=14)
        if metric == 'accuracy':
            axis.yaxis.set_major_formatter(PercentFormatter(1))
        axis.spines[['top', 'right']].set_visible(False)
        axis.grid(axis='y', alpha=.18)
        axis.set_axisbelow(True)
    matrix_axis = fig.add_subplot(grid[1, :2])
    matrix = np.asarray(report['grading']['confusion_matrix'], dtype=int)
    if matrix.shape != (5, 5) or matrix.sum() != report['images']:
        raise ValueError('Selected confusion matrix does not match the actual population')
    matrix_axis.imshow(matrix, cmap='Blues', interpolation='nearest')
    for row, column in np.ndindex(matrix.shape):
        matrix_axis.text(column, row, str(matrix[row, column]), ha='center', va='center', fontsize=11,
                         color='white' if matrix[row, column] > matrix.max()/2 else '#15304a')
    matrix_axis.set(xticks=range(5), yticks=range(5), xlabel='Predicted DR grade', ylabel='Source DR grade',
                    title=f"Selected {NAMES[selected]} · all {report['images']} selection images")
    matrix_axis.tick_params(length=0)
    matrix_axis.spines[:].set_visible(False)
    note = fig.add_subplot(grid[1, 2])
    note.axis('off')
    before, after = report['grading_before_temperature'], report['grading']
    lines = ['Selected by macro-F1, then NLL:', NAMES[selected], '',
        f"Training seed: {report['summary']['training_seed']}",
        f"Selected epoch: {next(row['selected_epoch'] for row in candidates if row['backbone'] == selected)}", '',
        'Independent temperature fit', f"T = {report['calibration']['temperature']:.4f}",
        'Selection diagnostics before → after:', f"NLL: {before['nll']:.4f} → {after['nll']:.4f}",
        f"ECE (10 bins): {before['ece']:.4f} → {after['ece']:.4f}",
        f"Brier: {before['brier']:.4f} → {after['brier']:.4f}", '',
        'Green marks the declared study selection.',
        'No checkpoint was promoted to the app.']
    note.text(0, 1, '\n'.join(lines), va='top', fontsize=10, linespacing=1.5)
    fig.suptitle('APTOS: two genuinely trained frozen-feature candidates', fontsize=17, y=.985)
    fig.text(.5, .041, 'Same development selection population; epochs and architecture were chosen here. '
             'One training seed; no fresh-test or clinical claim.', ha='center', fontsize=10)
    fig.text(.5, .018, f'Metric bars precede temperature fitting. Source aggregate SHA256: {digest[:16]}…',
             ha='center', fontsize=9, color='#4c5966')
    fig.subplots_adjust(top=.90, bottom=.12, left=.08, right=.97)
    fig.savefig(output, dpi=160, facecolor='white')
    plt.close(fig)


def conformal_figure(report, output, digest):
    lookup = {(row['method'], row['randomization_seed'], row['alpha']): row for row in report['comparisons']}
    labels = ['LAC', 'Ordinal\nfixed'] + [f'Ordinal\nseed {seed}' for seed in range(2026, 2031)]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(2, 2, figsize=(15, 9.4), facecolor='white')
    for column, alpha in enumerate((.1, .05)):
        records = [lookup[(method, seed, alpha)] for method, seed in METHODS]
        outcomes = [row['argmax_fallback']['all_selection_images'] for row in records]
        representatives = [row['argmax_fallback']['selection_group_representatives'] for row in records]
        coverage = [row['marginal_coverage'] for row in outcomes]
        rep_coverage = [row['marginal_coverage'] for row in representatives]
        sizes = [row['mean_set_size'] for row in outcomes]
        upper, lower = axes[0, column], axes[1, column]
        upper.bar(x, coverage, color=COLORS, width=.62, zorder=2)
        upper.scatter(x, rep_coverage, color='#162e44', marker='D', s=25, zorder=3,
                      label='One representative per selection fingerprint component')
        upper.axhline(1-alpha, color='#9b3437', linestyle='--', linewidth=1.2,
                      label=f'{1-alpha:.0%} nominal guide; not a coverage guarantee')
        for index, value in enumerate(coverage):
            upper.text(index, value+.014, f'{value:.1%}', ha='center', fontsize=8)
        upper.set(ylim=(0, 1.105), ylabel='Empirical inclusion', title=f'Alpha {alpha:.2f} · selection coverage')
        upper.yaxis.set_major_formatter(PercentFormatter(1))
        upper.legend(loc='lower left', fontsize=8, frameon=True)
        lower.bar(x, sizes, color=COLORS, width=.62)
        for index, value in enumerate(sizes):
            lower.text(index, value+.07, f'{value:.3f}', ha='center', fontsize=9)
        lower.set(ylim=(0, 5.55), ylabel='Mean grades in prediction set', title=f'Alpha {alpha:.2f} · selection set size')
        for axis in (upper, lower):
            axis.set_xticks(x, labels, fontsize=8)
            axis.spines[['top', 'right']].set_visible(False)
            axis.grid(axis='y', alpha=.17)
            axis.set_axisbelow(True)
        raw_empty = [row['raw']['all_selection_images']['empty_set_count'] for row in records]
        lower.text(.5, -.28, 'Raw empty counts, in method order: '+', '.join(map(str, raw_empty)),
                   transform=lower.transAxes, ha='center', fontsize=9)
    fig.suptitle('APTOS conformal diagnostics · common argmax fallback', fontsize=17, y=.985)
    fig.text(.5, .070, f"{report['images']} previously used selection images; selected classifier "
             f"{NAMES[report['summary']['selected_backbone']]}. All bars use the common argmax fallback.",
             ha='center', fontsize=9)
    fig.text(.5, .050, 'Ordinal fixed = deterministic greedy ordinal; seed-labelled columns retain all five '
             'randomized ordinal replicates, not model-training replicates.', ha='center', fontsize=9)
    fig.text(.5, .030, 'Conformal calibration uses one image per fingerprint component, not known patients. '
             'Unknown patient overlap and grade-stratified partitions limit coverage claims.', ha='center', fontsize=9)
    fig.text(.5, .010, f'Raw empty sets and grade-specific/full-set outcomes remain in the complete report. '
             f'Source SHA256: {digest[:16]}…', ha='center', fontsize=9, color='#4c5966')
    fig.subplots_adjust(left=.075, right=.985, top=.91, bottom=.185, hspace=.47, wspace=.18)
    fig.savefig(output, dpi=160, facecolor='white')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('reports/aptos_v1/evaluation.json'))
    parser.add_argument('--output', type=Path, default=Path('reports/aptos_v1'))
    parser.add_argument('--overwrite-derived', action='store_true', help='Replace derived figures during visual QA only')
    args = parser.parse_args()
    raw = args.source.read_bytes()
    report = json.loads(raw)
    _validate(report)
    outputs = [args.output/'candidate_comparison.png', args.output/'conformal_diagnostics.png']
    if not args.overwrite_derived and any(path.exists() for path in outputs):
        raise ValueError('Derived figures already exist; preserve or explicitly regenerate them')
    args.output.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(raw).hexdigest()
    candidate_figure(report, outputs[0], digest)
    conformal_figure(report, outputs[1], digest)
    print(json.dumps({'status': 'plotted_saved_aggregates', 'source_sha256': digest,
                      'figures': [path.name for path in outputs]}, indent=2))


if __name__ == '__main__':
    main()
