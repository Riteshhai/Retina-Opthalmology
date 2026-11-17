"""Plot completed fine-tuning aggregates without reselecting a classifier or seed."""
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


SEEDS = (2026, 2027, 2028)
COLORS = ('#315f84', '#a76d35', '#358a7b')


def validate(report):
    if report.get('experiment_id') != 'aptos_finetune_v1' or report.get('dataset') != 'APTOS_2019':
        raise ValueError('Expected completed APTOS fine-tuning aggregates')
    if [row['training_seed'] for row in report['runs']] != list(SEEDS) or len(report['comparisons']) != 42:
        raise ValueError('All three training seeds and 42 conformal comparisons must remain visible')
    if report['summary']['primary_training_seed'] != 2026 or report['summary']['deployment_changed'] is not False:
        raise ValueError('The primary seed and unchanged-deployment declaration differ')


def grading_figure(report, path, digest):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    baseline = report['baseline']['grading_before_temperature']
    for axis, key, title in zip(axes.flat, ('accuracy', 'macro_f1', 'grade3', 'grade4'),
            ('Selection accuracy', 'Selection macro-F1', 'Grade-3 recall', 'Grade-4 recall'), strict=True):
        if key.startswith('grade'):
            grade = int(key[-1])
            values = [row['grading_before_temperature']['per_class'][grade]['recall'] for row in report['runs']]
            initial = baseline['per_class'][grade]['recall']
        else:
            values = [row['grading_before_temperature'][key] for row in report['runs']]
            initial = baseline[key]
        axis.bar(range(3), values, color=COLORS, width=.6)
        axis.axhline(initial, color='#b1484c', linestyle='--', linewidth=1.5, label='Frozen-feature warm start')
        for index, value in enumerate(values):
            axis.text(index, value+.018, f'{value:.2%}' if key != 'macro_f1' else f'{value:.4f}',
                      ha='center', fontsize=10)
        axis.set_xticks(range(3), [str(seed) for seed in SEEDS])
        axis.set(xlabel='Fine-tuning training seed', title=title, ylim=(0, 1.09))
        if key != 'macro_f1':
            axis.yaxis.set_major_formatter(PercentFormatter(1))
        axis.grid(axis='y', alpha=.18)
        axis.set_axisbelow(True)
        axis.legend(loc='lower right', fontsize=8)
    fig.suptitle('APTOS parameter fine-tuning · all three declared training seeds', fontsize=17)
    fig.text(.5, .065, 'Same 521 selection images and common previously selected initialization; primary seed 2026 fixed before outcomes.',
             ha='center', fontsize=9)
    fig.text(.5, .041, 'Selection labels also choose epochs. These are development diagnostics; seed variability is conditional, not fresh-test evidence.',
             ha='center', fontsize=9)
    fig.text(.5, .019, f'Raw grading before temperature; original deployed model unchanged. Source aggregate SHA256: {digest[:18]}…',
             ha='center', fontsize=9, color='#526070')
    fig.subplots_adjust(top=.9, bottom=.15, hspace=.37, wspace=.26)
    fig.savefig(path, dpi=160, facecolor='white')
    plt.close(fig)


def conformal_figure(report, path, digest):
    families = ('lac', 'ordinal_greedy', 'ordinal_greedy_randomized')
    labels = ('LAC', 'Ordinal\nfixed', 'Ordinal\nrandom')
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    for column, alpha in enumerate((.1, .05)):
        for row_index, metric in enumerate(('marginal_coverage', 'mean_set_size')):
            axis = axes[row_index, column]
            points = []
            for seed in SEEDS:
                for family in families:
                    rows = [row for row in report['comparisons'] if row['model_training_seed'] == seed and
                            row['alpha'] == alpha and row['method'] == family]
                    if len(rows) != (5 if family.endswith('randomized') else 1):
                        raise ValueError('Conformal comparison grid is incomplete')
                    raw = np.mean([row['raw']['all_selection_images'][metric] for row in rows])
                    fallback = np.mean([row['argmax_fallback']['all_selection_images'][metric] for row in rows])
                    points.append((seed, family, raw, fallback))
            for index, (_, family, raw, fallback) in enumerate(points):
                color = COLORS[families.index(family)]
                axis.bar(index, fallback, color=color, alpha=.75, width=.65)
                axis.plot(index, raw, 'o', color='#152e43', markersize=5)
            axis.set_xticks(range(9), [f'{str(seed)[-2:]}\n{labels[families.index(family)]}'
                                      for seed, family, _, _ in points], fontsize=8)
            axis.set_xlabel('Training seed ending / conformal method', fontsize=9)
            axis.grid(axis='y', alpha=.18)
            axis.set_axisbelow(True)
            if metric == 'marginal_coverage':
                axis.set(ylim=(0, 1.08), ylabel='Empirical selection inclusion', title=f'Alpha {alpha:.2f} · inclusion')
                axis.yaxis.set_major_formatter(PercentFormatter(1))
                axis.axhline(1-alpha, color='#b1484c', linestyle='--', linewidth=1.3)
            else:
                axis.set(ylim=(0, 5.45), ylabel='Mean number of grades', title=f'Alpha {alpha:.2f} · set size')
    fig.suptitle('APTOS fine-tuning · raw sets and common argmax fallback', fontsize=17)
    fig.text(.5, .078, 'Bars: common fallback. Dots: raw inversion. Random means retain all five uniform seeds per classifier, with full rows in the report.',
             ha='center', fontsize=9)
    fig.text(.5, .053, 'Dashed 90% / 95% guides are not coverage guarantees. All 521 previously inspected selection images; no patient or grade-conditional guarantee.',
             ha='center', fontsize=9)
    fig.text(.5, .027, f'Training seeds differ from conformal uniform seeds. Source aggregate SHA256: {digest[:18]}…',
             ha='center', fontsize=9, color='#526070')
    fig.subplots_adjust(top=.91, bottom=.17, hspace=.39, wspace=.23)
    fig.savefig(path, dpi=160, facecolor='white')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('reports/aptos_finetune_v1/evaluation.json'))
    parser.add_argument('--output', type=Path, default=Path('reports/aptos_finetune_v1'))
    parser.add_argument('--overwrite-derived', action='store_true')
    args = parser.parse_args()
    raw = args.source.read_bytes()
    report = json.loads(raw)
    validate(report)
    args.output.mkdir(parents=True, exist_ok=True)
    paths = (args.output/'grading_by_seed.png', args.output/'conformal_diagnostics.png')
    if any(path.exists() for path in paths) and not args.overwrite_derived:
        raise ValueError('Existing derived plots require --overwrite-derived; scientific report is never changed')
    digest = hashlib.sha256(raw).hexdigest()
    grading_figure(report, paths[0], digest)
    conformal_figure(report, paths[1], digest)
    if args.source.read_bytes() != raw:
        raise ValueError('Scientific aggregate changed during plotting')
    print(json.dumps({'source_sha256': digest, 'figures': [path.name for path in paths]}, indent=2))


if __name__ == '__main__':
    main()
