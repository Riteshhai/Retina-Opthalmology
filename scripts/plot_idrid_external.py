"""Derived scientific figures for the completed locked external report only."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'reports/idrid_external_v1/evaluation.json'
OUTPUT = SOURCE.parent


def main(overwrite_derived=False):
    data = SOURCE.read_bytes(); report = json.loads(data)
    assert len(report['models']) == 4 and len(report['comparisons']) == 56
    assert report['summary']['primary_model'] == 'finetune_seed2026'
    digest = hashlib.sha256(data).hexdigest()
    paths = [OUTPUT/'external_grading.png', OUTPUT/'external_conformal.png']
    if any(path.exists() for path in paths) and not overwrite_derived:
        raise ValueError('Existing derived figures require --overwrite-derived; the report is never changed')
    colors = ['#315f84', '#a76d35', '#358a7b']
    populations = list(report['summary']['population_counts'])
    short = ['Full 516', 'Filtered 513', 'Representatives 505']
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for axis, metric, title in zip(axes.flat, ['accuracy', 'macro_f1', 'grade3', 'grade4'],
            ['Accuracy against supplied labels', 'Macro-F1 against supplied labels', 'Grade-3 recall', 'Grade-4 recall'], strict=True):
        for index, population in enumerate(populations):
            grading = [model['grading'][population] for model in report['models']]
            values = [g['per_class'][int(metric[-1])]['recall'] if metric.startswith('grade') else g[metric] for g in grading]
            axis.bar(np.arange(4)+(index-1)*.24, values, width=.23, color=colors[index], label=short[index])
        axis.set_xticks(range(4), ['Frozen\nbaseline', 'Fine-tune\n2026 (primary)', 'Fine-tune\n2027', 'Fine-tune\n2028'])
        axis.set(title=title, ylim=(0, 1.04))
        axis.yaxis.set_major_formatter(PercentFormatter(1)); axis.grid(axis='y', alpha=.18);axis.set_axisbelow(True)
        axis.legend(loc='upper right', fontsize=8)
    fig.suptitle('IDRiD external-domain diagnostics · all four locked classifiers and three populations', fontsize=15)
    fig.text(.5, .065, 'Source temperatures/checkpoints fixed before outcomes. All four annotation-conflicting components remain flagged.', ha='center', fontsize=9)
    fig.text(.5, .04, 'Fingerprint representatives are not patients. Models are not deployed; external outcomes are now consumed.', ha='center', fontsize=9)
    fig.text(.5, .016, f'Aggregate SHA256: {digest[:24]}…', ha='center', fontsize=9, color='#526070')
    fig.subplots_adjust(top=.9, bottom=.15, hspace=.33, wspace=.23)
    fig.savefig(paths[0], dpi=160, facecolor='white'); plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    main_population = report['summary']['primary_population']
    families = ['lac', 'ordinal_greedy', 'ordinal_greedy_randomized']
    names = ['BL', '26', '27', '28']
    for column, alpha in enumerate([.1, .05]):
        for row_index, metric in enumerate(['marginal_coverage', 'mean_set_size']):
            axis = axes[row_index, column]
            for index, model in enumerate(report['models']):
                for family_index, family in enumerate(families):
                    rows = [r for r in report['comparisons'] if r['model_id'] == model['model_id'] and r['alpha'] == alpha and r['method'] == family]
                    assert len(rows) == (5 if family.endswith('randomized') else 1)
                    raw = np.mean([r['raw'][main_population][metric] for r in rows])
                    enlarged = np.mean([r['argmax_fallback'][main_population][metric] for r in rows])
                    x = 3*index+family_index
                    axis.bar(x, enlarged, color=colors[family_index], width=.65, alpha=.78)
                    axis.plot(x, raw, 'o', color='#152e43', markersize=4)
            axis.set_xticks(range(12), [f'{name}\n{method}' for name in names for method in ['LAC', 'Ordinal\nfixed', 'Ordinal\nrandom']], fontsize=7)
            axis.grid(axis='y', alpha=.18);axis.set_axisbelow(True)
            axis.set_title(f'Alpha {alpha:.2f} · '+('inclusion' if metric == 'marginal_coverage' else 'set size'))
            if metric == 'marginal_coverage':
                axis.set(ylim=(0, 1.08), ylabel='Empirical inclusion')
                axis.yaxis.set_major_formatter(PercentFormatter(1));axis.axhline(1-alpha, linestyle='--', color='#b1484c')
            else: axis.set(ylim=(0, 5.3), ylabel='Mean number of grades')
    fig.suptitle('IDRiD · frozen APTOS thresholds on the predeclared 505-representative population', fontsize=15)
    fig.text(.5, .110, 'BL: frozen baseline. 26 / 27 / 28: fine-tuning training seeds 2026 / 2027 / 2028.', ha='center', fontsize=9)
    fig.text(.5, .083, 'Bars: common argmax fallback. Dots: raw inversion. Random values average all five fixed uniform seeds; none is selected.', ha='center', fontsize=9)
    fig.text(.5, .055, 'Dashed 90% / 95% guides are not coverage guarantees under shift. Full and filtered observation views remain in the report.', ha='center', fontsize=9)
    fig.text(.5, .027, f'No external fitting, no corrected clinical labels. Aggregate SHA256: {digest[:24]}…', ha='center', fontsize=9, color='#526070')
    fig.subplots_adjust(top=.91, bottom=.18, hspace=.36, wspace=.23)
    fig.savefig(paths[1], dpi=160, facecolor='white');plt.close(fig)
    assert SOURCE.read_bytes() == data
    print(json.dumps({'source_sha256': digest, 'figures': [p.name for p in paths]}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--overwrite-derived', action='store_true')
    main(parser.parse_args().overwrite_derived)
