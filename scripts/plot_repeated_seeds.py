"""Plot completed aggregate seed outcomes; never select a seed or retrain."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    root = Path(__file__).resolve().parents[1]
    report = json.loads((root/'reports/repeated_seeds_v1/summary.json').read_text(encoding='utf-8'))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    dr = sorted((row for row in report['runs'] if row['task'] == 'dr'), key=lambda row: row['seed'])
    for key, label, marker in (('accuracy', 'DR accuracy', 'o'), ('macro_f1', 'DR macro-F1', 's')):
        axes[0].plot([str(row['seed']) for row in dr], [row['metrics']['dr_grade'][key] for row in dr],
                     marker+'-', label=label)
    axes[0].set_xlabel('Training seed')
    axes[0].set_ylim(0, 1)
    axes[0].legend()
    axes[0].set_title('All three DR runs')
    quality = report['summary']['quality']
    heads = ('quality_grade', 'clarity_class', 'field_definition_class', 'artifact_class')
    means = [quality[head]['accuracy']['mean'] for head in heads]
    sd = [quality[head]['accuracy']['sample_standard_deviation'] for head in heads]
    axes[1].bar(range(4), means, color='#468e9c', yerr=sd, capsize=5)
    for offset, run in zip((-.07, 0, .07), sorted((r for r in report['runs'] if r['task'] == 'quality'),
                                               key=lambda row: row['seed']), strict=True):
        axes[1].scatter(np.arange(4)+offset, [run['metrics'][head]['accuracy'] for head in heads],
                        label=str(run['seed']), s=22, zorder=3)
    axes[1].set_xticks(range(4), ('Overall quality', 'Clarity', 'Field definition', 'Artifact'), rotation=15)
    axes[1].set_ylim(0, 1)
    axes[1].set_title('Quality/defect accuracy: all seeds, mean and sample SD')
    axes[1].legend(title='Training seed', fontsize=8)
    for axis in axes:
        axis.grid(axis='y', alpha=.2)
    fig.suptitle('Cache-independent head initialization | fixed development selection, 168 images / 42 patients\n'
                 'Three training seeds; SD does not measure cohort/domain uncertainty')
    fig.savefig(root/'reports/repeated_seeds_v1/selection_seed_variability.png', dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    main()
