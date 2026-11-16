"""Plot the audited acquisition ledger at the fixed extra-view budgets."""

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator


ROOT = Path(__file__).resolve().parents[1] / 'reports' / 'paired_view_v1'
METHODS = {
    'voi': ('Estimated value', '#1e7567', 'o'),
    'quality': ('Lowest quality', '#d18a20', 's'),
    'entropy': ('Highest entropy', '#496f9e', '^'),
    'referable_risk': ('Highest referable risk', '#8d467b', 'D'),
    'random': ('Seeded random', '#677282', 'x'),
    'voi_positive': ('Estimated value above cost', '#25383e', 'P'),
}


def main():
    payload = (ROOT / 'evaluation.json').read_bytes()
    receipt = json.loads((ROOT / 'independent_audit.json').read_bytes())
    if receipt['audit_status'] != 'passed' or receipt['evaluation_sha256'] != hashlib.sha256(payload).hexdigest():
        raise ValueError('The report must match the passed independent audit')
    report = json.loads(payload)
    scenarios = report['scenarios']
    anchors = scenarios[0], scenarios[-1]
    figure, axis = plt.subplots(figsize=(10, 5.5))
    figure.subplots_adjust(left=.105, right=.98, top=.80, bottom=.24)
    figure.suptitle('DeepDRiD paired-view acquisition', x=.105, y=.96, ha='left', fontsize=19, weight='bold')
    figure.text(.105, .88, '84 eyes from 42 selection patients · lower ledger values indicate lower proxy cost',
                fontsize=10, color='#43546a')
    for method, (label, color, marker) in METHODS.items():
        rows = [row for row in scenarios if row['method'] == method]
        if method != 'voi_positive':
            rows = [anchors[0], *rows, anchors[1]]
        points = list(dict.fromkeys((row['outcomes']['extra_views'], row['outcomes']['ledger_per_eye'])
                                    for row in rows))
        axis.plot(*zip(*points, strict=True), marker=marker, color=color, label=label,
                  linestyle='none' if method == 'voi_positive' else '-',
                  markersize=6 if method == 'voi_positive' else 4, linewidth=2 if method == 'voi' else 1.2)
    axis.set_xlabel('Additional recorded views acquired')
    axis.set_ylabel('Mean research ledger per eye')
    axis.set_xticks([0, 8, 21, 42, 63, 84])
    axis.yaxis.set_major_locator(MaxNLocator(nbins=5))
    axis.grid(axis='y', color='#e3e8ee')
    axis.set_axisbelow(True)
    axis.spines[['top', 'right']].set_visible(False)
    axis.legend(loc='upper center', bbox_to_anchor=(.5, -.23), frameon=False, ncol=3, fontsize=9)
    with (ROOT / 'acquisition_workload.png').open('xb') as stream:
        figure.savefig(stream, format='png', dpi=175)
    plt.close(figure)


if __name__ == '__main__':
    main()
