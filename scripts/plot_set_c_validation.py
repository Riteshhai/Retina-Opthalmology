"""Plot published patient dispositions from the audited Set-C validation."""

from hashlib import sha256
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.patches import Patch


REPORTS = Path(__file__).resolve().parents[1] / 'reports' / 'set_c_validation_v1'
MODES = ('ordinal', 'lac', 'no_cp')
TITLES = ('Deployed ordinal gate', 'LAC diagnostic', 'No-conformal diagnostic')
SERIES = (
    ('AUTO_NONREFERABLE', 'Fully automatic', '#207365'),
    ('RECAPTURE', 'Recapture', '#cb901e'),
    ('SPECIALIST_REVIEW', 'Specialist review', '#406498'),
)


def main():
    payload = (REPORTS / 'evaluation.json').read_bytes()
    report = json.loads(payload)
    audit = json.loads((REPORTS / 'independent_audit.json').read_bytes())
    if audit['audit_status'] != 'passed' or audit['evaluation_sha256'] != sha256(payload).hexdigest():
        raise ValueError('The report must match the passed independent audit')
    output = REPORTS / 'patient_dispositions.png'
    if output.exists():
        raise FileExistsError('A published figure already exists')

    figure, axes = plt.subplots(1, 3, sharey=True, figsize=(11, 4.65))
    figure.subplots_adjust(left=.08, right=.98, top=.80, bottom=.26, wspace=.15)
    figure.suptitle('DeepDRiD Set-C patient dispositions', x=.08, y=.96, ha='left', fontsize=17, weight='bold')
    figure.text(.08, .885, '99 patients · four views per patient · image-based dual-budget allocation',
                color='#40536a', fontsize=10)
    axes[0].set_ylabel('Patients')
    for mode, title, axis in zip(MODES, TITLES, axes, strict=True):
        scenarios = [row for row in report['scenarios'] if row['mode'] == mode and row['method'] == 'dual_budget']
        axis.set_title(title, fontsize=11, loc='left', weight='bold', pad=12)
        axis.set_ylim(0, 106)
        axis.set_xlim(-.58, 3.58)
        axis.set_xticks(range(4), [f'{review:.0%}\n{recapture:.0%}' for review, recapture in report['budget_pairs']])
        axis.tick_params(length=0)
        axis.set_axisbelow(True)
        axis.grid(axis='y', color='#e3e8ef', linewidth=.8)
        for spine in axis.spines.values():
            spine.set_visible(False)
        for index, scenario in enumerate(scenarios):
            if scenario['outcomes'] is None:
                axis.text(index, 45, 'Overload', ha='center', va='center', rotation=90,
                          fontsize=10, color='#8d4144')
                continue
            counts = scenario['outcomes']['patient_actions']
            base = 0
            for action, _, color in SERIES:
                count = counts[action]
                axis.bar(index, count, .66, bottom=base, color=color)
                if count >= 8:
                    axis.text(index, base + count / 2, str(count), ha='center', va='center',
                              color='white', fontsize=9, weight='bold')
                base += count
    figure.text(.53, .14, 'Image review cap / image recapture cap', ha='center', fontsize=9, color='#40536a')
    figure.legend(handles=[Patch(facecolor=color, label=label) for _, label, color in SERIES],
                  loc='lower center', bbox_to_anchor=(.53, .025), ncol=3, frameon=False, fontsize=9)
    with output.open('xb') as stream:
        figure.savefig(stream, format='png', dpi=175)
    plt.close(figure)


if __name__ == '__main__':
    main()
