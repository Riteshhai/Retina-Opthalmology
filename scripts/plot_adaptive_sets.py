"""Plot only frozen adaptive-set results; no refitting or score recomputation."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    root=Path(__file__).resolve().parents[1];folder=root/'reports/adaptive_sets_v1'
    report=json.loads((folder/'evaluation.json').read_text(encoding='utf-8'))
    methods=['lac','aps','raps','ordinal_greedy_randomized','ordinal_greedy']
    labels=['LAC','APS','RAPS\n(fixed)','Ordinal\nrandomized','Ordinal\nfull mass']
    rows=[next(row for row in report['comparisons'] if row['method']==method and row['alpha']==.1
        and row['randomization_seed'] in (None,2026)) for method in methods]
    fig,axes=plt.subplots(1,3,figsize=(13.8,4.7),layout='constrained')
    x=np.arange(len(rows));width=.35
    colors={'raw':'#0072B2','argmax_fallback':'#E69F00'}
    for state,offset,title in [('raw',-width/2,'Raw'),('argmax_fallback',width/2,'Argmax fallback')]:
        values=[row[state]['all_selection_images'] for row in rows]
        axes[0].bar(x+offset,[v['mean_set_size'] for v in values],width,color=colors[state],label=title)
        axes[1].bar(x+offset,[v['marginal_coverage'] for v in values],width,color=colors[state])
    raw=[row['raw']['all_selection_images'] for row in rows]
    for grade,offset,color in [('3',-width/2,'#009E73'),('4',width/2,'#CC79A7')]:
        axes[2].bar(x+offset,[v['per_class'][grade]['coverage'] for v in raw],width,color=color,
                    label=f'Grade {grade} (n={raw[0]["per_class"][grade]["n"]})')
    axes[0].set(title='Set size',ylabel='Mean grades included',ylim=(0,5.25))
    axes[1].set(title='Overall inclusion',ylabel='Fraction of 521 selection rows',ylim=(0,1.05))
    axes[2].set(title='Severe-grade inclusion: raw sets',ylabel='Within-grade fraction',ylim=(0,1.05))
    for axis in axes:
        axis.set_xticks(x,labels);axis.tick_params(axis='x',labelsize=8)
        axis.spines[['top','right']].set_visible(False);axis.grid(axis='y',alpha=.18);axis.set_axisbelow(True)
    for axis in axes[1:]:axis.axhline(.9,color='#444444',ls='--',lw=1,label='90% marginal target')
    axes[0].legend(fontsize=8,loc='upper right');axes[2].legend(fontsize=8,loc='lower left')
    fig.suptitle('Fixed APTOS development model | alpha 0.10 | uniform seed 2026',fontsize=13)
    fig.text(.5,-.025,'Reused development data; fixed RAPS k=2, lambda=0.10. Marginal target is not a grade-coverage guarantee. '
             'No model/method selection or clinical validation.',ha='center',fontsize=8)
    for ext in ('png','svg'):
        path=folder/f'primary_set_comparison.{ext}'
        if path.exists():raise ValueError('Preserve existing measured figure')
        fig.savefig(path,dpi=180,bbox_inches='tight')
    plt.close(fig)
    print('Saved primary_set_comparison.png and .svg from the frozen report.')


if __name__=='__main__':main()
