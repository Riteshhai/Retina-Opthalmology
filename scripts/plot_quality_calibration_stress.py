"""Derive quality-study figures from the immutable aggregate only."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
PUBLIC=ROOT/'reports/quality_calibration_stress_v1'
HEADS={'quality_grade':'Overall quality','clarity_class':'Clarity',
       'field_definition_class':'Field definition','artifact_class':'Artifact'}
COLORS={2026:'#0072B2',2027:'#D55E00',2028:'#009E73'}


def main():
    report=json.loads((PUBLIC/'evaluation.json').read_text())
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(10.5,9))
    for ax,(head,label) in zip(axes.flat,HEADS.items(),strict=True):
        for run in report['runs']:
            y=[run['clean_metrics']['validation'][state][head]['nll'] for state in ('raw','temperature')]
            ax.plot([0,1],y,color=COLORS[run['training_seed']],linestyle='--' if run['arm']=='frozen' else '-',
                    marker='s' if run['arm']=='frozen' else 'o',linewidth=1.6)
        ax.set(xlim=(-.25,1.25),ylim=(0,1.5),xticks=[0,1],xticklabels=['Raw','Temperature scaled'],
               ylabel='Clean selection NLL',title=label)
        ax.grid(axis='y',alpha=.2)
    handles=[Line2D([0],[0],color=color,label=str(seed)) for seed,color in COLORS.items()]
    handles += [Line2D([0],[0],color='#555555',linestyle=style,marker=marker,label=arm)
                for style,marker,arm in [('--','s','Frozen'),('-','o','Adapted')]]
    fig.suptitle('Quality calibration — all six fixed models',fontsize=15,y=.98)
    fig.legend(handles=handles,ncols=5,loc='upper center',bbox_to_anchor=(.5,.945),frameon=False)
    fig.subplots_adjust(left=.09,right=.97,top=.85,bottom=.13,hspace=.38,wspace=.25)
    fig.text(.5,.025,'Selection: 168 images / 42 patients; temperatures fitted on disjoint-role 168 / 42.\n'
             'Development roles are historically reused. Class predictions and production models remain unchanged.',
             ha='center',fontsize=9)
    fig.savefig(PUBLIC/'calibration_diagnostics.png',dpi=160)
    plt.close(fig)

    fig,axes=plt.subplots(2,2,figsize=(11.5,10))
    conditions=['blur_1','blur_2','exposure_0_5','exposure_1_5','jpeg_50','jpeg_20']
    labels=['Blur 1','Blur 2','sRGB ×0.5','sRGB ×1.5','JPEG50','JPEG20']
    responses={'poor_probability':'Poor-quality probability', 'expected_clarity':'Normalized expected clarity class',
               'expected_field_definition':'Normalized expected field class','expected_artifact':'Normalized expected artifact class'}
    x=np.arange(6)
    for ax,(response,label) in zip(axes.flat,responses.items(),strict=True):
        for arm,color in [('frozen','#0072B2'),('adapted','#D55E00')]:
            run=next(r for r in report['runs'] if r['training_seed']==2026 and r['arm']==arm)
            rows={r['condition']:r for r in run['stress']}
            for state,style in [('raw','--'),('temperature','-')]:
                data=[rows[c][state]['changes'][response] for c in conditions]
                ax.plot(x,[v['difference'] for v in data],color=color,linestyle=style,marker='o',
                        label=arm+' / '+state,linewidth=1.5)
                if state=='temperature':
                    ax.fill_between(x,[v['lower_95'] for v in data],[v['upper_95'] for v in data],color=color,alpha=.12)
        ax.axhline(0,color='#444444',linestyle=':',linewidth=1)
        ax.set(xticks=x,xticklabels=labels,ylabel='Change from the same clean image',title=label)
        ax.tick_params(axis='x',rotation=25)
        ax.grid(axis='y',alpha=.18)
    fig.suptitle('Fixed primary seed 2026 — score responses to altered inputs',fontsize=15,y=.98)
    handles,legend_labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,legend_labels,ncols=4,loc='upper center',bbox_to_anchor=(.5,.945),frameon=False,fontsize=9)
    fig.subplots_adjust(left=.09,right=.98,top=.855,bottom=.16,hspace=.55,wspace=.28)
    fig.text(.5,.035,'168 selection images / 42 components. Bands: descriptive 95% group intervals for scaled responses only.\n'
             'No new corrupted-quality truth: these are score shifts, not defect-detection accuracy or clinical benefit.\n'
             'All models, states, conditions and 486 declared intervals remain in the aggregate.',ha='center',fontsize=9)
    fig.savefig(PUBLIC/'stress_responses.png',dpi=160)
    plt.close(fig)
    print('Saved both quality calibration/stress figures.')


if __name__=='__main__':main()
