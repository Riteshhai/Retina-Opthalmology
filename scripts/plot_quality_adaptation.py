"""Derived figures use the immutable aggregate and saved development histories."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

ROOT=next(parent for parent in Path(__file__).resolve().parents if (parent/'pyproject.toml').is_file())
PUBLIC=ROOT/'reports/quality_adaptation_v1'
HEADS={'quality_grade':'Overall quality','clarity_class':'Clarity',
       'field_definition_class':'Field definition','artifact_class':'Artifact'}
COLORS={2026:'#0072B2',2027:'#D55E00',2028:'#009E73'}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    report=read(PUBLIC/'evaluation_v1_1.json')
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig=plt.figure(figsize=(11,12.5))
    grid=fig.add_gridspec(3,2,height_ratios=[1.15,1,1],hspace=.46,wspace=.28)
    ax=fig.add_subplot(grid[0,:])
    for run in report['runs']:
        seed,arm=run['seed'],run['arm']
        history=read(ROOT/f'checkpoints/quality_adaptation_v1/seed{seed}_{arm}/training_history.json')
        x=[row['epoch'] for row in history]
        y=[row['mean_head_macro_f1'] for row in history]
        ax.plot(x,y,color=COLORS[seed],linestyle='--' if arm=='frozen' else '-',
                label=f'{seed} {arm}',linewidth=1.8)
        chosen=run['selected_epoch']
        ax.scatter(chosen,y[chosen-1],color=COLORS[seed],marker='s' if arm=='frozen' else 'o',s=45,zorder=3)
    ax.set(xlim=(.6,15.4),ylim=(0,1),xticks=range(1,16),xlabel='Completed epoch (same fixed budget in both arms)',
           ylabel='Mean four-head macro-F1',title='All six trajectories; marks show selected trained epochs')
    ax.grid(alpha=.18)
    ax.legend(ncols=3,fontsize=9,loc='lower right')
    for index,(head,label) in enumerate(HEADS.items()):
        ax=fig.add_subplot(grid[1+index//2,index%2])
        for seed,color in COLORS.items():
            runs={r['arm']:r for r in report['runs'] if r['seed']==seed}
            warm=runs['frozen']['warm_metrics'][head]['macro_f1']
            pair=[runs[a]['selected_metrics'][head]['macro_f1'] for a in ('frozen','adapted')]
            ax.scatter(0,warm,marker='D',s=32,color=color,zorder=3)
            ax.plot([1,2],pair,marker='o',color=color,label=str(seed),linewidth=1.5)
        majority=report['majority_references'][head]['macro_f1']
        ax.axhline(majority,color='#666666',linestyle=':',label='Fitting majority')
        ax.set(xlim=(-.35,2.35),ylim=(0,1),xticks=[0,1,2],xticklabels=['Legacy warm','Matched frozen','Adapted'],
               ylabel='Macro-F1 (fixed full class universe)',title=label)
        ax.grid(axis='y',alpha=.18)
        if index==0:ax.legend(fontsize=8,ncols=2,loc='lower left')
    fig.suptitle('QSART quality adaptation v1 — reused development selection',fontsize=15,y=.978)
    fig.text(.5,.015,'168 images / 42 patients and split components; primary seed 2026.\n'
             'Warm comparisons also change the selector/fitting procedure. Production stays fixed.',ha='center',fontsize=10)
    fig.savefig(PUBLIC/'training_and_head_metrics.png',dpi=160,bbox_inches=None)
    plt.close(fig)

    fig=plt.figure(figsize=(11.5,12))
    grid=fig.add_gridspec(3,2,height_ratios=[1,1,1.05],hspace=.55,wspace=.28)
    for index,(head,label) in enumerate(HEADS.items()):
        ax=fig.add_subplot(grid[index//2,index%2])
        support=report['population']['class_supports'][head]['validation']
        fit=report['population']['class_supports'][head]['train']
        x=np.arange(len(support))
        for arm,offset,color in [('frozen',-.14,'#56B4E9'),('adapted',.14,'#CC79A7')]:
            values=np.array([r['selected_metrics'][head]['recall'] for r in report['runs'] if r['arm']==arm])
            ax.bar(x+offset,values.mean(0),width=.27,color=color,alpha=.65,label=arm+' mean')
            for seed_index,seed in enumerate(COLORS):
                ax.scatter(x+offset,values[seed_index],marker=['o','s','^'][seed_index],
                           color=COLORS[seed],s=18,zorder=3)
        ax.set(ylim=(0,1.06),xticks=x,xticklabels=[f'{i}\nn={n}' for i,n in enumerate(support)],
               ylabel='Selection recall',xlabel='Class and selection support',title=label)
        ax.grid(axis='y',alpha=.18)
        ax.text(.02,-.34,'Fitting supports: '+', '.join(map(str,fit)),transform=ax.transAxes,fontsize=9)
        if index==0:ax.legend(fontsize=8,ncols=2,loc='lower right')
    ax=fig.add_subplot(grid[2,:])
    labels=[];points=[];lower=[];upper=[];colors=[]
    for row in report['paired_changes']:
        values=row['metrics']['mean_head_macro_f1']
        labels.append(f"{row['seed']}: mean four-head F1")
        points.append(values['difference']);lower.append(values['lower_95']);upper.append(values['upper_95'])
        colors.append(COLORS[row['seed']])
    primary=next(row for row in report['paired_changes'] if row['seed']==2026)
    for head,label in HEADS.items():
        values=primary['metrics'][head+'_macro_f1']
        labels.append('Primary: '+('Field' if head=='field_definition_class' else label)+' F1')
        points.append(values['difference']);lower.append(values['lower_95']);upper.append(values['upper_95'])
        colors.append('#555555')
    for y,(point,lo,hi,color) in enumerate(zip(points,lower,upper,colors,strict=True)):
        ax.plot([lo,hi],[y,y],color=color,linewidth=2)
        ax.scatter(point,y,color=color,s=35,zorder=3)
    ax.axvline(0,color='#333333',linestyle='--',linewidth=1)
    ax.set(yticks=range(len(labels)),yticklabels=labels,xlabel='Adapted minus matched frozen; descriptive 95% interval',
           title='Paired F1 changes; all 27 intervals retained in JSON')
    ax.invert_yaxis();ax.grid(axis='x',alpha=.2)
    fig.suptitle('Class-level failures and paired development changes',fontsize=15,y=.975)
    fig.legend(handles=[Line2D([0],[0],marker=marker,color=color,linestyle='None',label=str(seed))
                        for (seed,color),marker in zip(COLORS.items(),['o','s','^'],strict=True)],
               ncols=3,loc='upper center',bbox_to_anchor=(.5,.955),frameon=False,fontsize=9)
    fig.text(.5,.012,'Points retain all 3 seeds. Field class 0 has one selection image.\n'
             '2,000 whole-component draws; selected checkpoints fixed; no multiplicity correction or clinical claim.',
             ha='center',fontsize=10)
    fig.subplots_adjust(left=.12,right=.97,top=.925,bottom=.085)
    ax.set_position([.30,.085,.67,.23])
    fig.savefig(PUBLIC/'class_recall_and_paired_changes.png',dpi=160,bbox_inches=None)
    plt.close(fig)
    print('Saved both quality-adaptation figures.')


if __name__=='__main__':main()
