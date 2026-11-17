"""Saved-prediction development comparison; no image access, fitting or selection."""
from __future__ import annotations

import numpy as np

from qsart.uncertainty.adaptive import fit_adaptive, predict_adaptive
from qsart.uncertainty.conformal import coverage_metrics, fit_conformal, predict_sets
from qsart.uncertainty.randomized_ordinal import fit_randomized, predict_randomized

STATES=('raw','argmax_fallback')
POPULATIONS=('all_selection_images','selection_group_representatives')


def representatives(values):
    text=values.astype(str)
    if not text.isin(['True','False']).all():raise ValueError('Explicit Boolean representative flags required')
    return text.eq('True').to_numpy()


def validate_saved_frames(calibration, selection):
    common={'image_id','split_group_id','dr_grade','is_group_representative',*[f'logit_{k}' for k in range(5)]}
    for frame in (calibration,selection):
        if frame.empty or not common.issubset(frame) or frame[list(common)].isna().any().any():
            raise ValueError('Complete saved development prediction rows required')
        if frame.image_id.duplicated().any() or frame.dr_grade.dtype.kind not in 'iu' or not frame.dr_grade.isin(range(5)).all():
            raise ValueError('Unique source identities and integer DR grades required')
        if not np.isfinite(frame[[f'logit_{k}' for k in range(5)]].to_numpy(float)).all():
            raise ValueError('Saved logits must be finite')
        flags=representatives(frame.is_group_representative)
        if not frame.assign(flag=flags).groupby('split_group_id').flag.sum().eq(1).all():
            raise ValueError('Exactly one representative per fingerprint component required')
        if frame.groupby('split_group_id').dr_grade.nunique().gt(1).any():raise ValueError('Conflicting component labels')
    for name in ('image_id','split_group_id'):
        if set(calibration[name])&set(selection[name]):raise ValueError('Calibration and selection overlap')
    if ('calibration_role' not in calibration or set(calibration.calibration_role)!={'temperature','conformal'}
        or calibration.groupby('split_group_id').calibration_role.nunique().gt(1).any()):
        raise ValueError('Separate frozen temperature/conformal component roles required')
    return (calibration.calibration_role.eq('conformal').to_numpy() & representatives(calibration.is_group_representative),
            representatives(selection.is_group_representative))


def probabilities(logits, temperature):
    if isinstance(temperature,bool) or not np.isfinite(temperature) or temperature<=0:
        raise ValueError('Fixed positive source temperature required')
    values=np.asarray(logits,np.float64)/temperature
    if values.ndim!=2 or values.shape[1]!=5 or not np.isfinite(values).all():raise ValueError('Five finite saved logits required')
    values=np.exp(values-values.max(1,keepdims=True))
    return values/values.sum(1,keepdims=True)


def describe(sets, truth, mask):
    output={}
    for name,indices in [(POPULATIONS[0],np.arange(len(truth))),(POPULATIONS[1],np.flatnonzero(mask))]:
        selected=[sets[i] for i in indices]
        output[name]={**coverage_metrics(selected,truth[indices]),
            'full_five_grade_set_count':sum(len(row)==5 for row in selected)}
    return output


def bootstrap_weights(groups, repeats, seed):
    groups=np.asarray(groups)
    if groups.ndim!=1 or not len(groups) or type(repeats) is not int or repeats<2:
        raise ValueError('Nonempty groups and at least two bootstrap draws required')
    unique,inverse=np.unique(groups,return_inverse=True)
    if len(unique)<2:raise ValueError('At least two fingerprint components required')
    draws=np.random.default_rng(seed).integers(0,len(unique),size=(repeats,len(unique)))
    counts=np.stack([np.bincount(draw,minlength=len(unique)) for draw in draws])
    return counts[:,inverse]


def paired_changes(before, after, truth, weights):
    if len(before)!=len(after) or len(before)!=len(truth) or weights.ndim!=2 or weights.shape[1]!=len(truth):
        raise ValueError('Paired sets, truth and shared bootstrap weights must align')
    first=np.array([y in row for y,row in zip(truth,before,strict=True)],float)
    second=np.array([y in row for y,row in zip(truth,after,strict=True)],float)
    size=np.array([len(row) for row in after])-np.array([len(row) for row in before])
    result={}
    for name,values,mask in [('mean_set_size',size,np.ones(len(truth),bool)),
        ('marginal_coverage',second-first,np.ones(len(truth),bool)),
        ('grade_3_inclusion',second-first,truth==3),('grade_4_inclusion',second-first,truth==4)]:
        denominator=weights[:,mask].sum(1)
        valid=denominator>0
        if not mask.any():result[name]={'delta':None,'lower':None,'upper':None,'valid_draws':0};continue
        distribution=(weights[:,mask]@values[mask])[valid]/denominator[valid]
        low,high=np.quantile(distribution,[.025,.975])
        result[name]={'delta':float(values[mask].mean()),'lower':float(low),'upper':float(high),
                      'valid_draws':int(valid.sum())}
    return result


def evaluate(calibration_p, calibration_y, selection_p, selection_y, groups, mask, config):
    streams={}
    for seed in config['uniform_seeds']:
        cal_seed,val_seed=np.random.SeedSequence(seed).spawn(2)
        streams[seed]=(np.random.default_rng(cal_seed).random(len(calibration_y)),
                       np.random.default_rng(val_seed).random(len(selection_y)))
    results=[];private=[];sets_by_key={}
    for alpha in config['alphas']:
        methods=[('lac',None),('ordinal_greedy',None)]+[(method,seed)
            for method in ('ordinal_greedy_randomized','aps','raps') for seed in config['uniform_seeds']]
        for method,seed in methods:
            if seed is None:
                artifact=fit_conformal(calibration_p,calibration_y,alpha=alpha,method=method,empty_policy='allow')
                raw=predict_sets(selection_p,artifact)
            else:
                cu,vu=streams[seed]
                if method=='ordinal_greedy_randomized':
                    artifact=fit_randomized(calibration_p,calibration_y,cu,alpha=alpha)
                    raw=predict_randomized(selection_p,artifact,vu,empty_policy='allow')
                else:
                    params={'k_reg':config['raps_k_reg'],'lam':config['raps_lambda']} if method=='raps' else {}
                    artifact=fit_adaptive(calibration_p,calibration_y,cu,alpha=alpha,method=method,**params)
                    raw=predict_adaptive(selection_p,artifact,vu,empty_policy='allow')
            fallback=[row if row else [int(np.argmax(selection_p[i]))] for i,row in enumerate(raw)]
            row={'method':method,'alpha':alpha,'randomization_seed':seed,'model_training_seed':config['model_training_seed'],
                 'artifact':artifact,'raw':describe(raw,selection_y,mask),'argmax_fallback':describe(fallback,selection_y,mask)}
            results.append(row)
            private.append({'method':method,'alpha':alpha,'randomization_seed':seed,'raw':raw,'argmax_fallback':fallback})
            sets_by_key[(alpha,method,seed)]={'raw':raw,'argmax_fallback':fallback}
    weights=bootstrap_weights(groups,config['bootstrap_draws'],config['bootstrap_seed'])
    contrasts=[]
    for alpha in config['alphas']:
        for seed in config['uniform_seeds']:
            for method in ('raps','ordinal_greedy_randomized'):
                for state in STATES:
                    changes=paired_changes(sets_by_key[(alpha,'aps',seed)][state],sets_by_key[(alpha,method,seed)][state],selection_y,weights)
                    contrasts.append({'alpha':alpha,'randomization_seed':seed,'empty_policy':state,
                        'direction':method+'_minus_aps','changes':changes})
    return results,contrasts,private
