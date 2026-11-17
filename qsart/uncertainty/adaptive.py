"""Research APS/RAPS with explicit uniforms and exact conformal ranks.

Independently implements the ranked-mass score described by Angelopoulos et al.
(arXiv:2009.14193v5). One supplied uniform is shared across candidate labels.
The convention is preceding mass + (1-u)*p + lambda*max(rank-k_reg,0).
RAPS scores and thresholds may exceed one. A caller must keep classifier fitting,
temperature/parameter selection and threshold calibration separate. Image-level
artifacts cannot establish patient, grade-conditional or shifted-domain coverage.
Existing production sidecars and historical constructions are unchanged.
"""
from __future__ import annotations

import math

import numpy as np

from qsart.uncertainty.conformal import (
    _alpha, _conformal_rank, _labels, _metadata, _probabilities, conformal_quantile,
)
from qsart.uncertainty.randomized_ordinal import _uniforms

SCHEMA = 'qsart.adaptive_research.v1'
TIES = 'descending_probability;lowest_grade_on_equal_probability;inclusive_threshold'
RANDOMIZATION = 'preceding_mass_plus_(1-u)_times_candidate_probability;one_uniform_per_row'
KEYS = {'schema','method','alpha','num_classes','n_calibration_units','rank','qhat',
        'k_reg','lam','unit','tie_policy','randomization','metadata'}


def _parameters(method, k_reg, lam, classes):
    if not isinstance(method,str) or method not in {'aps','raps'}:
        raise ValueError('Choose APS or RAPS explicitly')
    if type(k_reg) is not int or not 1 <= k_reg <= classes:
        raise ValueError('k_reg must be an integer rank in the class range')
    if isinstance(lam,(bool,str)) or not isinstance(lam,(int,float)) or not math.isfinite(lam) or lam < 0:
        raise ValueError('lambda must be a finite nonnegative number')
    if method=='aps' and (k_reg!=1 or lam!=0):
        raise ValueError('APS requires k_reg=1 and lambda=0; no implicit ignored penalty')
    return float(lam)


def adaptive_scores(probabilities, uniforms, *, method='aps', k_reg=1, lam=0.):
    """Return candidate scores in original class order, with stable rank ties."""
    p=_probabilities(probabilities);u=_uniforms(uniforms,len(p))
    lam=_parameters(method,k_reg,lam,p.shape[1])
    order=np.argsort(-p,axis=1,kind='stable')
    ranked=np.take_along_axis(p,order,axis=1)
    preceding=np.column_stack([np.zeros(len(p)),np.cumsum(ranked[:,:-1],axis=1)])
    mass=np.minimum(preceding+(1-u[:,None])*ranked,1.)
    ranked_scores=mass+lam*np.maximum(np.arange(1,p.shape[1]+1)-k_reg,0)
    result=np.empty_like(p);np.put_along_axis(result,order,ranked_scores,axis=1)
    return result


def fit_adaptive(probabilities, labels, uniforms, *, alpha=.1, method='aps', k_reg=1, lam=0., metadata=None):
    p=_probabilities(probabilities);y=_labels(labels,len(p),p.shape[1]);alpha=_alpha(alpha)
    scores=adaptive_scores(p,uniforms,method=method,k_reg=k_reg,lam=lam)[np.arange(len(y)),y]
    threshold=conformal_quantile(scores,alpha)
    return validate_adaptive({'schema':SCHEMA,'method':method,'alpha':alpha,'num_classes':p.shape[1],
        'n_calibration_units':len(y),'rank':_conformal_rank(len(y),alpha),
        'qhat':None if math.isinf(threshold) else threshold,'k_reg':k_reg,'lam':float(lam),
        'unit':'image','tie_policy':TIES,'randomization':RANDOMIZATION,
        'metadata':{} if metadata is None else metadata})


def validate_adaptive(artifact):
    if not isinstance(artifact,dict) or set(artifact)!=KEYS or artifact.get('schema')!=SCHEMA:
        raise ValueError('Unrecognized adaptive research artifact')
    for name in ('num_classes','n_calibration_units','rank'):
        if type(artifact[name]) is not int or artifact[name]<1:raise ValueError('Invalid adaptive counts/rank')
    if artifact['num_classes']<2:raise ValueError('At least two classes required')
    alpha=_alpha(artifact['alpha']);n=artifact['n_calibration_units']
    lam=_parameters(artifact['method'],artifact['k_reg'],artifact['lam'],artifact['num_classes'])
    if artifact['rank']!=_conformal_rank(n,alpha):raise ValueError('Exact conformal rank differs')
    q=artifact['qhat'];maximum=1+lam*(artifact['num_classes']-artifact['k_reg'])
    if q is None:
        if artifact['rank']<=n:raise ValueError('Null threshold requires rank beyond calibration size')
    elif (isinstance(q,(bool,str)) or not isinstance(q,(int,float)) or not math.isfinite(q)
          or not 0<=q<=maximum or artifact['rank']>n):
        raise ValueError('Invalid adaptive threshold')
    if artifact['unit']!='image' or artifact['tie_policy']!=TIES or artifact['randomization']!=RANDOMIZATION:
        raise ValueError('Unsupported adaptive calibration/randomization convention')
    return {**artifact,'alpha':alpha,'lam':lam,'metadata':_metadata(artifact['metadata'])}


def predict_adaptive(probabilities, artifact, uniforms, *, empty_policy='allow'):
    artifact=validate_adaptive(artifact);p=_probabilities(probabilities)
    if p.shape[1]!=artifact['num_classes']:raise ValueError('Class count differs from calibration')
    if not isinstance(empty_policy,str) or empty_policy not in {'allow','argmax'}:
        raise ValueError('Explicit supported empty-set policy required')
    scores=adaptive_scores(p,uniforms,method=artifact['method'],k_reg=artifact['k_reg'],lam=artifact['lam'])
    mask=scores <= (math.inf if artifact['qhat'] is None else artifact['qhat'])
    if empty_policy=='argmax':
        empty=np.flatnonzero(~mask.any(1));mask[empty,np.argmax(p[empty],axis=1)]=True
    return [np.flatnonzero(row).tolist() for row in mask]
