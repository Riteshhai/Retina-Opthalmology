"""Hand-calculated APS/RAPS and explicit research-boundary software fixtures."""
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from qsart.evaluation_adaptive import bootstrap_weights, paired_changes, validate_saved_frames
from qsart.uncertainty.adaptive import adaptive_scores, fit_adaptive, predict_adaptive, validate_adaptive
from qsart.uncertainty.conformal import nonconformity_scores


def test_ranked_mass_and_raps_penalty_have_hand_calculated_values():
    p=np.array([[.1,.4,.2,.3]])
    np.testing.assert_allclose(adaptive_scores(p,[.25]),[[.975,.3,.85,.625]],atol=1e-15)
    np.testing.assert_allclose(adaptive_scores(p,[.25],method='raps',k_reg=2,lam=.2),[[1.375,.3,1.05,.625]],atol=1e-15)
    np.testing.assert_array_equal(p,[[.1,.4,.2,.3]])


def test_ties_are_grade_stable_lambda_zero_is_exact_aps_and_no_rng_is_advanced():
    np.random.seed(91);state=np.random.get_state()
    p=np.full((2,4),.25)
    aps=adaptive_scores(p,[.5,1.])
    np.testing.assert_array_equal(aps,[[.125,.375,.625,.875],[0,.25,.5,.75]])
    np.testing.assert_array_equal(aps,adaptive_scores(p,[.5,1.],method='raps',k_reg=2,lam=0.))
    after=np.random.get_state()
    assert state[0]==after[0] and np.array_equal(state[1],after[1]) and state[2:]==after[2:]


def test_exact_rank_and_valid_raps_threshold_above_one_are_preserved():
    p=np.tile([.1,.4,.2,.3],(10,1));u=np.linspace(.01,.99,10);y=np.zeros(10,int)
    artifact=fit_adaptive(p,y,u,method='raps',k_reg=2,lam=.2,alpha=.2)
    assert artifact['rank']==9 and artifact['qhat']>1
    assert artifact['qhat']==sorted(adaptive_scores(p,u,method='raps',k_reg=2,lam=.2)[:,0])[8]
    assert validate_adaptive(artifact)==artifact


def test_infinity_small_sample_and_empty_fallback_are_explicit():
    p=np.tile([.1,.9],(10,1));y=np.ones(10,int)
    small=fit_adaptive(p[:2],y[:2],[.1,.2],alpha=.1)
    assert small['qhat'] is None and predict_adaptive(p[:1],small,[0])==[[0,1]]
    tight=fit_adaptive(p,y,np.full(10,.99),alpha=.2)
    assert predict_adaptive(p[:1],tight,[0],empty_policy='allow')==[[]]
    assert predict_adaptive(p[:1],tight,[0],empty_policy='argmax')==[[1]]


def test_rank_sets_can_be_noncontiguous_and_inclusive_threshold_keeps_ties():
    p=np.tile([.4,.01,.3,.02,.27],(10,1))
    artifact=fit_adaptive(p,np.full(10,2),np.zeros(10),alpha=.2)
    assert predict_adaptive(p[:1],artifact,[0])==[[0,2]]
    assert adaptive_scores(p[:1],[0])[0,2]==artifact['qhat']
    assert not np.array_equal(adaptive_scores(p[:1],[0]),nonconformity_scores(p[:1],method='ordinal_greedy'))


@pytest.mark.parametrize('uniforms',[[-.1],[1.1],[float('nan')],[True],['0.2'],[[.2]],[]])
def test_invalid_uniforms_are_not_silently_generated_or_replaced(uniforms):
    with pytest.raises(ValueError):adaptive_scores([[.4,.6]],uniforms)


@pytest.mark.parametrize('kwargs',[{'method':'unknown'},{'method':[]},{'k_reg':True},{'k_reg':0},
    {'method':'raps','k_reg':3},{'method':'raps','lam':-.1},{'method':'raps','lam':True},
    {'method':'raps','lam':float('inf')},{'method':'aps','lam':.1}])
def test_invalid_score_parameters_fail_closed(kwargs):
    with pytest.raises(ValueError):adaptive_scores([[.4,.6]],[.2],**kwargs)


@pytest.mark.parametrize('change',[{'qhat':None},{'qhat':2.},{'rank':True},{'rank':1},
    {'metadata':{'x':float('nan')}},{'unit':'patient_max'},{'extra':1},{'schema':'legacy'},
    {'randomization':'implicit_random_generator'},{'tie_policy':'unstable'}])
def test_artifact_validation_rejects_changed_conventions_and_unsafe_values(change):
    artifact=fit_adaptive(np.tile([.4,.6],(10,1)),np.ones(10,int),np.full(10,.2),alpha=.2)
    with pytest.raises(ValueError):validate_adaptive({**artifact,**change})


def frames():
    def part(prefix,labels):
        return pd.DataFrame({'image_id':[prefix+str(i) for i in range(len(labels))],
            'split_group_id':[prefix+'g'+str(i) for i in range(len(labels))],
            'dr_grade':labels,'is_group_representative':True,**{f'logit_{k}':float(k) for k in range(5)}})
    calibration=part('c',[0,1,2,3,4,0]);calibration['calibration_role']=['temperature']*3+['conformal']*3
    return calibration,part('v',[0,1,2,3,4])


def test_source_identity_roles_and_component_representatives_cannot_leak():
    cal,val=frames();cp,reps=validate_saved_frames(cal,val)
    assert cp.tolist()==[False]*3+[True]*3 and reps.all()
    for mutate in ('identity','group','role','representative','logit','label'):
        c,v=deepcopy(cal),deepcopy(val)
        if mutate=='identity':v.loc[0,'image_id']=c.loc[0,'image_id']
        elif mutate=='group':v.loc[0,'split_group_id']=c.loc[0,'split_group_id']
        elif mutate=='role':c.loc[0,'calibration_role']='test'
        elif mutate=='representative':c.loc[0,'is_group_representative']=False
        elif mutate=='logit':v.loc[0,'logit_0']=float('inf')
        else:c.loc[0,'dr_grade']=9
        with pytest.raises(ValueError):validate_saved_frames(c,v)


def test_paired_bootstrap_preserves_whole_components_and_endpoint_denominators():
    groups=np.array(['a','a','b','c']);truth=np.array([0,3,4,0])
    weights=bootstrap_weights(groups,20,91)
    assert np.array_equal(weights[:,0],weights[:,1])
    first=[[0],[3],[],[0]];second=[[0,1],[],[4],[0]]
    changes=paired_changes(first,second,truth,weights)
    assert changes['marginal_coverage']['delta']==0 and changes['mean_set_size']['delta']==.25
    assert changes['grade_3_inclusion']['delta']==-1 and changes['grade_4_inclusion']['delta']==1
    zero=paired_changes(first,first,truth,weights)
    assert all(x['delta']==x['lower']==x['upper']==0 for x in zero.values())
    missing=paired_changes(first,second,np.zeros(4,int),weights)
    assert missing['grade_3_inclusion']=={'delta':None,'lower':None,'upper':None,'valid_draws':0}
