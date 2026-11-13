import pandas as pd
import pytest
from qsart.policy import validate, decide, tune, metrics
from qsart.synthetic import make_mock
from qsart.cli import evaluate


def test_risk_high_routes_to_review():
    d=make_mock(6)
    d.loc[0,['quality','p_rdr','p_severe','uncertainty']]=[.2,.8,.85,.2]
    assert decide(d)[0]=='review'


def test_ungradable_not_cleared():
    d=make_mock(20)
    d['quality']=.1
    assert not (decide(d)=='routine').any()


def test_ungradable_low_risk_is_recaptured():
    d=make_mock(2)
    d.loc[0,['quality','p_rdr','p_severe','uncertainty']]=[.1,.01,.01,.1]
    assert decide(d)[0]=='recapture'


def test_empty_invalid():
    with pytest.raises(ValueError): validate(pd.DataFrame())


def test_metric_actions_sum():
    d=make_mock(30)
    m=metrics(d,decide(d))
    assert abs(m['routine_rate']+m['review_rate']+m['recapture_rate']-1)<1e-10


def test_tuning_repeatable_and_leakage_blocked():
    a=make_mock(90,prefix='a')
    t1,_=tune(a,max_review=1,max_recapture=1,min_routine=0)
    t2,_=tune(a,max_review=1,max_recapture=1,min_routine=0)
    assert t1==t2
    with pytest.raises(ValueError,match='Patient leakage'):
        evaluate(a,a, {'max_review':1,'max_recapture':1,'min_routine':0})


def test_infeasible_budget_reports_error():
    d=make_mock(40)
    with pytest.raises(ValueError,match='No feasible'):
        tune(d,max_review=0,max_recapture=0,min_routine=1)
