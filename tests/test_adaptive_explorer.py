"""Explicit generated report fixtures test research API/control presentation."""
from copy import deepcopy
import json
from pathlib import Path
from urllib.parse import urlparse

from fastapi.testclient import TestClient
import httpx

from api.main import create_app


def fixture_report():
    metric={'n_images':5,'marginal_coverage':.8,'mean_set_size':1.6,'empty_set_count':1,
        'full_five_grade_set_count':0,'contiguous_nonempty_fraction':.6,
        'per_class':{str(k):{'n':1,'coverage':float(k!=3)} for k in range(5)}}
    comparisons=[]
    for alpha in (.1,.05):
        for method in ('lac','ordinal_greedy','ordinal_greedy_randomized','aps','raps'):
            for seed in ([None] if method in {'lac','ordinal_greedy'} else list(range(2026,2031))):
                view={name:deepcopy(metric) for name in ('all_selection_images','selection_group_representatives')}
                comparisons.append({'alpha':alpha,'method':method,'randomization_seed':seed,
                    'artifact':{'qhat':1.2 if method=='raps' else .8},'raw':deepcopy(view),'argmax_fallback':deepcopy(view)})
    return {'dataset':'APTOS_2019','comparisons':comparisons,'paired_comparisons':[],
        'configuration':{'raps_k_reg':2,'raps_lambda':.1},'summary':[],'baseline_reproduction':[],
        'bootstrap':{'interval_count':160},'model_training_seed':2026,'fixed_temperature':2.,
        'images_decoded':0,'model_inference_performed':False,'classifier_fitting_performed':False,
        'temperature_fitting_performed':False,'raps_parameter_tuning_performed':False,'production_model_changed':False,
        'limitations':['Generated software fixture; no measured retinal/model result.']}


def test_api_keeps_fixed_parameters_all_seeds_and_no_new_fitting(tmp_path,monkeypatch):
    payload=fixture_report();path=tmp_path/'adaptive_sets_v1/evaluation.json';path.parent.mkdir()
    path.write_text(json.dumps(payload));monkeypatch.setenv('QSART_REPORTS_DIR',str(tmp_path))
    with TestClient(create_app()) as client:
        detail=client.get('/research/metrics?experiment_id=adaptive_sets_v1').json()
        assert detail['scope']=='reused_development_fixed_aps_raps_score_comparison'
        for name in ('comparisons','configuration','model_training_seed','fixed_temperature','raps_parameter_tuning_performed'):
            assert detail['metrics'][name]==payload[name]
        assert detail['metrics']['model_inference_performed'] is False
        assert len(detail['metrics']['comparisons'])==34


def test_explorer_displays_uniform_controls_raw_fallback_and_scope(monkeypatch):
    from streamlit.testing.v1 import AppTest
    payload=fixture_report();record={'experiment_id':'adaptive_sets_v1','dataset':'APTOS_2019',
        'partition':'validation','scope':'reused_development_fixed_aps_raps_score_comparison',
        'metrics':payload,'notes':payload['limitations']}
    def get(url,**kwargs):
        path=urlparse(url).path
        value={'ready':True} if path=='/ready' else {'experiments':[record]} if path=='/research/experiments' else record if path=='/research/metrics' else {}
        return httpx.Response(200,json=value)
    monkeypatch.setattr(httpx,'get',get)
    app=AppTest.from_file(str(Path(__file__).resolve().parents[1]/'dashboard/app.py')).run(timeout=30)
    app.sidebar.radio[0].set_value('Experiment Explorer').run(timeout=30)
    assert not app.exception and len(app.dataframe[1].value)==5
    assert next(x for x in app.selectbox if x.label=='Adaptive uniform seed').value==2026
    assert app.dataframe[1].value.loc[app.dataframe[1].value.Method.eq('raps'),'Threshold'].iloc[0]==1.2
    for label,value in [('Recorded adaptive-set alpha',.05),('Adaptive empty-set policy','argmax_fallback'),
        ('Adaptive evaluation population','selection_group_representatives'),('Adaptive uniform seed',2030)]:
        next(x for x in app.selectbox if x.label==label).set_value(value).run(timeout=30)
        assert not app.exception and len(app.dataframe[1].value)==5
    captions=' '.join(str(x.value) for x in app.caption)
    for text in ('no tuning','not model training variance','not verified patient identities','do not change these intervals'):
        assert text in captions,text
    assert any('no fresh validation claim' in x.value for x in app.info)
    assert any('No method or parameter is selected' in x.value for x in app.warning)
