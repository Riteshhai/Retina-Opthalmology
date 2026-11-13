"""Generated saved-result fixtures verify calibration/stress presentation."""
from copy import deepcopy
import json
from pathlib import Path
from urllib.parse import urlparse

from fastapi.testclient import TestClient
import httpx

from api.main import create_app
from qsart.evaluation_quality import CONDITIONS, HEADS, RESPONSES


def generated_report():
    clean = {head:{'accuracy':.5,'macro_f1':.4,'nll':1.,'ece_10':.1,'brier':.5,
                   'support':[1]*k,'recall':[.5]*k} for head,k in HEADS.items()}
    interval = {'difference':0.,'lower_95':-.1,'upper_95':.1}
    response = {'levels':{name:.5 for name in RESPONSES},'changes':{name:deepcopy(interval) for name in RESPONSES}}
    runs = []
    for seed in (2026,2027,2028):
        for arm in ('frozen','adapted'):
            runs.append({'model_id':f'quality_seed{seed}_{arm}','training_seed':seed,'arm':arm,
                'temperatures':{head:2. for head in HEADS},
                'clean_metrics':{role:{state:deepcopy(clean) for state in ('raw','temperature')}
                                 for role in ('calibration','validation')},
                'calibration_changes':{'mean_head_nll':deepcopy(interval)},
                'stress':[{'condition':c,'images':4,'raw':deepcopy(response),'temperature':deepcopy(response)} for c in CONDITIONS]})
    return {'dataset':'DeepDRiD','primary_model_id':'quality_seed2026_adapted','runs':runs,
        'summary':{},'response_contrasts':[{'training_seed':2026,'probability_state':'raw'}],
        'bootstrap':{'bootstrap_seed':20261108},'interval_counts':{'clean_calibration_changes':54},
        'corrupted_quality_ground_truth_available':False,'classifier_fitting_performed':False,
        'temperature_fitting_performed':True,'production_model_changed':False,
        'limitations':['Explicit generated software fixture, not retinal evidence.']}


def test_api_preserves_roles_all_models_rng_fields_and_absent_corrupted_truth(tmp_path,monkeypatch):
    payload = generated_report()
    path = tmp_path/'quality_calibration_stress_v1/evaluation.json'
    path.parent.mkdir();path.write_text(json.dumps(payload))
    monkeypatch.setenv('QSART_REPORTS_DIR',str(tmp_path))
    with TestClient(create_app()) as client:
        result = client.get('/research/metrics?experiment_id=quality_calibration_stress_v1').json()
        assert result['scope']=='development_quality_temperature_and_input_stress'
        assert result['metrics']['runs']==payload['runs']
        assert result['metrics']['bootstrap']=={'bootstrap_seed':20261108}
        assert result['metrics']['corrupted_quality_ground_truth_available'] is False
        assert result['metrics']['classifier_fitting_performed'] is False
        assert result['metrics']['temperature_fitting_performed'] is True
        assert result['metrics']['response_contrasts']==payload['response_contrasts']
        record = client.get('/research/experiments').json()['experiments'][0]
        assert record['partition']=='calibration_and_validation'


def test_explorer_separates_fit_clean_metrics_from_unlabeled_stress(monkeypatch):
    from streamlit.testing.v1 import AppTest
    payload = generated_report()
    record = {'experiment_id':'quality_calibration_stress_v1','dataset':'DeepDRiD',
        'partition':'calibration_and_validation','scope':'development_quality_temperature_and_input_stress',
        'metrics':payload,'notes':payload['limitations']}
    def get(url,**kwargs):
        path = urlparse(url).path
        data = {'ready':True} if path=='/ready' else {'experiments':[record]} if path=='/research/experiments' else record if path=='/research/metrics' else {}
        return httpx.Response(200,json=data)
    monkeypatch.setattr(httpx,'get',get)
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1]/'dashboard/app.py')).run(timeout=30)
    app.sidebar.radio[0].set_value('Experiment Explorer').run(timeout=30)
    assert not app.exception and len(app.dataframe[1].value)==6
    model_box = next(x for x in app.selectbox if x.label=='Recorded calibrated quality model')
    assert model_box.value=='quality_seed2026_adapted'
    for run in payload['runs']:
        next(x for x in app.selectbox if x.label=='Recorded calibrated quality model').set_value(run['model_id']).run(timeout=30)
        assert not app.exception
    for head,k in HEADS.items():
        next(x for x in app.selectbox if x.label=='Clean quality head').set_value(head).run(timeout=30)
        assert len(app.dataframe[3].value)==k
    for role in ('calibration','validation'):
        next(x for x in app.selectbox if x.label=='Recorded clean role').set_value(role).run(timeout=30)
        assert list(app.dataframe[2].value['Probabilities'])==['raw','temperature']
    for state in ('raw','temperature'):
        next(x for x in app.selectbox if x.label=='Recorded probability state').set_value(state)
        for response in RESPONSES:
            next(x for x in app.selectbox if x.label=='Recorded stress response').set_value(response).run(timeout=30)
            assert not app.exception
            stress = app.dataframe[4].value
            assert len(stress)==7 and stress.Condition.tolist()==list(CONDITIONS)
            assert not {'accuracy','nll','macro_f1'}.intersection(stress.columns)
    text = ' '.join(str(x.value) for x in app.caption)
    assert 'no new quality labels' in text and 'fitting data' in text and 'preserve class' in text
    assert 'No refitting, reselection or multiplicity correction' in text
