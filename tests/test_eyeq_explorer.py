"""Generated annotation-readiness fixtures do not claim neural quality results."""
import json
from pathlib import Path
from urllib.parse import urlparse

from fastapi.testclient import TestClient
import httpx

from api.main import create_app


def fixture_report():
    return {'dataset':'EyeQ_EyePACS','annotation_profile':{'images_annotated':6,'source_subject_keys':4,
        'original_splits':{name:{'images':3,'source_subject_keys':2,'quality_counts':{'0':1,'1':1,'2':1}} for name in ['train','test']}},
        'images_decoded':0,'images_downloaded':0,'model_inference_performed':False,
        'classifier_fitting_performed':False,'temperature_fitting_performed':False,
        'count_reference_differences':[{'original_split':'test','difference':1,'labels_changed':False}],
        'author_source_revision':'generated_software_fixture','access_review':{'status':'generated_fixture'},
        'readiness':{'metadata_verified':True,'ready_for_model_evaluation':False},
        'storage_snapshot':{'whole_archive_exceeds_current_free_space':True},
        'limitations':['Generated fixture, not a real metadata/model result.']}


def test_api_preserves_metadata_only_scope_and_missing_pixel_readiness(tmp_path,monkeypatch):
    payload=fixture_report();path=tmp_path/'eyeq_data_v1/data_audit.json';path.parent.mkdir();path.write_text(json.dumps(payload))
    monkeypatch.setenv('QSART_REPORTS_DIR',str(tmp_path))
    with TestClient(create_app()) as client:
        result=client.get('/research/metrics?experiment_id=eyeq_data_v1').json()
        assert result['scope']=='external_quality_annotation_and_access_readiness_only'
        assert result['metrics']['annotation_profile']==payload['annotation_profile']
        assert result['metrics']['model_inference_performed'] is False
        assert result['metrics']['readiness']['ready_for_model_evaluation'] is False
        assert not {'grading','quality','accuracy','nll'}.intersection(result['metrics'])
        record=client.get('/research/experiments').json()['experiments'][0]
        assert record['partition']=='external_candidate_unscored' and record['model_version'] is None


def test_explorer_displays_source_categories_and_readiness_without_accuracy(monkeypatch):
    from streamlit.testing.v1 import AppTest
    payload=fixture_report();record={'experiment_id':'eyeq_data_v1','dataset':'EyeQ_EyePACS',
        'scope':'external_quality_annotation_and_access_readiness_only','partition':'external_candidate_unscored',
        'metrics':payload,'notes':payload['limitations']}
    def get(url,**kwargs):
        path=urlparse(url).path
        value={'ready':True} if path=='/ready' else {'experiments':[record]} if path=='/research/experiments' else record if path=='/research/metrics' else {}
        return httpx.Response(200,json=value)
    monkeypatch.setattr(httpx,'get',get)
    app=AppTest.from_file(str(Path(__file__).resolve().parents[1]/'dashboard/app.py')).run(timeout=30)
    app.sidebar.radio[0].set_value('Experiment Explorer').run(timeout=30)
    assert not app.exception
    table=app.dataframe[1].value
    assert {'Good','Usable','Reject'}.issubset(table.columns) and len(table)==2
    assert not {'accuracy','nll','macro_f1'}.intersection(table.columns)
    assert next(x for x in app.metric if x.label=='Images decoded').value=='0'
    assert any('No EyeQ photographs' in x.value for x in app.info)
    assert any('remain incomplete' in x.value for x in app.warning)
    captions=' '.join(str(x.value) for x in app.caption)
    assert 'not DeepDRiD poor class 0' in captions and 'without relabeling' in captions
    assert 'does not prove image or patient independence' in captions
