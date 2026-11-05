"""Explicit generated saved-result fixtures verify API/UI contracts."""
from copy import deepcopy
import json
from pathlib import Path
from urllib.parse import urlparse

from fastapi.testclient import TestClient
import httpx

from api.main import create_app
from qsart.training.quality_adaptation import ARMS, HEADS, SEEDS


def fixture_report():
    metrics = {head: {'accuracy': .5, 'macro_f1': .4, 'nll': 1.,
                'support': list(range(1, size+1)), 'recall': [.5]*size} for head, size in HEADS.items()}
    return {'dataset': 'DeepDRiD', 'primary_seed': 2026, 'fixed_epochs_per_run': 15,
        'population': {'class_supports': {head: {'train': list(range(2, size+2)),
                       'validation': list(range(1, size+1))} for head, size in HEADS.items()}},
        'runs': [{'seed': seed, 'arm': arm, 'selected_epoch': 1, 'mean_head_macro_f1': .4,
                  'selected_metrics': deepcopy(metrics), 'warm_metrics': deepcopy(metrics)} for seed in SEEDS for arm in ARMS],
        'majority_references': {head: {'class': 0, 'accuracy': .3, 'macro_f1': .1} for head in HEADS},
        'production_model_changed': False, 'summary': {'frozen': {}, 'adapted': {}},
        'paired_changes': [{'seed': seed, 'components': 3} for seed in SEEDS],
        'limitations': ['Generated software fixture; no retinal performance claim.']}


def test_quality_api_preserves_all_arms_seeds_supports_and_scope(tmp_path, monkeypatch):
    report = tmp_path/'quality_adaptation_v1/evaluation.json'
    report.parent.mkdir()
    payload = fixture_report()
    report.write_text(json.dumps(payload))
    monkeypatch.setenv('QSART_REPORTS_DIR', str(tmp_path))
    with TestClient(create_app()) as client:
        rows = client.get('/research/experiments').json()['experiments']
        assert len(rows) == 1 and rows[0]['partition'] == 'validation'
        data = client.get('/research/metrics?experiment_id=quality_adaptation_v1').json()
        assert data['scope'] == 'development_matched_quality_backbone_adaptation'
        assert data['metrics']['runs'] == payload['runs']
        assert data['metrics']['majority_references'] == payload['majority_references']
        assert data['metrics']['population'] == payload['population']
        assert data['metrics']['production_model_changed'] is False
        assert data['metrics']['primary_seed'] == 2026


def test_quality_explorer_shows_every_saved_seed_and_head_without_fitting(monkeypatch):
    from streamlit.testing.v1 import AppTest
    payload = fixture_report()
    record = {'experiment_id': 'quality_adaptation_v1', 'dataset': 'DeepDRiD', 'partition': 'validation',
              'scope': 'development_matched_quality_backbone_adaptation', 'metrics': payload, 'notes': payload['limitations']}

    def get(url, **kwargs):
        path = urlparse(url).path
        if path == '/ready':
            data = {'ready': True}
        elif path == '/research/experiments':
            data = {'experiments': [record]}
        elif path == '/research/metrics':
            data = record
        else:
            data = {}
        return httpx.Response(200, json=data)

    monkeypatch.setattr(httpx, 'get', get)
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1]/'dashboard/app.py')).run(timeout=30)
    app.sidebar.radio[0].set_value('Experiment Explorer').run(timeout=30)
    assert not app.exception
    assert len(app.dataframe[1].value) == 6
    assert next(x for x in app.selectbox if x.label == 'Recorded quality seed').value == 2026
    for seed in SEEDS:
        for head, size in HEADS.items():
            next(x for x in app.selectbox if x.label == 'Recorded quality seed').set_value(seed)
            next(x for x in app.selectbox if x.label == 'Recorded quality head').set_value(head).run(timeout=30)
            assert not app.exception
            assert len(app.dataframe[2].value) == 4
            assert len(app.dataframe[3].value) == size
    captions = ' '.join(str(item.value) for item in app.caption)
    assert 'one selection image' in captions and 'Production weights are unchanged' in captions
    assert 'different selector' in captions and 'without reselection' in captions
