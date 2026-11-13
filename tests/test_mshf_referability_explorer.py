"""Exercise the experiment views with the committed aggregate reports."""

import json
from pathlib import Path
from urllib.parse import urlsplit

from fastapi.testclient import TestClient
import httpx
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from api.main import create_app


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def research_client(monkeypatch):
    monkeypatch.setenv('QSART_REPORTS_DIR', str(ROOT / 'reports'))
    with TestClient(create_app()) as client:
        yield client


def test_registered_diagnostics_preserve_aggregate_reports(research_client):
    registry = research_client.get('/research/experiments')
    assert registry.status_code == 200
    records = {record['experiment_id']: record for record in registry.json()['experiments']}
    for name, partition, fields in (
        ('mshf_quality_v1', 'external', ('inventory', 'overlap', 'populations', 'model')),
        ('referability_gate_v1', 'validation', ('fits', 'budget_pairs', 'policy', 'calibration')),
    ):
        report = json.loads((ROOT / 'reports' / name / 'evaluation.json').read_text(encoding='utf-8'))
        assert records[name]['partition'] == partition
        assert records[name]['scope'] == report['scope']
        response = research_client.get('/research/metrics', params={'experiment_id': name})
        assert response.status_code == 200
        detail = response.json()
        assert detail['notes'] == report['limitations']
        assert {key: detail['metrics'][key] for key in fields} == {key: report[key] for key in fields}
        assert not {'witness_sha256', 'private_output_sha256', 'rows'}.intersection(detail['metrics'])
    assert records['mshf_quality_v1']['model_version'] == records['mshf_quality_v1']['metrics']['model']['model_version']


def open_experiment(client, monkeypatch, name):
    def local_get(url, **kwargs):
        target = urlsplit(url)
        return client.get(target.path + ('?' + target.query if target.query else ''))

    monkeypatch.setattr(httpx, 'get', local_get)
    app = AppTest.from_file(str(ROOT / 'dashboard/app.py')).run(timeout=30)
    app.sidebar.radio[0].set_value('Experiment Explorer').run(timeout=30)
    select(app, 'Experiment').set_value(name).run(timeout=30)
    assert not app.exception
    return app


def select(app, label):
    return next(widget for widget in app.selectbox if widget.label == label)


def table(app, column):
    return next(frame.value for frame in app.dataframe if column in frame.value.columns)


def test_mshf_populations_show_source_errors_and_missing_class_metrics(research_client, monkeypatch):
    app = open_experiment(research_client, monkeypatch, 'mshf_quality_v1')
    assert not any(column.startswith('metrics.populations.') for column in app.dataframe[0].value.columns)
    groups = table(app, 'Good flagged').set_index('Source')
    assert groups.loc['DR-XJU', ['Good photos', 'Good flagged']].tolist() == [118, 87]
    assert groups.loc['DR-ZJU', ['Poor photos', 'Poor flagged']].tolist() == [41, 21]
    assert pd.isna(groups.loc['Healthy', 'Poor AUROC'])
    calibration = table(app, 'Frozen log loss').set_index('Source').loc['Pooled']
    assert calibration['Frozen log loss'] > calibration['Raw log loss']
    assert calibration['Frozen Brier'] < calibration['Raw Brier']

    select(app, 'MSHF population').set_value('unanimous').run(timeout=30)
    assert not app.exception
    unanimous = table(app, 'Good flagged').set_index('Source')
    assert unanimous.loc['Pooled', ['Poor photos', 'Good photos']].tolist() == [188, 257]
    assert pd.isna(unanimous.loc['Glaucoma', 'Good specificity'])


def test_referability_controls_distinguish_overload_from_zero_misses(research_client, monkeypatch):
    app = open_experiment(research_client, monkeypatch, 'referability_gate_v1')
    allocation = table(app, 'Referable auto misses').set_index('Method')
    assert allocation.loc['Projected ordinal', 'Status'] == 'Overloaded'
    assert pd.isna(allocation.loc['Projected ordinal', 'Referable auto misses'])
    assert allocation.loc['Binary LAC', ['Automatic', 'Referable auto misses']].tolist() == [3, 0]

    select(app, 'Conformal alpha').set_value(.05).run(timeout=30)
    assert not app.exception
    assert table(app, 'Referable auto misses').set_index('Method').loc['Projected LAC', 'Status'] == 'Overloaded'
    select(app, 'Cohort capacity').set_value((.3, .2)).run(timeout=30)
    assert not app.exception
    allocation = table(app, 'Referable auto misses')
    assert allocation['Status'].eq('Overloaded').all()
    assert allocation[['Automatic', 'Recapture', 'Review', 'Referable auto misses']].isna().all().all()
