"""Exercise the aggregate catalog contract and its Streamlit view."""

import hashlib
import json
from pathlib import Path
import shutil
from urllib.parse import urlsplit

from fastapi.testclient import TestClient
import httpx
import pandas as pd
from streamlit.testing.v1 import AppTest

from api.datasets import (SET_C_AUDIT, SET_C_EVALUATION, SET_C_METRICS, SET_C_READINESS, SET_C_READINESS_AUDIT,
                          SET_C_VALIDATION_ID)
from api.main import create_app


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / 'reports'


def test_catalog_reports_roles_without_accessing_images(monkeypatch, tmp_path):
    monkeypatch.setenv('QSART_REPORTS_DIR', str(REPORTS))
    monkeypatch.setenv('QSART_DR_CHECKPOINT', str(tmp_path / 'missing.pt'))
    with TestClient(create_app()) as client:
        assert client.get('/ready').status_code == 503
        response = client.get('/research/datasets')
        assert response.status_code == 200

    entries = {entry['dataset_id']: entry for entry in response.json()['datasets']}
    aptos, deepdrid, idrid, eyeq = (entries[key] for key in ('aptos_2019', 'deepdrid', 'idrid', 'eyeq'))
    assert [entry['status'] for entry in entries.values()] == ['audited', 'audited', 'audited', 'metadata_only']
    for entry in entries.values():
        for evidence in entry['evidence']:
            payload = (REPORTS / evidence['report']).read_bytes()
            assert hashlib.sha256(payload).hexdigest() == evidence['sha256']
        for partition in entry['partitions'] + entry['study_populations']:
            for field in ('dr_grade_counts', 'quality_counts'):
                counts = partition[field]
                assert not counts or sum(counts.values()) == partition['observations']

    assert [(p['role'], p['observations']) for p in aptos['partitions']] == [
        ('fitting', 2115), ('selection', 521), ('calibration', 527), ('reserved', 351),
        ('excluded', 148), ('not_imported', None),
    ]
    assert [p['observations'] for p in aptos['study_populations']] == [262, 265]
    assert all(p['role'] != 'fitting' for p in aptos['study_populations'])
    set_c_evaluated = (REPORTS / SET_C_AUDIT).is_file()
    assert [(p['role'], p['observations'], p['groups']) for p in deepdrid['partitions'][-2:]] == [
        ('consumed', 340, 85), ('source_roster' if set_c_evaluated else 'reserved', 400, 100),
    ]
    assert [(p['role'], p['observations'], p['groups']) for p in deepdrid['study_populations']] == [
        ('consumed' if set_c_evaluated else 'reserved', 396, 99), ('excluded', 4, 1),
    ]
    assert [(p['role'], p['observations']) for p in idrid['partitions']] == [('consumed', 413), ('consumed', 103)]
    assert [p['observations'] for p in idrid['study_populations']] == [516, 513, 505]
    assert [(p['role'], p['observations'], p['transformed_matches']) for p in eyeq['partitions']] == [
        ('metadata_only', 12543, 12529), ('metadata_only', 16249, 0),
    ]
    assert eyeq['partitions'][0]['quality_counts']['2'] == 2320
    forbidden = ('"image_id"', '"patient_id"', '"image_path"', '"private_output_sha256"', '"input_bindings"')
    assert all(field not in response.text for field in forbidden)


def test_catalog_isolates_missing_and_changed_evidence(monkeypatch, tmp_path):
    files = {
        'aptos_v1': ('data_audit.json', 'independent_audit.json'),
        'eyeq_data_v1': ('data_audit.json', 'independent_audit.json'),
        'eyeq_hf_v1': ('acquisition.json', 'independent_audit.json'),
    }
    for directory, names in files.items():
        for name in names:
            rel = Path(directory) / name
            (tmp_path / rel).parent.mkdir(exist_ok=True)
            shutil.copyfile(REPORTS / rel, tmp_path / rel)
    # Preserve a valid JSON document while breaking its independent digest binding.
    with (tmp_path / 'aptos_v1/data_audit.json').open('ab') as stream:
        stream.write(b'\n')
    monkeypatch.setenv('QSART_REPORTS_DIR', str(tmp_path))
    monkeypatch.delenv('QSART_CONFORMAL', raising=False)
    with TestClient(create_app(object())) as client:
        response = client.get('/research/datasets')
    assert response.status_code == 200
    datasets = {entry['dataset_id']: entry for entry in response.json()['datasets']}
    assert datasets['eyeq']['status'] == 'metadata_only'
    for name in ('aptos_2019', 'deepdrid', 'idrid'):
        assert datasets[name]['status'] == 'unavailable'
        assert not datasets[name]['partitions'] and not datasets[name]['evidence']
    assert str(tmp_path) not in response.text


def test_set_c_uses_only_bound_aggregate_evidence(monkeypatch, tmp_path):
    """Use a software report fixture to test the two endpoints before and after audit publication."""
    paths = ('development_v1/data_summary.json', 'confirmation_v1/evaluation.json', SET_C_READINESS,
             SET_C_READINESS_AUDIT)
    for name in paths:
        target = tmp_path / name
        target.parent.mkdir(exist_ok=True)
        shutil.copyfile(REPORTS / name, target)
    readiness = json.loads((tmp_path / SET_C_READINESS).read_bytes())
    digests = {'readiness_report': hashlib.sha256((tmp_path / SET_C_READINESS).read_bytes()).hexdigest(),
               'readiness_audit': hashlib.sha256((tmp_path / SET_C_READINESS_AUDIT).read_bytes()).hexdigest(),
               'readiness_witness': readiness['witness_sha256']}
    report = {
        'experiment_id': SET_C_VALIDATION_ID, 'partition': 'same_release_held_out', 'model_inference_performed': True,
        'images': 396, 'patients': 99, 'source_commit': 'a' * 40, 'private_witness_sha256': 'b' * 64,
        'prediction_metrics': {'software_contract_fixture': True}, 'policy': {}, 'budget_pairs': [],
        'pointwise': {}, 'scenarios': [], 'comparisons': [],
        'input_provenance': {'artifact_sha256': digests, 'source_release': readiness['source_release'],
                             'model_version': 'software_contract_fixture'},
        'private_payload': 'fixture_field_that_must_not_be_published',
    }
    evaluation_path, receipt_path = tmp_path / SET_C_EVALUATION, tmp_path / SET_C_AUDIT
    evaluation_path.parent.mkdir()

    def publish_receipt():
        content = json.dumps(report).encode()
        evaluation_path.write_bytes(content)
        receipt_path.write_bytes(json.dumps({
            'audit_status': 'passed', 'evaluation_sha256': hashlib.sha256(content).hexdigest(),
            'input_artifact_sha256': digests, **{key: report[key] for key in (
                'experiment_id', 'source_commit', 'private_witness_sha256', 'images', 'patients')},
        }).encode())

    monkeypatch.setenv('QSART_REPORTS_DIR', str(tmp_path))
    monkeypatch.delenv('QSART_CONFORMAL', raising=False)
    with TestClient(create_app(object())) as client:
        def catalog():
            return next(row for row in client.get('/research/datasets').json()['datasets']
                        if row['dataset_id'] == 'deepdrid')

        def check_unpublished():
            assert client.get('/research/metrics', params={'experiment_id': SET_C_VALIDATION_ID}).status_code == 404
            listing = client.get('/research/experiments')
            assert listing.status_code == 200
            assert all(row['experiment_id'] != SET_C_VALIDATION_ID for row in listing.json()['experiments'])
            entry = catalog()
            assert entry['status'] == 'audited'
            assert entry['partitions'][-1]['role'] == entry['study_populations'][0]['role'] == 'reserved'
            assert {row['report'] for row in entry['evidence']} == set(paths)

        evaluation_path.write_bytes(json.dumps(report).encode())
        check_unpublished()
        publish_receipt()
        listing = client.get('/research/experiments')
        result = next(row for row in listing.json()['experiments'] if row['experiment_id'] == SET_C_VALIDATION_ID)
        metrics = client.get('/research/metrics', params={'experiment_id': SET_C_VALIDATION_ID})
        assert result['partition'] == 'same_release_held_out' and result['model_version'] == 'software_contract_fixture'
        assert metrics.status_code == 200 and result['metrics'] == metrics.json()['metrics']
        assert result['metrics'] == {
            **{key: report[key] for key in SET_C_METRICS},
            'evaluation_sha256': hashlib.sha256(evaluation_path.read_bytes()).hexdigest(),
            'independent_audit_sha256': hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
        }
        assert 'fixture_field_that_must_not_be_published' not in listing.text + metrics.text
        entry = catalog()
        assert entry['partitions'][-1]['name'] == 'Set-C source roster'
        assert entry['partitions'][-1]['role'] == 'source_roster'
        assert entry['study_populations'][0]['role'] == 'consumed'
        assert {row['report'] for row in entry['evidence']} == {*paths, SET_C_EVALUATION, SET_C_AUDIT}

        evaluation_path.write_bytes(evaluation_path.read_bytes() + b'\n')
        check_unpublished()
        digests['readiness_report'] = '0' * 64
        publish_receipt()
        check_unpublished()


def test_dataset_catalog_displays_unknowns_and_real_zeroes(monkeypatch):
    monkeypatch.setenv('QSART_REPORTS_DIR', str(REPORTS))
    monkeypatch.delenv('QSART_CONFORMAL', raising=False)
    with TestClient(create_app(object())) as client:
        def local_get(url, **kwargs):
            return client.get(urlsplit(url).path)

        monkeypatch.setattr(httpx, 'get', local_get)
        app = AppTest.from_file(str(ROOT / 'dashboard/app.py')).run(timeout=30)
        app.sidebar.radio[0].set_value('Dataset Catalog').run(timeout=30)
        assert not app.exception
        aptos = app.dataframe[0].value.set_index('Population')
        assert aptos.loc['QSART internal test', 'QSART use'] == 'Reserved for evaluation'
        assert pd.isna(aptos.loc['Kaggle competition test', 'Photos'])

        select = next(box for box in app.selectbox if box.label == 'Dataset')
        select.set_value('DeepDRiD').run(timeout=30)
        assert not app.exception
        assert app.dataframe[1].value['Patients'].tolist() == [99, 1]
        assert app.dataframe[3].value['Insufficient'].tolist() == [438, 98, 82, 188, 220]
        expected_use = 'Source roster' if (REPORTS / SET_C_AUDIT).is_file() else 'Reserved for evaluation'
        assert app.dataframe[0].value.iloc[-1]['QSART use'] == expected_use

        next(box for box in app.selectbox if box.label == 'Dataset').set_value('IDRiD').run(timeout=30)
        assert not app.exception
        assert app.dataframe[0].value['QSART use'].tolist() == ['Evaluated', 'Evaluated']

        next(box for box in app.selectbox if box.label == 'Dataset').set_value('EyeQ').run(timeout=30)
        assert not app.exception
        eyeq = app.dataframe[0].value
        assert eyeq['Annotations'].tolist() == [12543, 16249]
        assert eyeq['Transformed mirror matches'].tolist() == [12529, 0]
        assert app.get('download_button')
