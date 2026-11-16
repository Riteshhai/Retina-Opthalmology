"""Exercise the aggregate catalog contract and its Streamlit view."""

import hashlib
from pathlib import Path
import shutil
from urllib.parse import urlsplit

from fastapi.testclient import TestClient
import httpx
import pandas as pd
from streamlit.testing.v1 import AppTest

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
    assert [(p['role'], p['observations'], p['groups']) for p in deepdrid['partitions'][-2:]] == [
        ('consumed', 340, 85), ('reserved', 400, 100),
    ]
    assert [(p['role'], p['observations'], p['groups']) for p in deepdrid['study_populations']] == [
        ('reserved', 396, 99), ('excluded', 4, 1),
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

        next(box for box in app.selectbox if box.label == 'Dataset').set_value('IDRiD').run(timeout=30)
        assert not app.exception
        assert app.dataframe[0].value['QSART use'].tolist() == ['Evaluated', 'Evaluated']

        next(box for box in app.selectbox if box.label == 'Dataset').set_value('EyeQ').run(timeout=30)
        assert not app.exception
        eyeq = app.dataframe[0].value
        assert eyeq['Annotations'].tolist() == [12543, 16249]
        assert eyeq['Transformed mirror matches'].tolist() == [12529, 0]
        assert app.get('download_button')
