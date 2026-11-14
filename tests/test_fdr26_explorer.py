"""Verify FDR26 registry and dashboard data against the audited aggregates."""

import json
from pathlib import Path
from urllib.parse import urlsplit

from fastapi.testclient import TestClient
import httpx
import pytest
from streamlit.testing.v1 import AppTest

from api.main import create_app


ROOT = Path(__file__).resolve().parents[1]
READINESS = 'fdr26_readiness_v1'
GRADING = 'fdr26_grading_v1'
PAIRED = 'fdr26_aptos_v1'
REPORT_FIELDS = {
    READINESS: {'inventory', 'overlap', 'grade_standard', 'patient_linkage_available',
                'model_inference_performed', 'source_commit'},
    GRADING: {'model', 'reference_grade_standard', 'binary_reference_minimum_grade',
              'binary_probability_threshold', 'patient_linkage_available', 'populations', 'source_commit'},
    PAIRED: {'model', 'source_replay', 'populations', 'source_commit'},
}


@pytest.fixture
def reports():
    return {name: json.loads((ROOT / 'reports' / name / 'evaluation.json').read_bytes())
            for name in REPORT_FIELDS}


@pytest.fixture
def research_client(monkeypatch):
    monkeypatch.setenv('QSART_REPORTS_DIR', str(ROOT / 'reports'))
    with TestClient(create_app()) as client:
        yield client


def test_fdr26_registry_exports_only_report_aggregates(research_client, reports):
    response = research_client.get('/research/experiments')
    assert response.status_code == 200
    listed = response.json()['experiments']
    entries = {value['experiment_id']: value for value in listed}
    assert len(listed) == len(entries)
    for name, report in reports.items():
        record = entries[name]
        response = research_client.get('/research/metrics', params={'experiment_id': name})
        assert response.status_code == 200
        detail = response.json()
        assert record['dataset'] == report['dataset'] == 'FDR26'
        assert record['partition'] == report['partition']
        assert detail['scope'] == record['scope'] == report['scope']
        assert record['model_version'] == report.get('model', {}).get('version')
        assert detail['metrics'] == record['metrics'] == {key: report[key] for key in REPORT_FIELDS[name]}
        assert detail['notes'] == record['notes']
        assert 'ALFEDIAM' in ' '.join(record['notes'])
        assert 'AUTO_SCREEN' not in ' '.join(record['notes'])


def select(app, label, value):
    next(widget for widget in app.selectbox if widget.label == label).set_value(value).run(timeout=30)
    assert not app.exception


@pytest.fixture
def explorer(research_client, monkeypatch):
    def local_get(url, **kwargs):
        target = urlsplit(url)
        return research_client.get(target.path + ('?' + target.query if target.query else ''))

    monkeypatch.setattr(httpx, 'get', local_get)
    app = AppTest.from_file(str(ROOT / 'dashboard/app.py')).run(timeout=30)
    app.sidebar.radio[0].set_value('Experiment Explorer').run(timeout=30)
    select(app, 'Dataset', 'FDR26')
    return app


def table(app, column):
    return next(frame.value for frame in app.dataframe if column in frame.value.columns)


def test_readiness_view_keeps_exact_joins_and_all_populations(explorer, reports):
    select(explorer, 'Experiment', READINESS)
    readiness = reports[READINESS]
    population_keys = ['eligible_images', 'representatives', 'prefix_sensitivity_eligible_images',
                       'prefix_sensitivity_representatives']
    grades = table(explorer, 'Exact image joins')
    assert grades['Workbook rows'].tolist() == [readiness['inventory']['grade_counts'][str(i)] for i in range(5)]
    assert grades['Exact image joins'].tolist() == [readiness['inventory']['joined_grade_counts'][str(i)] for i in range(5)]
    assert table(explorer, 'Population')['Photos'].tolist() == [readiness['overlap'][key] for key in population_keys]
    assert explorer.get('download_button')


def test_grading_views_preserve_every_population_and_probability_path(explorer, reports):
    for name in (GRADING, PAIRED):
        select(explorer, 'Experiment', name)
        for key, population in reports[name]['populations'].items():
            select(explorer, 'FDR26 population', key)
            grades = table(explorer, 'ALFEDIAM grade')
            assert grades['Photos'].tolist() == [population['grade_counts'][str(i)] for i in range(5)]
            assert sum(grades['Photos']) == population['images']
            shown = table(explorer, 'Probability path').set_index(['Model', 'Probability path'])
            models = {'DeepDRiD': population} if name == GRADING else {
                'APTOS': population['aptos'], 'DeepDRiD': population['deepdrid'],
            }
            assert len(shown) == 2 * len(models)
            for label, paths in models.items():
                for path, source in [('Raw', 'raw'), ('Temperature-scaled', 'calibrated')]:
                    expected = paths[source]
                    row = shown.loc[(label, path)]
                    assert row['Grade accuracy'] == expected['accuracy']
                    assert row['Two-grade undercalls'] == expected['undergrade_by_two_or_more']
                    assert row['Grade ≥2 AUROC'] == expected['binary']['auroc']
                    assert row['Grade ≥2 sensitivity'] == expected['binary']['sensitivity']
                    assert row['Grade ≥2 specificity'] == expected['binary']['specificity']
                    assert row['NLL'] == expected['nll']
            if name == PAIRED:
                paired = population['paired']
                shown_counts = table(explorer, 'Both correct').iloc[0]
                assert shown_counts['Both correct'] == paired['both_correct']
                assert shown_counts['APTOS only correct'] == paired['aptos_only']
                assert shown_counts['DeepDRiD only correct'] == paired['deepdrid_only']
                assert shown_counts['Neither correct'] == paired['neither']
        assert explorer.get('download_button')
