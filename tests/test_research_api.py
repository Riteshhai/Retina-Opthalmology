"""Generated software scores exercise current API semantics, never research evidence."""
from io import BytesIO
import json

from fastapi.testclient import TestClient
from PIL import Image
import pytest

from api.main import create_app


def score(p=None, quality=.9, entropy=.2, **extra):
    return {'grade_probabilities': p or [.9, .04, .03, .02, .01],
            'quality_good_probability': quality, 'uncertainty': entropy, **extra}


def test_simulator_zero_budget_never_forces_referable_clearance():
    with TestClient(create_app()) as client:
        response = client.post('/triage/simulate', json={'scores': [score(p=[.01, .01, .03, .05, .9])],
            'config': {'review_budget': 0., 'recapture_budget': 1.}})
        assert response.status_code == 200
        data = response.json()
        assert data['status'] == 'overloaded' and not data['actions']
        assert data['diagnostics']['needed_review'] == 1
        assert 'missed_referable_auto' not in data['metrics']


def test_simulator_marks_true_referable_clearance_as_miss_and_counts_deferrals():
    scores = [score(true_grade=3), score(p=[.01, .01, .9, .04, .04], true_grade=2)]
    with TestClient(create_app()) as client:
        data = client.post('/triage/simulate', json={'scores': scores, 'method': 'plain',
            'config': {'review_budget': .5, 'recapture_budget': 0.}}).json()
        assert data['status'] == 'evaluated'
        assert data['actions'] == ['AUTO_NONREFERABLE', 'SPECIALIST_REVIEW']
        assert data['metrics']['missed_referable_auto'] == 1
        assert data['metrics']['missed_referable_per_1000_all'] == 500
        assert data['scope'] == 'development_exploration'


def test_simulator_rejects_heldout_and_invalid_configs_and_does_not_invent_cache(monkeypatch):
    monkeypatch.delenv('QSART_SIMULATION_DATA', raising=False)
    with TestClient(create_app()) as client:
        assert client.post('/triage/simulate', json={'dataset_id': 'confirmation_v1'}).status_code == 422
        assert client.post('/triage/simulate', json={'scores': [score()],
            'config': {'severe_grade': 1, 'referable_grade': 2}}).status_code == 422
        assert client.post('/triage/simulate', json={'scores': [score(p=[2, 0, 0, 0, 0])]}).status_code == 422
        result = client.post('/triage/simulate', json={}).json()
        assert result['status'] == 'unavailable' and not result['metrics'] and not result['actions']


def test_config_and_aggregate_experiments_are_typed_and_no_arbitrary_paths(tmp_path, monkeypatch):
    monkeypatch.setenv('QSART_REPORTS_DIR', str(tmp_path))
    with TestClient(create_app()) as client:
        config = client.get('/triage/config').json()
        assert config['policy_version'] == '2.0'
        assert 'AUTO_NONREFERABLE' in config['actions'] and 'AUTO_SCREEN' not in config['actions']
        assert 'entropy_only' in config['methods']
        assert client.get('/research/experiments').json() == {'experiments': []}
        assert client.get('/research/metrics', params={'experiment_id': '../../MASTER_SPEC.md'}).status_code == 404
        schema = client.get('/openapi.json').json()
        for path in ('/triage/config', '/triage/simulate', '/research/experiments', '/research/metrics'):
            assert path in schema['paths']


def test_saved_stress_experiment_preserves_scope_and_unfavorable_paired_results(tmp_path, monkeypatch):
    """Software fixture verifies discovery, not a measured retinal experiment."""
    report = tmp_path/'robustness_v1/evaluation.json'
    report.parent.mkdir()
    report.write_text(json.dumps({'images': 2, 'patients': 1, 'model_version': 'software_fixture',
        'conditions': {'blur_2': {'grading': {'accuracy': 0.}}},
        'paired_changes': {'blur_2': {'accuracy': {'mean_delta': -.5}}},
        'limitations': ['Synthetic-input development stress; no independent clinical validation.']}))
    monkeypatch.setenv('QSART_REPORTS_DIR', str(tmp_path))
    with TestClient(create_app()) as client:
        listed = client.get('/research/experiments').json()['experiments']
        assert len(listed) == 1 and listed[0]['experiment_id'] == 'controlled_degradation_v1'
        detail = client.get('/research/metrics?experiment_id=controlled_degradation_v1').json()
        assert detail['scope'] == 'development_model_input_stress'
        assert detail['metrics']['conditions']['blur_2']['grading']['accuracy'] == 0.
        assert detail['metrics']['paired_changes']['blur_2']['accuracy']['mean_delta'] == -.5
        assert 'no independent clinical validation' in detail['notes'][0]


class PositiveFixture:
    """Dependency-injected fixture bypasses artifact loading for an endpoint contract."""
    model_version = 'software-positive-fixture'

    def predict_images(self, images, second_pass_budget=0):
        return [{'model_version': self.model_version, 'predicted_grade': 4,
                 'grade_probabilities': [.01, .01, .03, .05, .9], 'referable_probability': .98,
                 'quality': {'probabilities': [.1, .9], 'predicted_class': 1, 'poor_probability': .1},
                 'uncertainty': {'entropy': .2, 'confidence': .9},
                 'second_pass_activated': False, 'processing_time_ms': 1.} for image in images]


def test_real_route_contract_positive_single_review_and_batch_overload():
    body = BytesIO(); Image.new('RGB', (64, 64), 'red').save(body, format='PNG')
    with TestClient(create_app(PositiveFixture())) as client:
        response = client.post('/predict', files={'file': ('fixture.png', body.getvalue(), 'image/png')})
        assert response.status_code == 200
        assert response.json()['action'] == 'SPECIALIST_REVIEW'
        assert response.json()['policy_version'] == '2.0'
        response = client.post('/predict/batch?review_budget=0&recapture_budget=1',
            files=[('files', ('fixture.png', body.getvalue(), 'image/png'))])
        assert response.status_code == 422
        assert response.json()['detail']['code'] == 'BUDGET_OVERLOADED'


def test_explicitly_configured_missing_conformal_artifact_makes_service_unready(tmp_path, monkeypatch):
    monkeypatch.setenv('QSART_CONFORMAL', str(tmp_path/'missing_conformal.json'))
    with TestClient(create_app(PositiveFixture())) as client:
        result = client.get('/ready')
        assert result.status_code == 503
        assert not result.json()['ready']
        assert result.json()['reason'].startswith('Conformal')


@pytest.mark.parametrize('sets', [[2], [0, 2], []])
def test_conformal_guards_are_honored_in_supplied_simulation(sets):
    with TestClient(create_app()) as client:
        data = client.post('/triage/simulate', json={'scores': [score(conformal_set=sets)],
            'config': {'review_budget': 0., 'recapture_budget': 0.}}).json()
        assert data['status'] == 'overloaded'
