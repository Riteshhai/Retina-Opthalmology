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


@pytest.mark.parametrize('relative,experiment,scope,payload', [
    ('ordinal_study_v1/evaluation.json', 'ordinal_randomization_v1', 'development_ordinal_diagnostics',
     {'comparisons': [{'metrics': {'patient_all_views_coverage': .81}}],
      'randomization_summaries': [{'mean_set_size': 3.6}]}),
    ('repeated_seeds_v1/summary.json', 'repeated_seeds_v1', 'development_selection',
     {'summary': {'dr': {'macro_f1': {'mean': .53, 'sample_standard_deviation': .01}}},
      'runs': [{'seed': 2026, 'head_rng_version': 'software_fixture_rng'}]}),
    ('policy_comparison_v1/dashboard_summary.json', 'policy_comparison_v1', 'development_out_of_fold_policy',
     {'summary': {'fully_evaluated_scenarios': 1, 'incomplete_scenarios': 1},
      'evaluation_ledger': {'review_cost': .25}, 'source_report_sha256': 'software_fixture',
      'paired_bootstrap': {'repeats': 1000, 'direction': 'left_minus_right', 'multiplicity_adjusted': False},
      'scenarios': [{'mode': 'ordinal', 'status': 'incomplete', 'pooled_metrics': None},
                    {'mode': 'no_cp', 'status': 'evaluated', 'pooled_metrics': {'missed_referable_auto': 1}}],
      'paired_comparisons': [{'status': 'evaluated',
                             'matched_realized_workload_within_every_fold': False}]}),
    ('aptos_v1/evaluation.json', 'aptos_development_v1', 'development_selection_and_independent_calibration',
     {'grading': {'macro_f1': .7}, 'candidates': [{'backbone': 'efficientnet_b0', 'macro_f1': .7}],
      'selected_candidate': 'efficientnet_b0',
      'comparisons': [{'method': 'lac', 'unit': 'image', 'calibration_unit_definition': 'fingerprint representative'}]}),
    ('aptos_finetune_v1/evaluation.json', 'aptos_finetune_v1',
     'development_finetuning_and_training_seed_diagnostics',
     {'summary': {'primary_training_seed': 2026, 'deployment_changed': False},
      'runs': [{'training_seed': 2026, 'grading': {'macro_f1': .7}}],
      'baseline': {'grading': {'macro_f1': .6}},
      'comparisons': [{'model_training_seed': 2026, 'randomization_seed': 2030, 'alpha': .1}]}),
    ('aptos_optimization_v2/evaluation.json', 'aptos_optimization_v2',
     'source_development_primary_seed_schedule_extension',
     {'summary': {'primary_training_seed': 2026, 'completed_runs': 1, 'seed_variance_estimated': False},
      'prior_primary': {'training_seed': 2026, 'selection_metrics': {'macro_f1': .6}},
      'runs': [{'training_seed': 2026, 'grading': {'macro_f1': .7}}],
      'paired_comparisons': {'four_epoch_primary': {'macro_f1': {'delta': .1}}},
      'comparisons': [{'model_training_seed': 2026, 'randomization_seed': 2030, 'alpha': .1}]}),
])
def test_saved_development_reports_preserve_evidence_scope(tmp_path, monkeypatch,
                                                          relative, experiment, scope, payload):
    """Generated contract fixtures are not measured research outputs."""
    report = tmp_path/relative
    report.parent.mkdir()
    dataset = 'APTOS_2019' if experiment.startswith('aptos_') else 'DeepDRiD'
    report.write_text(json.dumps({**payload, 'dataset': dataset,
                                 'limitations': ['Development fixture; no deployment selection.']}))
    monkeypatch.setenv('QSART_REPORTS_DIR', str(tmp_path))
    with TestClient(create_app()) as client:
        data = client.get('/research/metrics', params={'experiment_id': experiment})
        assert data.status_code == 200
        assert data.json()['scope'] == scope
        assert data.json()['metrics'] == payload
        assert data.json()['notes'] == ['Development fixture; no deployment selection.']
        records = client.get('/research/experiments').json()['experiments']
        assert records[0]['dataset'] == dataset


def test_external_data_audit_is_not_reported_as_model_evaluation(tmp_path, monkeypatch):
    """Generated external-data fixture contains no classifier outputs."""
    report = tmp_path/'idrid_data_v1/data_audit.json'
    report.parent.mkdir()
    payload = {'images': 10, 'images_decoded': 10,
        'source_provenance_status': 'local_package_download_origin_unconfirmed',
        'model_inference_performed': False, 'external_model_evaluation_completed': False,
        'official_partitions': {'official_train': {'images': 5, 'dr_grade_counts': {'0': 5}},
                                'official_test': {'images': 5, 'dr_grade_counts': {'1': 5}}},
        'cross_source_candidate_pairs': 1, 'independence_claim_blocked_by_detected_overlap': True}
    report.write_text(json.dumps({**payload, 'dataset': 'IDRiD', 'limitations': ['Generated data-readiness fixture.']}))
    monkeypatch.setenv('QSART_REPORTS_DIR', str(tmp_path))
    with TestClient(create_app()) as client:
        data = client.get('/research/metrics', params={'experiment_id': 'idrid_data_v1'}).json()
        assert data['scope'] == 'external_data_integrity_and_overlap_readiness_only'
        record = client.get('/research/experiments').json()['experiments'][0]
        assert record['partition'] == 'external_holdout' and record['dataset'] == 'IDRiD'
        assert data['metrics'] == payload and record['model_version'] is None
        assert 'grading' not in data['metrics'] and 'accuracy' not in data['metrics']


@pytest.mark.parametrize('stale', [False, True])
def test_bound_idrid_confirmation_preserves_audit_history_and_overlap_gate(tmp_path, monkeypatch, stale):
    import hashlib
    report = tmp_path/'idrid_data_v1/data_audit.json'
    report.parent.mkdir()
    raw = {'dataset': 'IDRiD', 'source_provenance_status': 'local_package_download_origin_unconfirmed',
        'private_output_sha256': {'external_manifest.csv': 'b'*64}, 'image_content_sha256': 'c'*64,
        'independence_claim_blocked_by_detected_overlap': True, 'model_inference_performed': False,
        'limitations': ['Generated provenance fixture.']}
    report.write_text(json.dumps(raw))
    before = report.read_bytes()
    receipt = {'schema_version': 1, 'dataset': 'IDRiD', 'confirmation_kind': 'user_statement',
        'confirmation_statement': 'Official source confirmed.', 'source_provenance_status': 'official_download_user_confirmed',
        'confirmed_at_utc': '2026-11-04T13:00:00+00:00', 'official_archive_checksum_verified': False,
        'source_urls': ['https://idrid.grand-challenge.org/Data/'],
        'data_audit_sha256': hashlib.sha256(before).hexdigest(),
        'manifest_sha256': ('a' if stale else 'b')*64, 'image_content_sha256': 'c'*64}
    (report.parent/'source_confirmation.json').write_text(json.dumps(receipt))
    monkeypatch.setenv('QSART_REPORTS_DIR', str(tmp_path))
    with TestClient(create_app()) as client:
        metrics = client.get('/research/metrics?experiment_id=idrid_data_v1').json()['metrics']
        expected = 'local_package_download_origin_unconfirmed' if stale else 'official_download_user_confirmed'
        assert metrics['source_provenance_status'] == expected
        assert metrics['independence_claim_blocked_by_detected_overlap'] is True
        assert metrics['model_inference_performed'] is False
        if not stale:
            assert metrics['audit_source_provenance_status'] == raw['source_provenance_status']
    assert report.read_bytes() == before


def test_idrid_population_review_has_no_classifier_performance(tmp_path, monkeypatch):
    report = tmp_path/'idrid_review_v1/data_review.json'
    report.parent.mkdir()
    report.write_text(json.dumps({'dataset': 'IDRiD', 'images': 5,
        'summary': {'populations': {'full_source': {'images': 5}, 'cross_source_filtered': {'images': 3}}},
        'model_inference_performed': False, 'external_model_evaluation_completed': False,
        'limitations': ['Generated mask fixture.']}))
    monkeypatch.setenv('QSART_REPORTS_DIR', str(tmp_path))
    with TestClient(create_app()) as client:
        data = client.get('/research/metrics?experiment_id=idrid_review_v1').json()
        assert data['scope'] == 'external_overlap_review_and_population_freeze'
        assert data['metrics']['summary']['populations']['cross_source_filtered']['images'] == 3
        assert data['metrics']['model_inference_performed'] is False and 'grading' not in data['metrics']


def test_locked_external_report_preserves_consumed_scope_and_all_models(tmp_path, monkeypatch):
    report = tmp_path/'idrid_external_v1/evaluation.json'
    report.parent.mkdir()
    payload = {'images': 6, 'summary': {'primary_model': 'finetune_seed2026'},
        'models': [{'model_id': name} for name in ('frozen_baseline', 'finetune_seed2026', 'finetune_seed2027', 'finetune_seed2028')],
        'external_outcomes_consumed': True, 'classifier_fitting_performed': False,
        'temperature_fitting_performed': False, 'conformal_threshold_fitting_performed': False,
        'paired_baseline_changes': [{'model_id': 'finetune_seed2026', 'accuracy': {'delta': -.1}}]}
    report.write_text(json.dumps({**payload, 'dataset': 'IDRiD', 'limitations': ['Generated external-outcome fixture.']}))
    monkeypatch.setenv('QSART_REPORTS_DIR', str(tmp_path))
    with TestClient(create_app()) as client:
        result = client.get('/research/metrics?experiment_id=idrid_external_v1').json()
        assert result['scope'] == 'locked_external_domain_diagnostics_consumed'
        assert result['metrics'] == payload
        record = client.get('/research/experiments').json()['experiments'][0]
        assert record['dataset'] == 'IDRiD' and record['partition'] == 'external_holdout'


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
