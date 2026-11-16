from hashlib import sha256
import json

from fastapi.testclient import TestClient

from api.main import create_app
from api.research import PAIRED_VIEW_AUDIT, PAIRED_VIEW_DEVELOPMENT, PAIRED_VIEW_EVALUATION, PAIRED_VIEW_ID


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload).encode()
    path.write_bytes(encoded)
    return sha256(encoded).hexdigest()


def write_study(tmp_path):
    source_commit, witness_digest, manifest_digest = 'a' * 40, 'b' * 64, 'c' * 64
    dataset_digest = write_json(tmp_path / PAIRED_VIEW_DEVELOPMENT, {'development_v1': {
        'manifest_sha256': manifest_digest,
        'partitions': {'train': {'subjects': 212, 'images': 848}, 'validation': {'subjects': 42, 'images': 168}},
    }})
    inputs = {'manifest': manifest_digest, 'dataset_report': dataset_digest,
              **{key: 'd' * 64 for key in ('dr', 'quality', 'calibration', 'model_card', 'selection_lock')}}
    study = {
        'experiment_id': PAIRED_VIEW_ID, 'source_commit': source_commit, 'partition': 'validation',
        'model_inference_performed': True, 'private_witness_sha256': witness_digest,
        'input_provenance': {'input_sha256': inputs, 'model_version': 'fixture-model'},
        'fit_population': {'split': 'train', 'patients': 212, 'eyes': 424, 'photos': 848},
        'selection_population': {'split': 'validation', 'patients': 42, 'eyes': 84, 'photos': 168},
        'model': {'feature_names': ['confidence']}, 'budgets': [.25],
        'scenarios': [{'id': 'voi:1', 'slots': 21, 'outcomes': {'eyes': 84}}],
        'comparisons': [{'candidate': 'voi:1', 'reference': 'quality:1', 'extra_views': 21}],
        'fit_rows': [{'patient_id': 'private-fixture', 'selected': [True]}],
    }
    study_digest = write_json(tmp_path / PAIRED_VIEW_EVALUATION, study)
    audit = {'audit_status': 'passed', 'experiment_id': PAIRED_VIEW_ID, 'source_commit': source_commit,
             'evaluation_sha256': study_digest, 'private_witness_sha256': witness_digest, 'input_sha256': inputs,
             'fit_patients': 212, 'selection_patients': 42, 'scenarios_verified': 1, 'comparisons_verified': 1,
             'routing_reconstructed_from_first_view': True}
    audit_digest = write_json(tmp_path / PAIRED_VIEW_AUDIT, audit)
    return study, audit, study_digest, audit_digest


def test_paired_view_publishes_allowlisted_metrics_with_digest_bindings(monkeypatch, tmp_path):
    monkeypatch.setenv('QSART_REPORTS_DIR', str(tmp_path))
    study, _, study_digest, audit_digest = write_study(tmp_path)
    with TestClient(create_app()) as client:
        records = client.get('/research/experiments').json()['experiments']
        response = client.get('/research/metrics', params={'experiment_id': PAIRED_VIEW_ID})

    assert len(records) == 1
    assert records[0]['dataset'] == 'DeepDRiD'
    assert records[0]['partition'] == 'validation'
    assert records[0]['model_version'] == 'fixture-model'
    assert response.status_code == 200
    assert response.json()['scope'] == 'development_retrospective_paired_view_acquisition'
    assert response.json()['metrics'] == {
        **{key: study[key] for key in ('fit_population', 'selection_population', 'model', 'budgets',
                                       'scenarios', 'comparisons', 'source_commit')},
        'evaluation_sha256': study_digest, 'independent_audit_sha256': audit_digest,
    }
    assert all(token not in response.text for token in ('private-fixture', 'private_witness_sha256', 'fit_rows'))


def test_invalid_or_unavailable_audit_cannot_publish_even_via_legacy_list(monkeypatch, tmp_path):
    monkeypatch.setenv('QSART_REPORTS_DIR', str(tmp_path))
    write_json(tmp_path / 'spec_v1/model_comparison.json', {'experiments': [
        {'experiment_id': experiment_id, 'dataset': 'DeepDRiD', 'partition': 'validation', 'scope': 'fixture'}
        for experiment_id in ('other_fixture', PAIRED_VIEW_ID)
    ]})
    with TestClient(create_app()) as client:
        def absent():
            assert client.get('/research/metrics', params={'experiment_id': PAIRED_VIEW_ID}).status_code == 404
            result = client.get('/research/experiments')
            assert result.status_code == 200
            assert [row['experiment_id'] for row in result.json()['experiments']] == ['other_fixture']

        study, audit, _, _ = write_study(tmp_path)
        (tmp_path / PAIRED_VIEW_AUDIT).unlink()
        absent()

        write_json(tmp_path / PAIRED_VIEW_AUDIT, audit)
        write_json(tmp_path / PAIRED_VIEW_EVALUATION, {**study, 'budgets': [.50]})
        absent()

        write_study(tmp_path)
        write_json(tmp_path / PAIRED_VIEW_AUDIT, {**audit, 'private_witness_sha256': 'e' * 64})
        absent()

        leaked = {**study, 'scenarios': [{'id': 'voi:1', 'selected': [True]}]}
        digest = write_json(tmp_path / PAIRED_VIEW_EVALUATION, leaked)
        write_json(tmp_path / PAIRED_VIEW_AUDIT, {**audit, 'evaluation_sha256': digest})
        absent()

        write_study(tmp_path)
        development_path = tmp_path / PAIRED_VIEW_DEVELOPMENT
        development_path.write_bytes(development_path.read_bytes() + b'\n')
        absent()
