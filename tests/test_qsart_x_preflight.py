"""Exercise the panel preflight with read-only HTTP fixtures and archived aggregates."""

import json
from pathlib import Path

import httpx
import pytest

from scripts.preflight_qsart_x_demo import preflight


REPORTS = Path(__file__).resolve().parents[1] / 'reports'
API, PUBLIC, WORKSPACE, LAB = 18127, 18727, 18701, 18627
STUDIES = ('set_c_validation_v1', 'paired_view_v1')
CHECKS = {'api', 'model', 'portal', 'public', 'workspace', 'lab', 'experiment_registry', *STUDIES}


def read_report(relative_path):
    return json.loads((REPORTS / relative_path).read_bytes())


def html(title):
    return {'text': f'<!doctype html><html><head><title>{title}</title></head><body></body></html>',
            'headers': {'content-type': 'text/html; charset=utf-8'}}


@pytest.fixture
def responses():
    receipt = read_report('qsart_x_release_v1/verification.json')
    cpu = read_report('qsart_x_release_v1/cpu_profiles.json')
    artifacts = cpu['runtime']['artifact_sha256']
    health = {'status': 'ok', 'service': 'qsart-api'}
    ready = {'ready': True, 'policy_version': '2.0', 'conformal_status': 'available'}
    model = {
        'ready': True, 'research_only': True, 'device': 'cpu', 'data_purpose': 'real_research',
        'policy_version': '2.0', 'model_card_status': 'matched', 'model_version': cpu['model_version'],
        'cpu_threads': receipt['runtime']['cpu_threads'],
        'model_card': {
            'dr_checkpoint_sha256': artifacts['dr'], 'quality_checkpoint_sha256': artifacts['quality'],
            'calibration_sha256': artifacts['calibration'],
        },
        'conformal': {'status': 'available', 'artifact_sha256': artifacts['conformal']},
    }
    served = {
        (API, '/health'): {'json': health},
        (API, '/ready'): {'json': ready},
        (API, '/model/info'): {'json': model},
        (API, '/portal/ready'): {'json': {'ready': True, 'status': 'ready'}},
        (API, '/research/experiments'): {'json': {'experiments': [
            {'experiment_id': slug} for slug in STUDIES
        ]}},
        (PUBLIC, '/api/health'): {'json': health},
        (PUBLIC, '/api/ready'): {'json': ready},
        (PUBLIC, '/api/portal/session'): {'status_code': 404},
        (WORKSPACE, '/'): html('Screening workspace · QSART-X'),
        (WORKSPACE, '/api/model/info'): {'json': model},
        (WORKSPACE, '/api/portal/session'): {'status_code': 401},
        (LAB, '/_stcore/health'): {'text': 'ok', 'headers': {'content-type': 'text/plain'}},
    }
    for page in ('/demo', '/research', '/publications'):
        served[(PUBLIC, page)] = html('QSART-X · Retinal screening research')
    for slug in STUDIES:
        study = read_report(f'{slug}/evaluation.json')
        expected = receipt['reconciled_research_results'][slug]
        served[(API, f'/research/metrics?experiment_id={slug}')] = {'json': {
            'experiment_id': slug, 'metrics': {
                'evaluation_sha256': expected['evaluation_sha256'],
                'independent_audit_sha256': expected['independent_audit_sha256'],
                'scenarios': study['scenarios'], 'comparisons': study['comparisons'],
            },
        }}
    return served


def run(responses, requests, reports=REPORTS):
    def respond(request):
        assert request.method == 'GET'
        assert request.content == b''
        assert 'authorization' not in request.headers and 'cookie' not in request.headers
        assert request.url.host == '127.0.0.1'
        key = request.url.port, request.url.raw_path.decode('ascii')
        requests.append(key)
        return httpx.Response(**({'status_code': 200} | responses[key]))

    origins = [f'http://127.0.0.1:{port}' for port in (API, PUBLIC, WORKSPACE, LAB)]
    return preflight(*origins, reports=reports, transport=httpx.MockTransport(respond))


def assert_only_check_failed(result, name):
    assert result['status'] == 'attention_required'
    assert result['checks'][name]['status'] == 'failed'
    assert result['checks'][name]['message']
    assert {key for key, check in result['checks'].items() if check['status'] == 'passed'} == CHECKS - {name}


def test_preflight_uses_only_get_and_matches_both_archived_studies(responses):
    requests = []
    result = run(responses, requests)

    assert result['status'] == 'ready'
    assert set(result['checks']) == CHECKS
    assert all(check['status'] == 'passed' for check in result['checks'].values())
    assert {(API, f'/research/metrics?experiment_id={slug}') for slug in STUDIES} <= set(requests)
    assert {(PUBLIC, '/publications'), (WORKSPACE, '/api/portal/session')} <= set(requests)


@pytest.mark.parametrize(('check', 'route', 'body'), [
    ('portal', (API, '/portal/ready'), {'status_code': 503, 'json': {'ready': False}}),
    ('public', (PUBLIC, '/api/ready'), html('QSART-X · Retinal screening research')),
    ('lab', (LAB, '/_stcore/health'), {
        'status_code': 307, 'headers': {'location': 'http://remote.invalid/not-the-research-lab'},
    }),
])
def test_preflight_reports_independent_service_failures_without_following_redirects(responses, check, route, body):
    responses[route] = body
    result = run(responses, [])

    assert_only_check_failed(result, check)
    assert 'remote.invalid' not in json.dumps(result)


def test_preflight_rejects_changed_scenario_despite_matching_hash_and_count(responses):
    route = API, '/research/metrics?experiment_id=set_c_validation_v1'
    first = responses[route]['json']['metrics']['scenarios'][0]
    first['status'] = 'synthetic-changed-row'

    result = run(responses, [])

    assert_only_check_failed(result, 'set_c_validation_v1')
    assert 'synthetic-changed-row' not in json.dumps(result)


def test_wrong_conformal_hash_fails_model_check_when_both_bridges_agree(responses):
    conformal = responses[(API, '/model/info')]['json']['conformal']
    conformal['artifact_sha256'] = '0' * 64

    assert_only_check_failed(run(responses, []), 'model')


def test_unavailable_model_keeps_health_and_working_bridges_visible(responses):
    ready = {
        'status_code': 503,
        'json': {'ready': False, 'policy_version': '2.0', 'conformal_status': 'not_configured'},
    }
    info = {'json': {'ready': False, 'reason': 'synthetic-local-path'}}
    responses[(API, '/ready')] = responses[(PUBLIC, '/api/ready')] = ready
    responses[(API, '/model/info')] = responses[(WORKSPACE, '/api/model/info')] = info

    result = run(responses, [])

    assert_only_check_failed(result, 'model')
    assert 'synthetic-local-path' not in json.dumps(result)
