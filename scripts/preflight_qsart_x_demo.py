"""Check a local QSART-X panel demo using only read-only HTTP requests."""

import argparse
from contextlib import ExitStack
from hashlib import sha256
import json
from pathlib import Path

import httpx

from scripts.verify_qsart_x_release import CheckFailed, check, loopback_origin, require

REPORTS = Path(__file__).resolve().parents[1] / 'reports'
RELEASE = 'qsart_x_release_v1'
STUDIES = ('set_c_validation_v1', 'paired_view_v1')
HEALTH = {'status': 'ok', 'service': 'qsart-api'}


def object_json(content, label):
    try:
        result = json.loads(content)
    except ValueError:
        raise CheckFailed(f'{label} did not contain valid JSON.') from None
    require(isinstance(result, dict), f'{label} did not contain a JSON object.')
    return result


def file_json(path, digest=None):
    try:
        content = path.read_bytes()
    except OSError:
        raise CheckFailed(f'Missing or unreadable release file {path.parent.name}/{path.name}.') from None
    if digest is not None:
        require(sha256(content).hexdigest() == digest,
                f'{path.parent.name}/{path.name} differs from the release receipt.')
    return object_json(content, path.name)


def json_response(response, status=200):
    label = response.url.path
    if status is not None:
        check(response, status, label)
    require(response.headers.get('content-type', '').partition(';')[0] == 'application/json',
            f'{label} did not serve JSON.')
    return object_json(response.content, label)


def verify_health(client, route):
    require(json_response(client.get(route)) == HEALTH, f'{route} did not report a live QSART API.')


def model_ready(client, route):
    response = client.get(route)
    require(response.status_code in (200, 503), f'{route} returned HTTP {response.status_code}.')
    payload = json_response(response, status=None)
    require(payload.get('ready') is (response.status_code == 200), f'{route} returned inconsistent readiness.')
    return payload


def verify_model(api, reports):
    readiness = model_ready(api, '/ready')
    require(readiness['ready'], 'Trained model artifacts are unavailable. Check the release checkpoint mounts.')
    require(readiness.get('policy_version') == '2.0' and readiness.get('conformal_status') == 'available',
            'The demo needs policy 2.0 with the conformal artifact loaded.')
    info = json_response(api.get('/model/info'))
    release = file_json(reports / RELEASE / 'verification.json')
    profiles = file_json(reports / RELEASE / 'cpu_profiles.json', release['cpu_profiles_sha256'])
    hashes = profiles['runtime']['artifact_sha256']
    require(info.get('ready') is True and info.get('research_only') is True and info.get('policy_version') == '2.0'
            and info.get('device') == 'cpu' and info.get('data_purpose') == 'real_research'
            and info.get('model_version') == profiles['model_version'] and info.get('model_card_status') == 'matched',
            'The API did not report the release CPU research model with a matched model card.')
    card, conformal = info.get('model_card'), info.get('conformal')
    require(isinstance(card, dict) and all(card.get(field) == hashes[artifact] for artifact, field in (
        ('dr', 'dr_checkpoint_sha256'), ('quality', 'quality_checkpoint_sha256'), ('calibration', 'calibration_sha256'),
    )) and isinstance(conformal, dict) and conformal.get('status') == 'available'
            and conformal.get('artifact_sha256') == hashes['conformal'],
            'Model card or conformal metadata differs from the verified release.')
    threads = info.get('cpu_threads')
    require(type(threads) is int and threads > 0, 'The model did not report a valid CPU thread count.')
    return {'model_version': info['model_version'], 'cpu_threads': threads}


def verify_portal(api):
    require(json_response(api.get('/portal/ready')) == {'ready': True, 'status': 'ready'},
            'The portal database did not report readiness.')


def verify_page(client, route, title):
    response = check(client.get(route), 200, route)
    require(response.headers.get('content-type', '').startswith('text/html') and f'<title>{title}</title>' in response.text,
            f'{route} did not serve the expected QSART-X page.')


def verify_public(public, api):
    for route in ('/demo', '/research', '/publications'):
        verify_page(public, route, 'QSART-X · Retinal screening research')
    verify_health(public, '/api/health')
    require(model_ready(public, '/api/ready') == model_ready(api, '/ready'),
            'Public model readiness did not match the API.')
    check(public.get('/api/portal/session'), 404, 'Public portal session route')


def verify_workspace(workspace, api):
    verify_page(workspace, '/', 'Screening workspace · QSART-X')
    require(json_response(workspace.get('/api/model/info')) == json_response(api.get('/model/info')),
            'Workspace model information did not match the API.')
    check(workspace.get('/api/portal/session'), 401, 'Signed-out workspace session')


def verify_lab(lab):
    response = check(lab.get('/_stcore/health'), 200, 'Research lab')
    require(response.headers.get('content-type', '').startswith('text/plain') and response.text.strip() == 'ok',
            'The lab did not return the Streamlit health response.')


def verify_registry(api):
    experiments = json_response(api.get('/research/experiments')).get('experiments')
    require(isinstance(experiments, list) and all(isinstance(record, dict) for record in experiments),
            'The experiment registry did not return a list of records.')
    for study in STUDIES:
        require(any(row.get('experiment_id') == study for row in experiments), f'{study} is missing from the registry.')


def verify_study(api, reports, study):
    release = file_json(reports / RELEASE / 'verification.json')
    expected = release['reconciled_research_results'][study]
    report = file_json(reports / study / 'evaluation.json', expected['evaluation_sha256'])
    audit = file_json(reports / study / 'independent_audit.json', expected['independent_audit_sha256'])
    require(audit.get('audit_status') == 'passed' and audit.get('evaluation_sha256') == expected['evaluation_sha256'],
            f'{study} does not have a passing independent audit for its report.')
    observed = json_response(api.get('/research/metrics', params={'experiment_id': study}))
    metrics = observed.get('metrics')
    require(observed.get('experiment_id') == study and isinstance(metrics, dict)
            and all(metrics.get(field) == expected[field] for field in ('evaluation_sha256', 'independent_audit_sha256')),
            f'{study} live report or audit digest differs from the release receipt.')
    for field in ('scenarios', 'comparisons'):
        rows = metrics.get(field)
        require(isinstance(rows, list) and len(rows) == expected[field] and rows == report.get(field),
                f'{study} live {field} differ from the audited report.')
    return {field: expected[field] for field in ('scenarios', 'comparisons')}


def record(checks, name, run):
    try:
        checks[name] = {'status': 'passed', **(run() or {})}
    except CheckFailed as error:
        checks[name] = {'status': 'failed', 'message': str(error)}
    except httpx.RequestError as error:
        checks[name] = {'status': 'failed', 'message': f'Local HTTP request failed ({type(error).__name__}).'}
    except (KeyError, TypeError):
        checks[name] = {'status': 'failed', 'message': 'A release record has an unexpected format.'}


def preflight(api, public, workspace, lab, *, reports=REPORTS, transport=None):
    checks = {}
    with ExitStack() as stack:
        api, public, workspace, lab = (
            stack.enter_context(httpx.Client(base_url=loopback_origin(url), timeout=4, trust_env=False,
                                             follow_redirects=False, transport=transport))
            for url in (api, public, workspace, lab)
        )
        record(checks, 'api', lambda: verify_health(api, '/health'))
        record(checks, 'model', lambda: verify_model(api, reports))
        record(checks, 'portal', lambda: verify_portal(api))
        record(checks, 'public', lambda: verify_public(public, api))
        record(checks, 'workspace', lambda: verify_workspace(workspace, api))
        record(checks, 'lab', lambda: verify_lab(lab))
        record(checks, 'experiment_registry', lambda: verify_registry(api))
        for study in STUDIES:
            record(checks, study, lambda: verify_study(api, reports, study))
    ready = all(result['status'] == 'passed' for result in checks.values())
    return {'status': 'ready' if ready else 'attention_required', 'checks': checks}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag, port in (('api', 18000), ('public', 18700), ('workspace', 18701), ('lab', 18501)):
        parser.add_argument(f'--{flag}', type=loopback_origin, default=f'http://127.0.0.1:{port}')
    args = parser.parse_args()
    result = preflight(args.api, args.public, args.workspace, args.lab)
    print(json.dumps(result, indent=2))
    return 0 if result['status'] == 'ready' else 1


if __name__ == '__main__':
    raise SystemExit(main())
