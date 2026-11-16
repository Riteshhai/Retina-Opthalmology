"""Verify a loopback QSART-X Docker workspace with disposable accounts and a synthetic photo."""

import argparse
from contextlib import ExitStack
from hashlib import sha256
from http.cookies import SimpleCookie
from io import BytesIO
import json
import math
from pathlib import Path
import re
import sys
from urllib.parse import urlsplit
from uuid import UUID

import httpx
from PIL import Image, PngImagePlugin


PREFIX = '/api/portal'
CASES = f'{PREFIX}/cases'
MAX_UPLOAD_BYTES = 12 * 1024 * 1024
MAX_AUTH_BODY_BYTES = 64 * 1024
SCREENING_ACTIONS = {'AUTO_NONREFERABLE', 'RECAPTURE', 'SPECIALIST_REVIEW'}
EXPECTED_ROLES = {
    'operator': 'research_operator', 'other_operator': 'research_operator',
    'reviewer': 'clinician_reviewer', 'viewer': 'research_viewer',
}
MARKER = 'QSART_X_SYNTHETIC_METADATA'
FILENAME = 'qsart-x-release-synthetic.png'
OVERRIDE_NOTES = 'Synthetic case for QSART-X release verification.'
SNAPSHOTS = ('detail_sha256', 'image_sha256', 'report_sha256')


class CheckFailed(ValueError):
    """Report a failed check without request bodies, cookies, or case identifiers."""


def require(condition, description):
    if not condition:
        raise CheckFailed(description)


def loopback_origin(value):
    try:
        url = urlsplit(value)
        if (url.scheme == 'http' and url.hostname in {'127.0.0.1', 'localhost'}
                and url.port is not None and url.username is None and url.password is None
                and url.path in ('', '/') and not url.query and not url.fragment):
            return f'http://{url.hostname}:{url.port}'
    except ValueError:
        pass
    raise argparse.ArgumentTypeError('Use a loopback HTTP origin with an explicit port.')


def load_credentials(stream):
    credentials = json.load(stream)
    require(isinstance(credentials, dict) and credentials.keys() == EXPECTED_ROLES.keys(),
            'Provide the four disposable account records on stdin.')
    for account in credentials.values():
        require(isinstance(account, dict) and account.keys() == {'username', 'password'}
                and all(isinstance(value, str) and value.strip() for value in account.values()),
                'Each disposable account needs a username and password.')
    require(len({account['username'] for account in credentials.values()}) == len(EXPECTED_ROLES),
            'Use a distinct username for each disposable account.')
    return credentials


def check(response, status, label):
    require(response.status_code == status, f'{label} returned HTTP {response.status_code} instead of {status}.')
    return response


def login(client, credentials, role):
    result = check(client.post(f'{PREFIX}/login', json=credentials), 200, 'Disposable account login')
    identity = {'username': credentials['username'], 'role': role}
    require(result.json() == identity, 'Account identity or role differs.')
    cookie = SimpleCookie(result.headers.get('set-cookie', ''))
    session = cookie.get('qsart_portal_session')
    require(session and session['httponly'] and session['samesite'].lower() == 'strict'
            and session['path'] == PREFIX and session['domain'] == '', 'Session cookie scope or flags differ.')
    require(result.headers.get('cache-control') == 'no-store', 'Login response allows caching.')
    require(check(client.get(f'{PREFIX}/session'), 200, 'Account session').json() == identity,
            'Session identity or role differs.')


def logout(client):
    check(client.post(f'{PREFIX}/logout'), 204, 'Logout')
    check(client.get(f'{PREFIX}/session'), 401, 'Signed-out session')


def verify_services(api, workspace, public, lab):
    for path in ('/health', '/ready'):
        check(api.get(path), 200, f'API readiness {path}')
    portal = check(api.get('/portal/ready'), 200, 'Migrated portal database').json()
    require(portal == {'ready': True, 'status': 'ready'}, 'Portal database readiness differs.')
    info = check(api.get('/model/info'), 200, 'Trained model metadata').json()
    proxy_info = check(workspace.get('/api/model/info'), 200, 'Workspace model bridge').json()
    model_version = info['model_version']
    require(info['ready'] is True and info['device'] == 'cpu' and info['data_purpose'] == 'real_research'
            and isinstance(model_version, str) and bool(model_version)
            and info['policy_version'] == '2.0' and info['max_upload_bytes'] == MAX_UPLOAD_BYTES
            and info['conformal']['status'] == 'available'
            and all(info[field] == proxy_info[field] for field in ('model_version', 'quality_model_version',
                                                                    'policy_version', 'conformal')),
            'Expected the release CPU model, conformal artifact and upload contract through both routes.')
    check(workspace.get('/'), 200, 'Workspace frontend')
    check(public.get('/'), 200, 'Public frontend')
    check(public.get('/api/health'), 200, 'Public API health bridge')
    check(public.get('/api/ready'), 200, 'Public API readiness bridge')
    check(lab.get('/_stcore/health'), 200, 'Research Lab health')
    check(public.get(CASES), 404, 'Public case listing isolation')
    check(public.post(CASES), 404, 'Public case upload isolation')
    check(public.post('/api/predict'), 404, 'Public inference isolation')
    check(workspace.post('/api/predict'), 404, 'Workspace direct inference isolation')
    return model_version


def verify_auth_boundary(client, credentials, public_origin):
    check(client.get(CASES), 401, 'Anonymous case listing')
    check(client.post(f'{PREFIX}/login', headers={'Origin': public_origin}, json=credentials),
          403, 'Cross-origin login denial')
    check(client.post(f'{PREFIX}/login', headers={'X-QSART-Workspace': '0'}, json=credentials),
          403, 'Unmarked login denial')
    malformed = check(client.post(f'{PREFIX}/login', json={'username': [], 'password': MARKER}),
                      422, 'Invalid credential shape')
    require(MARKER not in malformed.text, 'Invalid credential response echoes supplied values.')
    check(client.post(f'{PREFIX}/login', headers={'Content-Type': 'application/json'},
                      content=b' ' * (MAX_AUTH_BODY_BYTES + 1)), 413, 'Oversize login body')


def synthetic_png(path):
    with Image.open(path) as source:
        photo = source.convert('RGB')
    with photo, BytesIO() as output:
        photo.info.clear()
        png_info = PngImagePlugin.PngInfo()
        png_info.add_text('SyntheticReleaseFixture', MARKER)
        photo.save(output, format='PNG', pnginfo=png_info)
        payload = output.getvalue()
    require(len(payload) <= MAX_UPLOAD_BYTES, 'The supplied synthetic image exceeds the upload size limit.')
    return payload


def verify_uploads(operator, viewer, reviewer, payload, model_version):
    files = {'file': (FILENAME, payload, 'image/png')}
    check(viewer.post(CASES, files=files), 403, 'Viewer upload denial')
    check(reviewer.post(CASES, files=files), 403, 'Reviewer upload denial')
    check(operator.post(CASES, files={'file': ('unsupported.txt', b'test fixture', 'text/plain')}),
          415, 'Unsupported upload type')
    check(operator.post(CASES, files={'file': (FILENAME, b'test fixture', 'image/png')}),
          422, 'Corrupt image upload')
    check(operator.post(CASES, files={'file': (FILENAME, b'\0' * (MAX_UPLOAD_BYTES + 1), 'image/png')}),
          413, 'Oversize upload')
    require(check(operator.get(CASES), 200, 'Empty disposable case queue').json() == [],
            'Use a fresh disposable operator with no existing cases.')
    created = check(operator.post(CASES, files=files), 201, 'Create synthetic case')
    require(FILENAME not in created.text and MARKER not in created.text, 'Upload metadata leaked into case data.')
    detail = created.json()
    identifier = str(UUID(detail['id']))
    prediction = detail['prediction']
    scores = prediction['grade_probabilities']
    require(prediction['model_version'] == model_version and prediction['policy_version'] == '2.0'
            and len(scores) == 5 and all(0. <= value <= 1. for value in scores)
            and math.isclose(sum(scores), 1., abs_tol=1e-5)
            and prediction['predicted_grade'] == scores.index(max(scores))
            and prediction['action'] in SCREENING_ACTIONS and prediction['policy_scope'] == 'single_image'
            and prediction['second_pass_activated'] is False and prediction['conformal']['status'] == 'available'
            and 0. <= prediction['quality']['poor_probability'] <= 1.
            and 0. <= prediction['uncertainty']['entropy'] <= 1., 'Trained case prediction contract differs.')
    require(detail['effective_action'] == prediction['action'] and detail['reviews'] == []
            and [event['event'] for event in detail['audit']] == ['case_created'],
            'Initial case disposition or history differs.')
    return identifier, prediction


def verify_case_boundaries(identifier, clients):
    path = f'{CASES}/{identifier}'
    owned = check(clients['operator'].get(CASES), 200, 'Owning operator case queue').json()
    require([case['id'] for case in owned] == [identifier], 'The expected case is absent from its owner queue.')
    foreign = clients['other_operator']
    require(check(foreign.get(CASES), 200, 'Other operator case queue').json() == [],
            'A different operator can list an unrelated case.')
    for suffix in ('', '/image', '/report'):
        check(foreign.get(f'{path}{suffix}'), 404, 'Other operator case scope')
    viewer = clients['viewer']
    check(viewer.get(path), 200, 'Viewer case read')
    check(viewer.get(f'{path}/report'), 403, 'Viewer report denial')


def verify_review(identifier, prediction, clients, reviewer_name):
    path = f'{CASES}/{identifier}/reviews'
    action = 'RECAPTURE' if prediction['action'] != 'RECAPTURE' else 'SPECIALIST_REVIEW'
    grade = (prediction['predicted_grade'] + 1) % 5
    assessment = {'grade': grade, 'action': action, 'notes': OVERRIDE_NOTES}
    for role in ('operator', 'viewer'):
        check(clients[role].post(path, json=assessment), 403, 'Unprivileged review denial')
    reviewer = clients['reviewer']
    check(reviewer.post(path, json={**assessment, 'notes': ' '}), 422, 'Override justification requirement')
    overridden = check(reviewer.post(path, json=assessment), 200, 'Reviewer action override').json()
    row, = overridden['reviews']
    require(overridden['prediction'] == prediction and overridden['effective_action'] == action
            and all(row[field] == assessment[field] for field in ('grade', 'action', 'notes'))
            and row['is_override'] is True and row['grade_disagreement'] is True and row['reviewer'] == reviewer_name,
            'Override changed the original model result or omitted reviewer evidence.')
    updated = check(reviewer.post(path, json={'grade': prediction['predicted_grade']}),
                    200, 'Append reviewer assessment').json()
    require(updated['prediction'] == prediction and updated['effective_action'] == action
            and len(updated['reviews']) == 2 and updated['reviews'][0] == row
            and [event['event'] for event in updated['audit']] == ['case_created', 'action_overridden', 'review_recorded'],
            'Reviewer history or effective disposition was replaced instead of preserved.')
    return updated


def case_snapshots(client, identifier, expected_photo=None):
    path = f'{CASES}/{identifier}'
    detail = check(client.get(path), 200, 'Private case detail')
    image = check(client.get(f'{path}/image'), 200, 'Private case image')
    report = check(client.get(f'{path}/report'), 200, 'Private case report')
    require(all(item.headers.get('cache-control') == 'no-store' for item in (detail, image, report)),
            'Private case data allows caching.')
    require(report.headers.get('content-disposition', '').startswith('attachment;')
            and report.headers.get('content-type') == 'application/json' and report.json() == detail.json(),
            'Private report download differs from the case detail.')
    with Image.open(BytesIO(image.content)) as photo:
        require(photo.format == 'PNG' and 'SyntheticReleaseFixture' not in photo.info
                and MARKER.encode() not in image.content, 'Original image metadata was retained.')
        if expected_photo is not None:
            with Image.open(BytesIO(expected_photo)) as fixture:
                require(photo.mode == fixture.mode and photo.size == fixture.size
                        and photo.tobytes() == fixture.tobytes(), 'Stored synthetic image pixels differ.')
    return detail.json(), {name: sha256(response.content).hexdigest()
                           for name, response in zip(SNAPSHOTS, (detail, image, report), strict=True)}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as output:
        output.write(json.dumps(value, indent=2, allow_nan=False) + '\n')


def replay_case(client, path):
    witness = json.loads(path.read_bytes())
    require(isinstance(witness, dict) and witness.keys() == {'case_id', *SNAPSHOTS}
            and isinstance(witness['case_id'], str) and all(
                isinstance(witness[key], str) and re.fullmatch(r'[0-9a-f]{64}', witness[key]) for key in SNAPSHOTS
            ), 'The private synthetic-case witness is invalid.')
    identifier = str(UUID(witness['case_id']))
    _, actual = case_snapshots(client, identifier)
    require(all(actual[key] == witness[key] for key in SNAPSHOTS),
            'The case, image, or report changed across the declared restart.')


def verify_release(args, credentials):
    options = {'timeout': 90, 'trust_env': False, 'follow_redirects': False}
    headers = {'Origin': args.workspace, 'X-QSART-Workspace': '1'}
    receipt = {'status': 'verified', 'scope': 'loopback HTTP against the running research release',
               'phase': 'restart_replay' if args.replay_witness else 'screening_and_review',
               'verifier_sha256': sha256(Path(__file__).read_bytes()).hexdigest()}
    with ExitStack() as stack:
        api, public, lab = (stack.enter_context(httpx.Client(base_url=url, **options))
                            for url in (args.api, args.public, args.lab))
        unsigned = stack.enter_context(httpx.Client(base_url=args.workspace, headers=headers, **options))
        model_version = verify_services(api, unsigned, public, lab)
        receipt.update(model_version=model_version, services_ready=True, public_upload_denied=True)
        if args.replay_witness:
            login(unsigned, credentials['reviewer'], EXPECTED_ROLES['reviewer'])
            stack.callback(logout, unsigned)
            replay_case(unsigned, args.replay_witness)
            return {**receipt, 'case_image_and_report_match_private_witness': True,
                    'restart_scope': 'Restart performed by the caller between verification phases.'}

        verify_auth_boundary(unsigned, credentials['operator'], args.public)
        clients = {role: stack.enter_context(httpx.Client(base_url=args.workspace, headers=headers, **options))
                   for role in EXPECTED_ROLES}
        for role, client in clients.items():
            login(client, credentials[role], EXPECTED_ROLES[role])
            stack.callback(logout, client)
        payload = synthetic_png(args.image)
        identifier, prediction = verify_uploads(clients['operator'], clients['viewer'], clients['reviewer'],
                                                payload, model_version)
        verify_case_boundaries(identifier, clients)
        updated = verify_review(identifier, prediction, clients, credentials['reviewer']['username'])
        detail, checksums = case_snapshots(clients['reviewer'], identifier, payload)
        require(detail == updated, 'Saved case differs from its latest authorized review.')
        if args.write_witness:
            write_json(args.write_witness, {'case_id': identifier, **checksums})
    return {**receipt, 'fixture': 'synthetic retinal image', 'role_and_owner_boundaries_verified': True,
            'origin_and_session_controls_verified': True, 'invalid_uploads_rejected': True,
            'oversize_login_body_rejected': True,
            'stored_image_pixels_match_and_metadata_removed': True,
            'immutable_prediction_and_review_history_verified': True,
            'private_case_report_verified': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__, epilog=(
        'Read credentials from a JSON object on stdin with operator, other_operator, reviewer and viewer keys, '
        'each containing username and password. Run on an isolated Compose project.'))
    for name in ('workspace', 'api', 'public', 'lab'):
        parser.add_argument(f'--{name}', type=loopback_origin, required=True)
    parser.add_argument('--image', type=Path, help='Local synthetic image for the disposable case')
    parser.add_argument('--output', type=Path, help='Optional new aggregate receipt path')
    witnesses = parser.add_mutually_exclusive_group()
    witnesses.add_argument('--write-witness', type=Path, help='New private, ignored case witness for restart replay')
    witnesses.add_argument('--replay-witness', type=Path, help='Compare the same synthetic case after caller restart')
    args = parser.parse_args()
    if len({args.workspace, args.api, args.public, args.lab}) != 4:
        parser.error('Provide distinct loopback origins for the four services.')
    if (args.replay_witness is None) != (args.image is not None):
        parser.error('Use --image for screening and review or --replay-witness for the replay phase.')
    if any(path is not None and path.exists() for path in (args.output, args.write_witness)):
        parser.error('Choose new paths for output files.')
    try:
        receipt = verify_release(args, load_credentials(sys.stdin))
        if args.output:
            write_json(args.output, receipt)
    except CheckFailed as exc:
        parser.exit(1, f'Verification failed. {exc}\n')
    except (KeyError, OSError, TypeError, ValueError, httpx.HTTPError):
        parser.exit(1, 'Verification failed. Check local service readiness and input formats.\n')
    print(json.dumps(receipt, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
