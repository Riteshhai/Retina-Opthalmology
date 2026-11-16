"""Software-only checks for local release verifier input and persistence contracts."""

from argparse import ArgumentTypeError
from io import BytesIO, StringIO
import json

import httpx
from PIL import Image, PngImagePlugin
import pytest

from scripts import verify_qsart_x_release as verifier


def test_verifier_accepts_only_bare_loopback_origins():
    assert verifier.loopback_origin('http://127.0.0.1:18701/') == 'http://127.0.0.1:18701'
    assert verifier.loopback_origin('http://localhost:18701') == 'http://localhost:18701'
    for url in ('https://localhost:18701', 'http://remote.invalid:8080', 'http://localhost/portal',
                'http://a@localhost:18701', 'http://127.0.0.1:99999', 'http://localhost:18701?x=y'):
        with pytest.raises(ArgumentTypeError):
            verifier.loopback_origin(url)


def test_input_requires_distinct_disposable_accounts_and_redacts_errors():
    credentials = {key: {'username': key, 'password': 'synthetic-password'} for key in verifier.EXPECTED_ROLES}
    assert verifier.load_credentials(StringIO(json.dumps(credentials))) == credentials
    credentials['other_operator']['username'] = credentials['operator']['username']
    with pytest.raises(verifier.CheckFailed, match='distinct username') as error:
        verifier.load_credentials(StringIO(json.dumps(credentials)))
    assert 'synthetic-password' not in str(error.value)


def test_prepared_photo_preserves_pixels_and_marks_synthetic_upload(tmp_path):
    path = tmp_path / 'software.png'
    source_info = PngImagePlugin.PngInfo()
    source_info.add_text('OriginalFixtureText', 'Synthetic input metadata')
    with Image.new('RGB', (32, 32), '#cc3322') as image:
        image.save(path, pnginfo=source_info)
    with Image.open(BytesIO(verifier.synthetic_png(path))) as result:
        assert result.info == {'SyntheticReleaseFixture': verifier.MARKER}
        assert result.getpixel((0, 0)) == (204, 51, 34)


def test_login_body_cap_is_verified():
    workspace, public = 'http://127.0.0.1:18701', 'http://127.0.0.1:18700'
    state = {'oversize_status': 413}

    def transport(request):
        if request.method == 'GET':
            assert request.url.path == verifier.CASES
            return httpx.Response(401)
        assert request.url.path == f'{verifier.PREFIX}/login'
        if request.headers['Origin'] == public or request.headers['X-QSART-Workspace'] != '1':
            return httpx.Response(403)
        if len(request.content) > verifier.MAX_AUTH_BODY_BYTES:
            assert request.headers['Content-Type'] == 'application/json'
            assert len(request.content) == verifier.MAX_AUTH_BODY_BYTES + 1
            return httpx.Response(state['oversize_status'])
        return httpx.Response(422)

    with httpx.Client(base_url=workspace, headers={'Origin': workspace, 'X-QSART-Workspace': '1'},
                      transport=httpx.MockTransport(transport)) as client:
        credentials = {'username': 'synthetic-test', 'password': 'synthetic-password'}
        verifier.verify_auth_boundary(client, credentials, public)
        state['oversize_status'] = 422
        with pytest.raises(verifier.CheckFailed, match='Oversize login body'):
            verifier.verify_auth_boundary(client, credentials, public)


def test_restart_replay_checks_private_case_image_and_report_digests(tmp_path):
    identifier = '00000000-0000-4000-8000-000000000001'
    case_path = f'{verifier.CASES}/{identifier}'
    detail = {'id': identifier, 'evidence': 'synthetic software test'}
    with BytesIO() as stream, Image.new('RGB', (32, 32), '#cc3322') as image:
        image.save(stream, format='PNG')
        original = stream.getvalue()
    state = {'image': original}

    def transport(request):
        private = {'Cache-Control': 'no-store'}
        if request.url.path == f'{case_path}/image':
            return httpx.Response(200, content=state['image'], headers=private)
        if request.url.path == f'{case_path}/report':
            return httpx.Response(200, json=detail, headers={**private, 'Content-Disposition': 'attachment; filename=fixture'})
        assert request.url.path == case_path
        return httpx.Response(200, json=detail, headers=private)

    witness = tmp_path / 'private-witness.json'
    with httpx.Client(base_url='http://127.0.0.1:18701', transport=httpx.MockTransport(transport)) as client:
        reproduced, hashes = verifier.case_snapshots(client, identifier, original)
        assert reproduced == detail
        verifier.write_json(witness, {'case_id': identifier, **hashes})
        assert set(json.loads(witness.read_bytes())) == {'case_id', *verifier.SNAPSHOTS}
        verifier.replay_case(client, witness)
        state['image'] = original + b'changed software fixture'
        with pytest.raises(verifier.CheckFailed, match='changed across the declared restart'):
            verifier.replay_case(client, witness)
