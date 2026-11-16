"""Exercise case authorization and persistence with a synthetic software fixture."""

from datetime import datetime, timezone
from io import BytesIO
from types import SimpleNamespace
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.testclient import TestClient
from PIL import Image, PngImagePlugin
import pytest

pytest.importorskip('sqlalchemy')

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from api.portal_auth import current_user
from api.portal_cases import case_router
from api.schemas import Prediction
from services.api.portal_db.models import Base, PortalCase, PortalRole, PortalUser


PORTAL = '/api/portal/cases'
WORKSPACE = {'X-QSART-Workspace': '1'}
OPERATOR = 'research_operator'
REVIEWER = 'clinician_reviewer'
VIEWER = 'research_viewer'
ADMIN = 'research_admin'


@pytest.fixture
def portal(tmp_path, monkeypatch):
    engine = create_engine('sqlite://', poolclass=StaticPool, connect_args={'check_same_thread': False})
    Base.metadata.create_all(engine)
    sessions = sessionmaker(engine, expire_on_commit=False)
    users = {role: PortalUser(username=role, role_code=role) for role in (OPERATOR, REVIEWER, VIEWER, ADMIN)}
    users['other_operator'] = PortalUser(username='other_operator', role_code=OPERATOR)
    with sessions.begin() as session:
        session.add_all(PortalRole(code=role, label=role) for role in (OPERATOR, REVIEWER, VIEWER, ADMIN))
        session.add_all(users.values())

    def software_actor(request: Request):
        user = users.get(request.headers.get('X-Software-Fixture-Actor'))
        if user is None:
            raise HTTPException(401)
        return user

    async def software_screen(file):
        with BytesIO(await file.read()) as stream, Image.open(stream) as source:
            image = source.convert('RGB')
        await file.close()
        return Prediction(
            request_id=str(uuid4()), image_index=0, model_version='synthetic-case-test', predicted_grade=3,
            grade_probabilities=[.1, .1, .1, .6, .1], referable_probability=.8, severe_probability=.7,
            quality={'probabilities': [.1, .9], 'predicted_class': 1, 'poor_probability': .1},
            uncertainty={'entropy': .2, 'confidence': .6}, action='SPECIALIST_REVIEW', reason_codes=['FIXTURE'],
            second_pass_activated=False, processing_time_ms=5, inference_timestamp=datetime.now(timezone.utc),
            policy_scope='single_image',
        ), image

    image_dir = tmp_path / 'private'
    image_dir.mkdir()
    monkeypatch.setenv('QSART_PORTAL_IMAGE_DIR', str(image_dir))
    app = FastAPI()
    app.state.portal_db = SimpleNamespace(session=sessions)
    app.dependency_overrides[current_user] = software_actor
    app.include_router(case_router(software_screen))
    with TestClient(app) as client:
        yield client, image_dir
    engine.dispose()


def auth(role):
    return {'X-Software-Fixture-Actor': role}


def upload(client, role, headers=WORKSPACE):
    out = BytesIO()
    metadata = PngImagePlugin.PngInfo()
    metadata.add_text('original_dicom_id', 'SYNTHETIC_METADATA')
    Image.new('RGB', (32, 32), 'red').save(out, format='PNG', pnginfo=metadata)
    return client.post(PORTAL, files={'file': ('synthetic.png', out.getvalue(), 'image/png')},
                       headers={**auth(role), **headers})


def test_case_scope_and_private_image(portal):
    client, image_dir = portal
    created = upload(client, OPERATOR)
    assert created.status_code == 201, created.text
    identifier = created.json()['id']
    assert [image.name for image in image_dir.iterdir()] == [f'{identifier}.png']
    assert client.get(PORTAL).status_code == 401
    assert client.get(PORTAL, headers=auth('other_operator')).json() == []
    case_path = f'{PORTAL}/{identifier}'
    for path in [case_path, f'{case_path}/image', f'{case_path}/report']:
        assert client.get(path, headers=auth('other_operator')).status_code == 404
    assert [case['id'] for case in client.get(PORTAL, headers=auth(VIEWER)).json()] == [identifier]
    assert client.get(case_path, headers=auth(VIEWER)).json() == created.json()
    raw_image = client.get(f'{case_path}/image', headers=auth(REVIEWER))
    assert raw_image.headers['cache-control'] == 'no-store'
    assert b'SYNTHETIC_METADATA' not in raw_image.content
    with Image.open(BytesIO(raw_image.content)) as image:
        assert image.getpixel((0, 0)) == (255, 0, 0)
    assert client.get(f'{case_path}/report', headers=auth(VIEWER)).status_code == 403


def test_role_and_origin_guard_before_case_writes(portal):
    client, image_dir = portal
    assert upload(client, '', WORKSPACE).status_code == 401
    assert upload(client, OPERATOR, {}).status_code == 403
    assert upload(client, VIEWER).status_code == 403
    assert upload(client, REVIEWER).status_code == 403
    assert list(image_dir.iterdir()) == []


def test_case_overrides_append_history_and_preserve_model_output(portal):
    client, _ = portal
    original = upload(client, OPERATOR).json()
    pending = upload(client, ADMIN).json()
    endpoint = f'{PORTAL}/{original["id"]}'
    review = {'grade': 2, 'action': 'RECAPTURE', 'notes': 'Obtain a second view.'}
    assert client.post(f'{endpoint}/reviews', json=review, headers={**auth(OPERATOR), **WORKSPACE}).status_code == 403
    assert client.post(f'{endpoint}/reviews', json=review, headers=auth(REVIEWER)).status_code == 403
    headers = {**auth(REVIEWER), **WORKSPACE}
    assert client.post(f'{endpoint}/reviews', json={**review, 'notes': '  '}, headers=headers).status_code == 422
    modified = client.post(f'{endpoint}/reviews', json=review, headers=headers)
    assert modified.status_code == 200, modified.text
    first = modified.json()
    assert first['prediction'] == original['prediction']
    assert first['effective_action'] == 'RECAPTURE'
    assert first['reviews'][0]['grade_disagreement'] and first['reviews'][0]['is_override']
    next_review = client.post(f'{endpoint}/reviews', json={'grade': 3}, headers=headers).json()
    assert next_review['effective_action'] == 'RECAPTURE'
    assert [row['event'] for row in next_review['audit']] == ['case_created', 'action_overridden', 'review_recorded']
    queue = client.get(PORTAL, headers=auth(ADMIN)).json()
    assert [(row['id'], row['reviewed']) for row in queue] == [(pending['id'], False), (original['id'], True)]
    assert client.get(f'{PORTAL}?limit=1&offset=1', headers=auth(ADMIN)).json() == queue[1:]
    report = client.get(f'{endpoint}/report', headers=auth(OPERATOR))
    assert report.headers['content-disposition'].startswith('attachment')
    assert report.json() == next_review


def test_failed_private_storage_rolls_back_case(portal, monkeypatch, tmp_path):
    client, image_dir = portal
    monkeypatch.setenv('QSART_PORTAL_IMAGE_DIR', str(tmp_path / 'missing'))
    assert upload(client, OPERATOR).status_code == 503
    assert client.get(PORTAL, headers=auth(ADMIN)).json() == []
    assert list(image_dir.iterdir()) == []


def test_legacy_case_stub_is_preserved_but_excluded(portal):
    client, _ = portal
    database = client.app.state.portal_db
    with database.session.begin() as session:
        operator = session.query(PortalUser).filter_by(username=OPERATOR).one()
        legacy = PortalCase(created_by=operator.id)
        session.add(legacy)
        session.flush()
        identifier = legacy.id
    assert client.get(PORTAL, headers=auth(ADMIN)).json() == []
    assert client.get(f'{PORTAL}/{identifier}', headers=auth(ADMIN)).status_code == 404
