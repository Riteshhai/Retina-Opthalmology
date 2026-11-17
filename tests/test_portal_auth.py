"""Exercise login and revocation with test-only accounts."""

from datetime import datetime, timedelta, timezone
import hashlib

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest


pytest.importorskip('sqlalchemy')
from sqlalchemy import create_engine, select, update
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from api import portal_auth
from services.api.portal_db.accounts import create_account, update_account
from services.api.portal_db.models import Base, PortalRole, PortalSession, PortalUser


WRITE = {'X-QSART-Workspace': '1'}
PASSWORD = 'test-only password'
USERNAME = 'test-operator'
LOGIN = {'username': USERNAME, 'password': PASSWORD}
PREFIX = '/api/portal'


@pytest.fixture
def portal():
    engine = create_engine('sqlite://', poolclass=StaticPool, connect_args={'check_same_thread': False})
    Base.metadata.create_all(engine, tables=[PortalRole.__table__, PortalUser.__table__, PortalSession.__table__])
    sessions = sessionmaker(engine)

    class TestDatabase:
        session = staticmethod(sessions)

    database = TestDatabase()
    with database.session() as session:
        session.add(PortalRole(code='research_operator', label='Research operator'))
        session.commit()
    create_account(database, USERNAME, 'research_operator', PASSWORD)

    app = FastAPI()
    app.state.portal_db = database
    app.include_router(portal_auth.router)
    try:
        with TestClient(app) as client:
            yield client, database
    finally:
        engine.dispose()


def test_login_sets_private_cookie_and_logout_revokes_server_session(portal):
    client, database = portal
    assert client.get(f'{PREFIX}/session').status_code == 401
    assert client.post(f'{PREFIX}/login', json=LOGIN).status_code == 403
    logged_in = client.post(f'{PREFIX}/login', json=LOGIN, headers=WRITE)
    expected = {'username': USERNAME, 'role': 'research_operator'}
    assert logged_in.status_code == 200
    assert logged_in.json() == expected
    assert logged_in.headers['cache-control'] == 'no-store'
    cookie_header = logged_in.headers['set-cookie'].lower()
    assert all(flag in cookie_header for flag in ('httponly', 'samesite=strict', 'path=/api/portal'))
    token = client.cookies[portal_auth.COOKIE_NAME]
    with database.session() as session:
        assert session.scalar(select(PortalSession.token_digest)) == hashlib.sha256(token.encode()).hexdigest()
        stored = session.scalar(select(PortalUser.password_hash).where(PortalUser.username == USERNAME))
        assert stored.startswith('scrypt-') and PASSWORD not in stored
        assert len(stored) <= PortalUser.__table__.c.password_hash.type.length
    assert client.get(f'{PREFIX}/session').json() == expected
    assert client.post(f'{PREFIX}/logout').status_code == 403
    assert client.post(f'{PREFIX}/logout', headers=WRITE).status_code == 204
    assert client.get(f'{PREFIX}/session', headers={'Cookie': f'{portal_auth.COOKIE_NAME}={token}'}).status_code == 401


def test_invalid_login_bodies_are_redacted(portal):
    client, _ = portal
    marker = 'software-fixture-secret'
    for payload in (
        f'{{"password":"{marker}"',
        f'{{"password":"{marker}","unexpected":"{marker}"}}',
        f'{{"username":"test-operator","password":"{marker * 30}"}}',
    ):
        response = client.post(f'{PREFIX}/login', content=payload,
                               headers={**WRITE, 'Content-Type': 'application/json'})
        assert response.status_code == 422
        assert marker not in response.text
        assert response.json() == {'detail': 'Enter a valid username and password.'}


def test_login_lockout_and_credentials_rotation_revoke_access(portal, monkeypatch):
    client, database = portal
    monkeypatch.setattr(portal_auth, 'MAX_FAILED_ATTEMPTS', 2)
    logged_in = client.post(f'{PREFIX}/login', json=LOGIN, headers=WRITE)
    assert logged_in.status_code == 200
    for _ in range(2):
        wrong = client.post(f'{PREFIX}/login', json={**LOGIN, 'password': 'wrong'}, headers=WRITE)
        assert wrong.status_code == 401
    blocked = client.post(f'{PREFIX}/login', json=LOGIN, headers=WRITE)
    assert blocked.status_code == 401
    assert blocked.json() == wrong.json()

    new_password = 'new test-only password'
    update_account(database, USERNAME, password=new_password)
    assert client.get(f'{PREFIX}/session').status_code == 401
    login = {**LOGIN, 'password': new_password}
    assert client.post(f'{PREFIX}/login', json=login, headers=WRITE).status_code == 200
    update_account(database, USERNAME)
    assert client.get(f'{PREFIX}/session').status_code == 401
    assert client.post(f'{PREFIX}/login', json=login, headers=WRITE).status_code == 401


def test_legacy_account_requires_password_and_expired_session_is_refused(portal):
    client, database = portal
    with database.session() as session:
        session.add(PortalUser(username='legacy-operator', role_code='research_operator'))
        session.commit()
    body = {**LOGIN, 'username': 'legacy-operator'}
    assert client.post(f'{PREFIX}/login', json=body, headers=WRITE).status_code == 401
    update_account(database, 'legacy-operator', password=PASSWORD)
    assert client.post(f'{PREFIX}/login', json=body, headers=WRITE).status_code == 200
    with database.session() as session:
        session.execute(update(PortalSession).values(expires_at=datetime.now(timezone.utc) - timedelta(seconds=1)))
        session.commit()
    assert client.get(f'{PREFIX}/session').status_code == 401


def test_portal_without_database_reports_unavailable():
    app = FastAPI()
    app.include_router(portal_auth.router)
    with TestClient(app) as client:
        assert client.get(f'{PREFIX}/session').status_code == 503
        assert client.post(f'{PREFIX}/login', json=LOGIN, headers=WRITE).status_code == 503
