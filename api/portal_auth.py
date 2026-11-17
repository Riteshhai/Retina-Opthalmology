"""Local portal login and database-backed browser sessions."""

from datetime import datetime, timedelta, timezone
import hashlib
import os
import re
import secrets
from threading import BoundedSemaphore

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.routing import APIRoute
from pydantic import BaseModel, ConfigDict, Field, SecretStr
from sqlalchemy import delete, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import load_only, undefer

from services.api.portal_db import PortalDatabase
from services.api.portal_db.accounts import MAX_PASSWORD_LENGTH, ROLES, verify_password
from services.api.portal_db.models import PortalSession, PortalUser


COOKIE_NAME = 'qsart_portal_session'
PORTAL_PATH = '/api/portal'
SESSION_LIFETIME = timedelta(hours=8)
LOCKOUT_PERIOD = timedelta(minutes=5)
MAX_FAILED_ATTEMPTS = 8
_LOGIN_SLOTS = BoundedSemaphore(2)
_TOKEN = re.compile(r'[A-Za-z0-9_-]{43}')
_PRIVATE = {'Cache-Control': 'no-store'}


class LoginRequest(BaseModel):
    model_config = ConfigDict(extra='forbid', hide_input_in_errors=True)

    username: str = Field(min_length=1, max_length=96)
    password: SecretStr = Field(min_length=1, max_length=MAX_PASSWORD_LENGTH)


class SessionIdentity(BaseModel):
    username: str
    role: str


def _invalid_login() -> HTTPException:
    return HTTPException(422, 'Enter a valid username and password.', headers=_PRIVATE)


class _AuthRoute(APIRoute):
    def get_route_handler(self):
        handler = super().get_route_handler()

        async def handle(request: Request) -> Response:
            try:
                return await handler(request)
            except RequestValidationError:
                # The default validation response includes submitted values.
                raise _invalid_login() from None

        return handle


router = APIRouter(prefix=PORTAL_PATH, tags=['Portal authentication'], route_class=_AuthRoute)


def _unavailable() -> HTTPException:
    return HTTPException(503, 'Portal database unavailable.', headers=_PRIVATE)


def _database(request: Request) -> PortalDatabase:
    database = getattr(request.app.state, 'portal_db', None)
    if database is None:
        raise _unavailable()
    return database


def _secure_cookie(request: Request) -> bool:
    return request.url.scheme == 'https' or os.environ.get('QSART_PORTAL_SECURE_COOKIE') == '1'


def _token_digest(token: str) -> str:
    return hashlib.sha256(token.encode('ascii')).hexdigest()


def _request_digest(request: Request) -> str | None:
    token = request.cookies.get(COOKIE_NAME, '')
    return _token_digest(token) if _TOKEN.fullmatch(token) else None


def require_workspace(request: Request) -> None:
    if request.headers.get('X-QSART-Workspace') != '1':
        raise HTTPException(403, 'Send portal changes from the local workspace.', headers=_PRIVATE)


def current_user(request: Request) -> PortalUser:
    """Return the active user's identity and role with no live database connection."""
    database = _database(request)
    digest = _request_digest(request)
    if digest is not None:
        try:
            with database.session() as session:
                user = session.scalar(select(PortalUser).options(load_only(
                    PortalUser.id, PortalUser.username, PortalUser.role_code, PortalUser.is_active,
                )).join(PortalSession, PortalSession.user_id == PortalUser.id).where(
                    PortalSession.token_digest == digest,
                    PortalSession.expires_at > datetime.now(timezone.utc),
                    PortalUser.is_active.is_(True),
                    PortalUser.password_hash.is_not(None),
                    PortalUser.role_code.in_(ROLES),
                ))
                if user is not None:
                    return user
        except SQLAlchemyError:
            raise _unavailable() from None
    raise HTTPException(401, 'Sign in to the research portal.', headers=_PRIVATE)


def _start_session(database: PortalDatabase, username: str, password: str, previous: str | None):
    now = datetime.now(timezone.utc)
    try:
        with database.session() as session:
            row = session.execute(select(PortalUser, PortalUser.locked_until > now).options(
                undefer(PortalUser.password_hash),
            ).where(PortalUser.username == username).with_for_update()).one_or_none()
            user, locked = row if row is not None else (None, False)
            password_hash = user.password_hash if user is not None and user.is_active and not locked else None
            valid = verify_password(password, password_hash)
            if user is None or locked or not user.is_active or user.role_code not in ROLES or not valid:
                if user is not None and user.is_active and not locked and password_hash is not None:
                    user.failed_attempts += 1
                    if user.failed_attempts >= MAX_FAILED_ATTEMPTS:
                        user.failed_attempts = 0
                        user.locked_until = now + LOCKOUT_PERIOD
                    session.commit()
                raise HTTPException(401, 'Username or password was rejected.', headers=_PRIVATE)

            user.failed_attempts = 0
            user.locked_until = None
            session.execute(delete(PortalSession).where(PortalSession.expires_at <= now))
            if previous is not None:
                session.execute(delete(PortalSession).where(PortalSession.token_digest == previous))
            token = secrets.token_urlsafe(32)
            session.add(PortalSession(token_digest=_token_digest(token), user_id=user.id,
                                      expires_at=now + SESSION_LIFETIME))
            identity = SessionIdentity(username=user.username, role=user.role_code)
            session.commit()
            return token, identity
    except SQLAlchemyError:
        raise _unavailable() from None


@router.post('/login', response_model=SessionIdentity, dependencies=[Depends(require_workspace)])
def login(body: LoginRequest, request: Request, response: Response) -> SessionIdentity:
    password = body.password.get_secret_value()
    try:
        # JSON escapes can represent Unicode code points that UTF-8 cannot encode.
        body.username.encode('utf-8')
        password.encode('utf-8')
    except UnicodeEncodeError:
        raise _invalid_login() from None
    if '\0' in body.username:
        raise _invalid_login()
    database = _database(request)
    if not _LOGIN_SLOTS.acquire(blocking=False):
        raise HTTPException(429, 'Too many sign-ins in progress. Try again shortly.',
                            headers={**_PRIVATE, 'Retry-After': '1'})
    try:
        token, identity = _start_session(database, body.username, password, _request_digest(request))
    finally:
        _LOGIN_SLOTS.release()
    response.headers.update(_PRIVATE)
    response.set_cookie(COOKIE_NAME, token, max_age=int(SESSION_LIFETIME.total_seconds()),
                        path=PORTAL_PATH, httponly=True, samesite='strict', secure=_secure_cookie(request))
    return identity


@router.get('/session', response_model=SessionIdentity)
def session_identity(response: Response, user: PortalUser = Depends(current_user)) -> SessionIdentity:
    response.headers.update(_PRIVATE)
    return SessionIdentity(username=user.username, role=user.role_code)


@router.post('/logout', status_code=204, dependencies=[Depends(require_workspace)])
def logout(request: Request) -> Response:
    database = _database(request)
    digest = _request_digest(request)
    if digest is not None:
        try:
            with database.session() as session:
                session.execute(delete(PortalSession).where(PortalSession.token_digest == digest))
                session.commit()
        except SQLAlchemyError:
            raise _unavailable() from None
    response = Response(status_code=204, headers=_PRIVATE)
    response.delete_cookie(COOKIE_NAME, path=PORTAL_PATH, httponly=True, samesite='strict',
                           secure=_secure_cookie(request))
    return response
