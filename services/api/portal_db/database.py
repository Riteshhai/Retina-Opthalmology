"""Explicit settings, sessions and readiness for the portal database."""

from dataclasses import dataclass, field
import os

from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL, make_url
from sqlalchemy.exc import ArgumentError, SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker


DATABASE_ENV = 'QSART_PORTAL_DATABASE_URL'
VERSION_TABLE = 'qsart_portal_alembic_version'
HEAD_REVISION = '0001_portal_foundation'
CONNECT_TIMEOUT_SECONDS = 3


@dataclass(frozen=True)
class DatabaseSettings:
    url: URL = field(repr=False)

    @classmethod
    def from_url(cls, value: str) -> 'DatabaseSettings':
        try:
            url = make_url(value.strip())
        except (ArgumentError, ValueError):
            raise ValueError(f'{DATABASE_ENV} must be a valid PostgreSQL URL') from None
        if url.drivername not in {'postgresql', 'postgresql+psycopg'} or not url.database:
            raise ValueError(f'{DATABASE_ENV} must identify a PostgreSQL database using psycopg')
        return cls(url.set(drivername='postgresql+psycopg'))

    @classmethod
    def from_environment(cls) -> 'DatabaseSettings | None':
        value = os.environ.get(DATABASE_ENV, '').strip()
        return cls.from_url(value) if value else None


class PortalDatabase:
    def __init__(self, settings: DatabaseSettings):
        self._engine = create_engine(
            settings.url,
            connect_args={'connect_timeout': CONNECT_TIMEOUT_SECONDS},
            hide_parameters=True,
            pool_pre_ping=True,
            pool_timeout=CONNECT_TIMEOUT_SECONDS,
        )
        self._sessions = sessionmaker(self._engine)

    def session(self) -> Session:
        """Open a session. The caller owns its transaction and lifetime."""
        return self._sessions()

    def is_ready(self) -> bool:
        """Confirm that PostgreSQL responds and has the expected portal schema."""
        try:
            with self._engine.connect() as connection:
                connection.execute(text('SET LOCAL statement_timeout = 3000'))
                versions = connection.execute(text(f'SELECT version_num FROM {VERSION_TABLE}')).scalars().all()
                return versions == [HEAD_REVISION]
        except SQLAlchemyError:
            return False

    def dispose(self) -> None:
        self._engine.dispose()
