"""Exercise optional portal configuration and PostgreSQL URL validation."""

import pytest

pytest.importorskip('sqlalchemy')

from services.api.portal_db import DatabaseSettings


@pytest.mark.parametrize('value', [None, '', '  '])
def test_portal_database_is_optional(monkeypatch, value):
    monkeypatch.delenv('QSART_PORTAL_DATABASE_URL', raising=False)
    if value is not None:
        monkeypatch.setenv('QSART_PORTAL_DATABASE_URL', value)
    assert DatabaseSettings.from_environment() is None


@pytest.mark.parametrize('driver', ['postgresql', 'postgresql+psycopg'])
def test_postgres_url_keeps_encoded_credentials_private(monkeypatch, driver):
    monkeypatch.setenv('QSART_PORTAL_DATABASE_URL', f'{driver}://portal:fixture%25%40%3A@db:5432/qsart_portal')
    settings = DatabaseSettings.from_environment()
    assert settings.url.drivername == 'postgresql+psycopg'
    assert settings.url.password == 'fixture%@:'
    assert settings.url.database == 'qsart_portal'
    assert 'fixture' not in repr(settings)


@pytest.mark.parametrize('url', ['malformed://', 'sqlite:///portal.db', 'postgresql://portal@db'])
def test_portal_url_requires_postgresql_and_explicit_database(url):
    with pytest.raises(ValueError, match='QSART_PORTAL_DATABASE_URL'):
        DatabaseSettings.from_url(url)
