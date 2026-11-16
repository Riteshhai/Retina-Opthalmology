"""Run migrations on the explicitly configured portal database."""

from alembic import context
from sqlalchemy import create_engine, pool

from services.api.portal_db.database import CONNECT_TIMEOUT_SECONDS, DATABASE_ENV, VERSION_TABLE, DatabaseSettings
from services.api.portal_db.models import Base


settings = DatabaseSettings.from_environment()
if settings is None:
    raise RuntimeError(f'{DATABASE_ENV} is required for portal migrations')

options = {'target_metadata': Base.metadata, 'version_table': VERSION_TABLE}
if context.is_offline_mode():
    context.configure(url=settings.url, literal_binds=True, **options)
    with context.begin_transaction():
        context.run_migrations()
else:
    engine = create_engine(settings.url, poolclass=pool.NullPool,
                           connect_args={'connect_timeout': CONNECT_TIMEOUT_SECONDS}, hide_parameters=True)
    try:
        with engine.connect() as connection:
            context.configure(connection=connection, **options)
            with context.begin_transaction():
                context.run_migrations()
    finally:
        engine.dispose()
