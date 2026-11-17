"""Add password credentials and revocable local portal sessions."""

from alembic import op
import sqlalchemy as sa


revision = '0002_portal_auth'
down_revision = '0001_portal_foundation'
branch_labels = None
depends_on = None


def upgrade():
    op.add_column('portal_users', sa.Column('password_hash', sa.String(160)))
    op.add_column('portal_users', sa.Column('failed_attempts', sa.Integer(), server_default='0', nullable=False))
    op.add_column('portal_users', sa.Column('locked_until', sa.DateTime(timezone=True)))
    op.create_table(
        'portal_sessions',
        sa.Column('token_digest', sa.String(64), primary_key=True),
        sa.Column('user_id', sa.Uuid(), sa.ForeignKey('portal_users.id', ondelete='CASCADE'), nullable=False),
        sa.Column('expires_at', sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index('ix_portal_sessions_user_id', 'portal_sessions', ['user_id'])
    op.create_index('ix_portal_sessions_expires_at', 'portal_sessions', ['expires_at'])


def downgrade():
    op.drop_table('portal_sessions')
    for column in ('locked_until', 'failed_attempts', 'password_hash'):
        op.drop_column('portal_users', column)
