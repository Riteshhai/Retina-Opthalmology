"""Create research portal metadata tables and four application roles."""

from alembic import op
import sqlalchemy as sa


revision = '0001_portal_foundation'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    roles = op.create_table(
        'portal_roles',
        sa.Column('code', sa.String(32), primary_key=True),
        sa.Column('label', sa.String(64), nullable=False),
    )
    op.bulk_insert(roles, [
        {'code': 'research_admin', 'label': 'Research administrator'},
        {'code': 'research_operator', 'label': 'Research operator'},
        {'code': 'clinician_reviewer', 'label': 'Clinician/reviewer'},
        {'code': 'research_viewer', 'label': 'Read-only research viewer'},
    ])
    op.create_table(
        'portal_users',
        sa.Column('id', sa.Uuid(), primary_key=True),
        sa.Column('username', sa.String(96), nullable=False, unique=True),
        sa.Column('role_code', sa.String(32), sa.ForeignKey('portal_roles.code'), nullable=False),
        sa.Column('is_active', sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    op.create_table(
        'portal_cases',
        sa.Column('id', sa.Uuid(), primary_key=True),
        sa.Column('created_by', sa.Uuid(), sa.ForeignKey('portal_users.id'), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    op.create_index('ix_portal_cases_created_by', 'portal_cases', ['created_by'])


def downgrade():
    op.drop_table('portal_cases')
    op.drop_table('portal_users')
    op.drop_table('portal_roles')
