"""Persist screening results and reviewer assessment history."""

from alembic import op
import sqlalchemy as sa


revision = '0003_portal_cases'
down_revision = '0002_portal_auth'
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        'portal_case_screenings',
        sa.Column('case_id', sa.Uuid(), sa.ForeignKey('portal_cases.id'), primary_key=True),
        sa.Column('prediction', sa.JSON(), nullable=False),
        sa.Column('effective_action', sa.String(32), nullable=False),
        sa.Column('review_count', sa.Integer(), server_default='0', nullable=False),
    )
    op.create_table(
        'portal_case_reviews',
        sa.Column('id', sa.Uuid(), primary_key=True),
        sa.Column('case_id', sa.Uuid(), sa.ForeignKey('portal_case_screenings.case_id'), nullable=False),
        sa.Column('reviewer_id', sa.Uuid(), sa.ForeignKey('portal_users.id'), nullable=False),
        sa.Column('revision', sa.Integer(), nullable=False),
        sa.Column('grade', sa.Integer(), nullable=False),
        sa.Column('action', sa.String(32)),
        sa.Column('notes', sa.Text(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.UniqueConstraint('case_id', 'revision', name='uq_portal_case_review_revision'),
        sa.CheckConstraint('grade BETWEEN 0 AND 4', name='ck_portal_case_review_grade'),
    )


def downgrade():
    op.drop_table('portal_case_reviews')
    op.drop_table('portal_case_screenings')
