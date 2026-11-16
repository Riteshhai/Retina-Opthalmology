"""Private screening results and append-only reviewer assessments."""

from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import JSON, CheckConstraint, DateTime, ForeignKey, String, Text, UniqueConstraint, Uuid, func
from sqlalchemy.orm import Mapped, mapped_column

from .models import Base


class CaseScreening(Base):
    __tablename__ = 'portal_case_screenings'

    case_id: Mapped[UUID] = mapped_column(Uuid, ForeignKey('portal_cases.id'), primary_key=True)
    prediction: Mapped[dict] = mapped_column(JSON)
    effective_action: Mapped[str] = mapped_column(String(32))
    review_count: Mapped[int] = mapped_column(server_default='0')


class CaseReview(Base):
    __tablename__ = 'portal_case_reviews'
    __table_args__ = (
        UniqueConstraint('case_id', 'revision', name='uq_portal_case_review_revision'),
        CheckConstraint('grade BETWEEN 0 AND 4', name='ck_portal_case_review_grade'),
    )

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    case_id: Mapped[UUID] = mapped_column(Uuid, ForeignKey('portal_case_screenings.case_id'))
    reviewer_id: Mapped[UUID] = mapped_column(Uuid, ForeignKey('portal_users.id'))
    revision: Mapped[int]
    grade: Mapped[int]
    action: Mapped[str | None] = mapped_column(String(32))
    notes: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
