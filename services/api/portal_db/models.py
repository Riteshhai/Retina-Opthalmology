"""Minimal portal role, user and research case metadata models."""

from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import Boolean, DateTime, ForeignKey, String, Uuid, func, true
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class PortalRole(Base):
    __tablename__ = 'portal_roles'

    code: Mapped[str] = mapped_column(String(32), primary_key=True)
    label: Mapped[str] = mapped_column(String(64))


class PortalUser(Base):
    __tablename__ = 'portal_users'

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    username: Mapped[str] = mapped_column(String(96), unique=True)
    role_code: Mapped[str] = mapped_column(ForeignKey('portal_roles.code'))
    is_active: Mapped[bool] = mapped_column(Boolean, server_default=true())
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class PortalCase(Base):
    __tablename__ = 'portal_cases'

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    created_by: Mapped[UUID] = mapped_column(ForeignKey('portal_users.id'), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
