"""Authenticated screening, review and private case reports."""

from collections.abc import Awaitable, Callable
from contextlib import contextmanager
from datetime import datetime, timezone
import logging
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Annotated, Literal
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, File, HTTPException, Query, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, Response
from PIL import Image
from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, StringConstraints, model_validator
from sqlalchemy import and_, case, select
from sqlalchemy.exc import SQLAlchemyError

from api.portal_auth import current_user, require_workspace
from api.schemas import DISCLAIMER, Prediction
from qsart.triage_v2 import Action
from services.api.portal_db import PortalDatabase
from services.api.portal_db.cases import CaseReview, CaseScreening
from services.api.portal_db.models import PortalCase, PortalUser


logger = logging.getLogger('qsart.portal.cases')
AUTHORS = {'research_admin', 'research_operator'}
REVIEWERS = {'research_admin', 'clinician_reviewer'}
PRIVATE_HEADERS = {'Cache-Control': 'no-store', 'X-Content-Type-Options': 'nosniff'}
ScreenImage = Callable[[UploadFile], Awaitable[tuple[Prediction, Image.Image]]]


class ReviewInput(BaseModel):
    model_config = ConfigDict(extra='forbid')
    grade: int = Field(strict=True, ge=0, le=4)
    action: Action | None = None
    notes: Annotated[str, StringConstraints(strip_whitespace=True, max_length=1500)] = ''

    @model_validator(mode='after')
    def require_override_reason(self):
        if self.action is not None and not self.notes:
            raise ValueError('Provide a reason for the action override')
        return self


class RecordedReview(ReviewInput):
    id: UUID
    reviewer: str
    created_at: AwareDatetime
    grade_disagreement: bool
    is_override: bool


class AuditEvent(BaseModel):
    event: Literal['case_created', 'review_recorded', 'action_overridden']
    actor: str
    created_at: AwareDatetime


class CaseSummary(BaseModel):
    id: UUID
    created_at: AwareDatetime
    predicted_grade: int = Field(ge=0, le=4)
    severe_probability: float = Field(ge=0, le=1)
    model_action: Action
    effective_action: Action
    reviewed: bool


class CaseDetail(BaseModel):
    id: UUID
    created_at: AwareDatetime
    prediction: Prediction
    effective_action: Action
    reviews: list[RecordedReview]
    audit: list[AuditEvent]
    disclaimer: str = DISCLAIMER
    review_limitations: str = 'Reviews document user-entered assessments for this research workflow.'


def _timestamp(value: datetime) -> datetime:
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value


def _database(request: Request) -> PortalDatabase:
    database = request.app.state.portal_db
    if database is None:
        raise HTTPException(503, 'Portal database is unavailable')
    return database


@contextmanager
def _case_session(database):
    try:
        with database.session() as session:
            yield session
    except SQLAlchemyError as exc:
        logger.warning('Portal case database unavailable: %s', type(exc).__name__)
        raise HTTPException(503, 'Portal case storage is unavailable') from None


def _case_rows(user):
    columns = select(PortalCase, CaseScreening, PortalUser.username).join(CaseScreening).join(
        PortalUser, PortalCase.created_by == PortalUser.id,
    )
    return columns.where(PortalCase.created_by == user.id) if user.role_code == 'research_operator' else columns


def _lookup(session, user, case_id, *, lock: Literal['read', 'write'] | None = None):
    query = _case_rows(user).where(PortalCase.id == case_id)
    if lock is not None:
        query = query.with_for_update(of=CaseScreening, read=lock == 'read')
    row = session.execute(query).one_or_none()
    if row is None:
        raise HTTPException(404, 'Case not found')
    return row


def _detail(session, row) -> CaseDetail:
    metadata, screening, creator = row
    grades = Prediction.model_validate(screening.prediction)
    entries = session.execute(select(CaseReview, PortalUser.username).join(PortalUser).where(
        CaseReview.case_id == metadata.id,
    ).order_by(CaseReview.revision))
    reviews = [RecordedReview(
        id=review.id, reviewer=actor, grade=review.grade, action=review.action, notes=review.notes,
        created_at=_timestamp(review.created_at), grade_disagreement=review.grade != grades.predicted_grade,
        is_override=review.action is not None,
    ) for review, actor in entries]
    created_at = _timestamp(metadata.created_at)
    audit = [AuditEvent(event='case_created', actor=creator, created_at=created_at)]
    audit.extend(AuditEvent(
        event='action_overridden' if review.is_override else 'review_recorded',
        actor=review.reviewer, created_at=review.created_at,
    ) for review in reviews)
    return CaseDetail(id=metadata.id, created_at=created_at, prediction=grades,
                      effective_action=screening.effective_action, reviews=reviews, audit=audit)


def _image_path(case_id: UUID) -> Path:
    directory = os.environ.get('QSART_PORTAL_IMAGE_DIR', '').strip()
    if not directory or not Path(directory).is_absolute():
        raise HTTPException(503, 'Private image storage is unavailable')
    return Path(directory) / f'{case_id}.png'


def _store_image(path: Path, image: Image.Image) -> None:
    temporary = None
    try:
        with NamedTemporaryFile(dir=path.parent, prefix='.upload-', delete=False) as file:
            temporary = Path(file.name)
            image.info.clear()
            image.save(file, format='PNG')
            file.flush()
            os.fsync(file.fileno())
        temporary.replace(path)
    except (OSError, ValueError, TypeError) as exc:
        logger.warning('Private case image could not be stored: %s', type(exc).__name__)
        raise HTTPException(503, 'Private image storage is unavailable') from None
    finally:
        if temporary is not None:
            _discard_image(temporary)


def _discard_image(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError as exc:
        logger.warning('Private case image cleanup failed: %s', type(exc).__name__)


def _store_case(database, user, prediction, image) -> CaseDetail:
    case_id = uuid4()
    path = _image_path(case_id)
    with _case_session(database) as session:
        metadata = PortalCase(id=case_id, created_by=user.id)
        session.add(metadata)
        session.flush()
        screening = CaseScreening(case_id=case_id, prediction=prediction.model_dump(mode='json'),
                                  effective_action=prediction.action.value)
        session.add(screening)
        session.flush()
        detail = _detail(session, (metadata, screening, user.username))
        _store_image(path, image)
        try:
            session.commit()
        except SQLAlchemyError:
            # A lost commit acknowledgement can leave a committed case. Check before removing its image.
            try:
                session.rollback()
                if session.get(CaseScreening, case_id) is None:
                    _discard_image(path)
            except SQLAlchemyError:
                logger.warning('Could not confirm the status of a private case commit')
            raise
        return detail


def case_router(screen: ScreenImage) -> APIRouter:
    router = APIRouter(prefix='/api/portal', tags=['Portal cases'])
    authenticated = Annotated[PortalUser, Depends(current_user)]
    write_checks = [Depends(require_workspace)]

    @router.post('/cases', response_model=CaseDetail, status_code=201, dependencies=write_checks)
    async def new_case(request: Request, user: authenticated, file: UploadFile = File(...)):
        if user.role_code not in AUTHORS:
            raise HTTPException(403, 'Case creation requires a research operator or administrator')
        database = _database(request)
        prediction, image = await screen(file)
        try:
            return await run_in_threadpool(_store_case, database, user, prediction, image)
        finally:
            image.close()

    @router.get('/cases', response_model=list[CaseSummary])
    def case_list(request: Request, user: authenticated,
                  limit: Annotated[int | None, Query(ge=1, le=500)] = None,
                  offset: Annotated[int, Query(ge=0)] = 0):
        prediction = CaseScreening.prediction
        reviewed = CaseScreening.review_count > 0
        severity = prediction['severe_probability'].as_float()
        model_action = prediction['action'].as_string()
        queue = case(
            (and_(~reviewed, model_action == Action.SPECIALIST_REVIEW.value), 0),
            (and_(~reviewed, model_action == Action.RECAPTURE.value), 1),
            (~reviewed, 2),
            else_=3,
        )
        query = select(PortalCase.id, PortalCase.created_at, prediction['predicted_grade'].as_integer(),
                       severity, model_action, CaseScreening.effective_action, reviewed).join(CaseScreening)
        if user.role_code == 'research_operator':
            query = query.where(PortalCase.created_by == user.id)
        query = query.order_by(queue, severity.desc(), PortalCase.created_at, PortalCase.id).offset(offset)
        if limit is not None:
            query = query.limit(limit)
        with _case_session(_database(request)) as session:
            return [CaseSummary(id=identifier, created_at=_timestamp(created), predicted_grade=grade,
                                severe_probability=risk, model_action=model, effective_action=effective,
                                reviewed=is_reviewed) for identifier, created, grade, risk, model, effective, is_reviewed
                    in session.execute(query)]

    @router.get('/cases/{case_id}', response_model=CaseDetail)
    def case_detail(request: Request, case_id: UUID, user: authenticated):
        with _case_session(_database(request)) as session:
            return _detail(session, _lookup(session, user, case_id, lock='read'))

    @router.get('/cases/{case_id}/image', response_class=FileResponse,
                responses={200: {'content': {'image/png': {}}}})
    def case_image(request: Request, case_id: UUID, user: authenticated):
        with _case_session(_database(request)) as session:
            _lookup(session, user, case_id)
        path = _image_path(case_id)
        try:
            image_exists = path.is_file()
        except OSError:
            image_exists = False
        if not image_exists:
            raise HTTPException(404, 'Case image is unavailable')
        return FileResponse(path, media_type='image/png', headers=PRIVATE_HEADERS)

    @router.post('/cases/{case_id}/reviews', response_model=CaseDetail, dependencies=write_checks)
    def record_review(request: Request, case_id: UUID, user: authenticated, assessment: ReviewInput):
        if user.role_code not in REVIEWERS:
            raise HTTPException(403, 'Review requires a clinician reviewer or administrator')
        with _case_session(_database(request)) as session:
            row = _lookup(session, user, case_id, lock='write')
            screening = row[1]
            screening.review_count += 1
            if assessment.action is not None:
                screening.effective_action = assessment.action.value
            session.add(CaseReview(case_id=case_id, reviewer_id=user.id, revision=screening.review_count,
                                   **assessment.model_dump(mode='json')))
            session.flush()
            detail = _detail(session, row)
            session.commit()
            return detail

    @router.get('/cases/{case_id}/report', response_model=CaseDetail)
    def case_report(request: Request, case_id: UUID, user: authenticated):
        if user.role_code == 'research_viewer':
            raise HTTPException(403, 'This role cannot export a case report')
        with _case_session(_database(request)) as session:
            detail = _detail(session, _lookup(session, user, case_id, lock='read'))
        return Response(detail.model_dump_json(indent=2), media_type='application/json', headers={
            **PRIVATE_HEADERS, 'Content-Disposition': f'attachment; filename="qsart-case-{case_id}.json"',
        })

    return router
