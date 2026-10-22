"""Temporary uploads are closed; production never substitutes generated predictions."""
from contextlib import asynccontextmanager
from io import BytesIO
import hashlib
import json
import logging
import os
from pathlib import Path
from threading import Lock
import uuid
import warnings

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError

from api.schemas import (BatchPrediction, DISCLAIMER, HealthResponse, ModelInformation,
                         Prediction, ReadyResponse)
from qsart.triage_v2 import (Action, BudgetOverloaded, PolicyConfig, allocate_rows,
                             pointwise_action)

MAX_BYTES = 12 * 1024 * 1024
MAX_BATCH = 64
MAX_PIXELS = 24_000_000
ALLOWED = {'image/png': 'PNG', 'image/jpeg': 'JPEG'}
logger = logging.getLogger('qsart.api')


def create_app(engine=None, conformal=None):
    @asynccontextmanager
    async def lifespan(app):
        if app.state.engine is None:
            try:
                from qsart.inference.engine import InferenceEngine
                app.state.engine = InferenceEngine.load(
                    os.environ.get('QSART_DR_CHECKPOINT', 'checkpoints/dr/best.pt'),
                    os.environ.get('QSART_QUALITY_CHECKPOINT', 'checkpoints/quality/best.pt'),
                    os.environ.get('QSART_CALIBRATION', 'checkpoints/calibration.json'),
                    cpu_threads=int(os.environ.get('QSART_CPU_THREADS', '2')))
                app.state.unavailable = None
            except Exception as exc:
                # Do not expose model paths, exception messages or upload content.
                app.state.unavailable = type(exc).__name__
                logger.warning('Model artifacts unavailable: %s', type(exc).__name__)
        conformal_path = os.environ.get('QSART_CONFORMAL')
        if app.state.engine is not None and app.state.conformal is None and conformal_path:
            try:
                from qsart.inference.conformal_sidecar import ConformalSidecar
                app.state.conformal = ConformalSidecar(conformal_path,
                    dr_checkpoint=os.environ.get('QSART_DR_CHECKPOINT', 'checkpoints/dr/best.pt'),
                    quality_checkpoint=os.environ.get('QSART_QUALITY_CHECKPOINT', 'checkpoints/quality/best.pt'),
                    calibration=os.environ.get('QSART_CALIBRATION', 'checkpoints/calibration.json'))
            except Exception as exc:
                app.state.engine = None
                app.state.unavailable = 'Conformal'+type(exc).__name__
                logger.warning('Configured conformal artifact unavailable: %s', type(exc).__name__)
        yield

    app = FastAPI(title='QSART Research API', version='0.3.0', lifespan=lifespan,
                  description=DISCLAIMER)
    app.state.engine = engine
    app.state.conformal = conformal
    app.state.unavailable = None if engine else 'ArtifactsNotLoaded'
    app.state.inference_lock = Lock()

    @app.get('/health', response_model=HealthResponse)
    def health():
        return {'status': 'ok', 'service': 'qsart-api'}

    @app.get('/ready', response_model=ReadyResponse)
    def ready():
        available = app.state.engine is not None
        return JSONResponse(status_code=200 if available else 503, content={
            'ready': available, 'reason': app.state.unavailable,
            'policy_version': '2.0',
            'conformal_status': 'available' if app.state.conformal is not None else 'not_configured',
            'disclaimer': DISCLAIMER})

    @app.get('/model/info', response_model=ModelInformation)
    def model_info():
        if app.state.engine is None:
            return {'ready': False, 'reason': app.state.unavailable,
                    'disclaimer': DISCLAIMER}
        e = app.state.engine
        info = e.info() if callable(getattr(e, 'info', None)) else {
            'model_version': getattr(e, 'model_version', 'unknown')}
        card_path = os.environ.get('QSART_MODEL_CARD')
        if card_path:
            try:
                card = json.loads(Path(card_path).read_text(encoding='utf-8'))
                if (card['dr_checkpoint_sha256'] != e.calibration['dr']['checkpoint_sha256']
                        or card['quality_checkpoint_sha256'] != e.calibration['quality']['checkpoint_sha256']
                        or card['calibration_content_sha256'] != hashlib.sha256(json.dumps(
                            e.calibration, sort_keys=True, separators=(',', ':')).encode()).hexdigest()):
                    raise ValueError('Model card does not match loaded checkpoints')
                info['model_card'] = card
                info['model_card_status'] = 'matched'
            except (OSError, ValueError, KeyError, AttributeError):
                info['model_card_status'] = 'unavailable_or_mismatched'
        return {'ready': True, **info, 'supported_types': list(ALLOWED),
                'policy_version': '2.0', 'action_semantics': 'provisional_nonreferable_clearance_only',
                'conformal': app.state.conformal.info() if app.state.conformal is not None
                            else {'status': 'not_configured'},
                'max_upload_bytes': MAX_BYTES, 'max_batch': MAX_BATCH,
                'disclaimer': DISCLAIMER}

    async def decode(upload):
        if upload.content_type not in ALLOWED:
            await upload.close()
            raise HTTPException(415, 'Only PNG and JPEG uploads are supported')
        data = await upload.read(MAX_BYTES+1)
        await upload.close()
        if len(data) > MAX_BYTES:
            raise HTTPException(413, 'Image exceeds the 12 MiB upload limit')
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('error', Image.DecompressionBombWarning)
                with Image.open(BytesIO(data)) as im:
                    if im.format != ALLOWED[upload.content_type]:
                        raise HTTPException(415, 'Image format does not match MIME type')
                    if min(im.size) < 32 or im.width*im.height > MAX_PIXELS:
                        raise HTTPException(422, 'Image dimensions are outside supported limits')
                    im.verify()
                with Image.open(BytesIO(data)) as im:
                    return im.convert('RGB').copy()
        except HTTPException:
            raise
        except (UnidentifiedImageError, OSError, ValueError,
                Image.DecompressionBombWarning, Image.DecompressionBombError):
            raise HTTPException(422, 'Corrupt or unreadable image') from None

    def predict(images, budget):
        if app.state.engine is None:
            raise HTTPException(503, 'Trained DR, quality and calibration artifacts are required')
        try:
            with app.state.inference_lock:
                if app.state.conformal is not None:
                    rows = app.state.engine.predict_images(images, second_pass_budget=budget,
                                                          include_uncalibrated=True)
                    return app.state.conformal.enrich(rows)
                return app.state.engine.predict_images(images, second_pass_budget=budget)
        except HTTPException:
            raise
        except Exception as exc:
            logger.error('Inference failed: %s', type(exc).__name__)
            raise HTTPException(503, 'Model inference failed; no result was produced') from None

    def response(row, request_id, index, action, reasons, scope):
        return Prediction(
            request_id=request_id, image_index=index,
            **{k: row[k] for k in ('model_version', 'predicted_grade',
                                  'grade_probabilities', 'referable_probability',
                                  'quality', 'uncertainty', 'second_pass_activated',
                                  'processing_time_ms')},
            action=action, reason_codes=reasons, policy_scope=scope,
            conformal=row.get('conformal'),
            model_warnings=['RESEARCH_NOT_CLINICALLY_VALIDATED'] + (
                ['CONFORMAL_NOT_CONFIGURED'] if app.state.conformal is None else []) + (
                ['CONDITIONAL_SECOND_PASS_COVERAGE_UNVERIFIED'] if row['second_pass_activated'] else []) + (
                ['CONFORMAL_SET_CONTAINS_ALL_GRADES'] if row.get('conformal_set') == [0, 1, 2, 3, 4] else []))

    @app.post('/predict', response_model=Prediction)
    async def single(file: UploadFile = File(...),
                     second_pass: bool = Query(False)):
        im = await decode(file)
        row = (await run_in_threadpool(predict, [im], float(second_pass)))[0]
        action, reasons = pointwise_action(row['grade_probabilities'],
            1-row['quality']['poor_probability'], row['uncertainty']['entropy'],
            conformal_set=row.get('conformal_set'))
        reasons.append('SINGLE_IMAGE_NO_POPULATION_BUDGET_GUARANTEE')
        return response(row, str(uuid.uuid4()), 0, action, reasons, 'single_image')

    @app.post('/predict/batch', response_model=BatchPrediction)
    async def batch(files: list[UploadFile] = File(...),
                    review_budget: float = Query(.3, ge=0, le=1),
                    recapture_budget: float = Query(.2, ge=0, le=1),
                    second_pass_budget: float = Query(0., ge=0, le=1)):
        if not files or len(files) > MAX_BATCH:
            for f in files:
                await f.close()
            raise HTTPException(422, f'Upload 1 to {MAX_BATCH} images')
        images = [await decode(f) for f in files]
        rows = await run_in_threadpool(predict, images, second_pass_budget)
        try:
            actions = allocate_rows(rows, PolicyConfig(review_budget, recapture_budget))
        except BudgetOverloaded as exc:
            raise HTTPException(422, {'code': 'BUDGET_OVERLOADED', 'message': str(exc),
                                     'diagnostics': exc.diagnostics}) from None
        request_id = str(uuid.uuid4())
        results = [response(row, request_id, i, action,
                    ['COHORT_BUDGET_ALLOCATION',
                     'PROVISIONAL_NONREFERABLE' if action == Action.AUTO_NONREFERABLE else 'DEFERRED_INTERPRETATION'],
                    'cohort') for i, (row, action) in enumerate(zip(rows, actions))]
        return BatchPrediction(request_id=request_id, results=results,
            review_rate=actions.count(Action.SPECIALIST_REVIEW)/len(rows),
            recapture_rate=actions.count(Action.RECAPTURE)/len(rows),
            second_pass_rate=sum(r['second_pass_activated'] for r in rows)/len(rows),
            budgets={'review': review_budget, 'recapture': recapture_budget,
                     'second_pass': second_pass_budget})

    from api.research import research_router
    app.include_router(research_router(app))
    return app


app = create_app()
