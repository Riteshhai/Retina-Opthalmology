"""HTTP boundary tests use explicit software fixtures, never trained predictions."""
from io import BytesIO
import math
import struct
import zlib

from fastapi.testclient import TestClient
from PIL import Image
import pytest

from api.main import MAX_BATCH, MAX_BYTES, MAX_PIXELS, create_app


def png_bytes(size=(64, 64)):
    stream = BytesIO()
    Image.new('RGB', size, (120, 40, 10)).save(stream, format='PNG')
    return stream.getvalue()


class BoundaryFixture:
    """Injected software-only rows; records whether validation allowed inference."""
    model_version = 'software-validation-fixture'

    def __init__(self, poor=False):
        self.poor = poor
        self.calls = []

    def predict_images(self, images, second_pass_budget=0):
        self.calls.append((len(images), second_pass_budget))
        poor = .9 if self.poor else .1
        return [dict(model_version=self.model_version, predicted_grade=0,
                     grade_probabilities=[.8, .05, .05, .05, .05], referable_probability=.15,
                     quality={'probabilities': [poor, 1-poor], 'predicted_class': int(poor < .5),
                              'poor_probability': poor},
                     uncertainty={'entropy': .3, 'confidence': .8},
                     second_pass_activated=False, processing_time_ms=1.) for _ in images]


def upload_batch(count):
    data = png_bytes()
    return [('files', (f'fixture_{index}.png', data, 'image/png')) for index in range(count)]


def test_upload_above_actual_byte_limit_rejected_before_inference():
    engine = BoundaryFixture()
    data = png_bytes()
    oversized = data + b'\0' * (MAX_BYTES + 1 - len(data))
    with TestClient(create_app(engine)) as client:
        response = client.post('/predict', files={'file': ('large.png', oversized, 'image/png')})
    assert response.status_code == 413
    assert 'upload limit' in response.json()['detail']
    assert engine.calls == []


def test_upload_at_byte_limit_is_accepted():
    # A valid PNG with trailing bytes exercises the <= comparison without a
    # large bitmap allocation. Pillow ignores the bytes after its IEND chunk.
    engine = BoundaryFixture()
    data = png_bytes()
    exact_limit = data + b'\0' * (MAX_BYTES - len(data))
    with TestClient(create_app(engine)) as client:
        response = client.post('/predict', files={'file': ('boundary.png', exact_limit, 'image/png')})
    assert response.status_code == 200, response.text
    assert engine.calls == [(1, 0.)]


@pytest.mark.parametrize('size', [(31, 64), (64, 31), (1, 1)])
def test_too_small_dimensions_rejected_before_inference(size):
    engine = BoundaryFixture()
    with TestClient(create_app(engine)) as client:
        response = client.post('/predict', files={'file': ('small.png', png_bytes(size), 'image/png')})
    assert response.status_code == 422
    assert 'dimensions' in response.json()['detail']
    assert engine.calls == []


def test_minimum_dimensions_are_accepted():
    engine = BoundaryFixture()
    with TestClient(create_app(engine)) as client:
        response = client.post('/predict', files={'file': ('minimum.png', png_bytes((32, 32)), 'image/png')})
    assert response.status_code == 200, response.text
    assert engine.calls == [(1, 0.)]


def test_excessive_declared_pixel_dimensions_rejected_without_bitmap_allocation():
    data = bytearray(png_bytes())
    # Modify a PNG's IHDR and its CRC. Its declared dimensions exceed the API's
    # cap, but no enormous decompressed image is needed to test header rejection.
    width, height = 6000, MAX_PIXELS // 6000 + 1
    data[16:24] = struct.pack('>II', width, height)
    data[29:33] = struct.pack('>I', zlib.crc32(data[12:29]) & 0xffffffff)
    engine = BoundaryFixture()
    with TestClient(create_app(engine)) as client:
        response = client.post('/predict', files={'file': ('huge.png', bytes(data), 'image/png')})
    assert response.status_code == 422
    assert 'dimensions' in response.json()['detail']
    assert engine.calls == []


def test_oversize_batch_rejected_before_inference():
    engine = BoundaryFixture()
    with TestClient(create_app(engine)) as client:
        response = client.post('/predict/batch', files=upload_batch(MAX_BATCH + 1))
    assert response.status_code == 422
    assert str(MAX_BATCH) in response.json()['detail']
    assert engine.calls == []


def test_batch_at_count_limit_is_accepted():
    engine = BoundaryFixture()
    with TestClient(create_app(engine)) as client:
        response = client.post('/predict/batch', files=upload_batch(MAX_BATCH))
    assert response.status_code == 200, response.text
    assert len(response.json()['results']) == MAX_BATCH
    assert engine.calls == [(MAX_BATCH, 0.)]


@pytest.mark.parametrize('count,review,recapture', [(4, 0., 0.), (3, .5, .5)])
def test_low_quality_infeasible_budget_never_silently_auto_screens(count, review, recapture):
    engine = BoundaryFixture(poor=True)
    with TestClient(create_app(engine)) as client:
        response = client.post('/predict/batch',
                               params={'review_budget': review, 'recapture_budget': recapture},
                               files=upload_batch(count))
    assert response.status_code == 422
    detail = response.json()['detail']
    assert detail['code'] == 'BUDGET_OVERLOADED'
    assert detail['diagnostics']['needed_total'] == count
    assert detail['diagnostics']['total_shortfall'] > 0
    assert 'results' not in response.json()
    assert engine.calls == [(count, 0.)]


def test_feasible_low_quality_batch_defers_with_independent_integer_capacities():
    engine = BoundaryFixture(poor=True)
    count, review, recapture = 4, .5, .5
    with TestClient(create_app(engine)) as client:
        response = client.post('/predict/batch',
                               params={'review_budget': review, 'recapture_budget': recapture},
                               files=upload_batch(count))
    assert response.status_code == 200, response.text
    payload = response.json()
    actions = [row['action'] for row in payload['results']]
    assert len(actions) == count
    assert 'AUTO_SCREEN' not in actions
    assert actions.count('SPECIALIST_REVIEW') <= math.floor(count * review)
    assert actions.count('RECAPTURE') <= math.floor(count * recapture)
    assert payload['review_rate'] == .5 and payload['recapture_rate'] == .5
    assert all(row['policy_scope'] == 'cohort' for row in payload['results'])
    assert all('DEFERRED_INTERPRETATION' in row['reason_codes'] for row in payload['results'])


@pytest.mark.parametrize('parameter,value', [('review_budget', -0.1), ('review_budget', 1.1),
                                            ('recapture_budget', -0.1), ('recapture_budget', 1.1),
                                            ('second_pass_budget', -0.1), ('second_pass_budget', 1.1)])
def test_invalid_budget_query_rejected_before_inference(parameter, value):
    engine = BoundaryFixture()
    with TestClient(create_app(engine)) as client:
        response = client.post('/predict/batch', params={parameter: value}, files=upload_batch(1))
    assert response.status_code == 422
    assert engine.calls == []


def test_model_failure_returns_unavailable_without_exposing_exception_text():
    class FailingFixture(BoundaryFixture):
        def predict_images(self, images, second_pass_budget=0):
            raise RuntimeError('sensitive-internal-value')

    with TestClient(create_app(FailingFixture())) as client:
        response = client.post('/predict', files={'file': ('fixture.png', png_bytes(), 'image/png')})
    assert response.status_code == 503
    assert 'sensitive-internal-value' not in response.text
    assert 'no result was produced' in response.json()['detail']
