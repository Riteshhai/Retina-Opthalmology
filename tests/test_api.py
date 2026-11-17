from datetime import datetime, timedelta
from io import BytesIO
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from api.main import create_app


def image_bytes():
    b = BytesIO()
    Image.new('RGB', (64, 64), 'red').save(b, format='PNG')
    return b.getvalue()


class SoftwareFixture:
    """Dependency-injected contract test; never loaded by production startup."""
    model_version = 'synthetic-software-fixture'

    def predict_images(self, images, second_pass_budget=0):
        return [dict(model_version=self.model_version, predicted_grade=0,
            grade_probabilities=[.8, .05, .05, .05, .05], referable_probability=.15,
            quality={'probabilities': [.1, .9], 'predicted_class': 1, 'poor_probability': .1},
            uncertainty={'entropy': .3, 'confidence': .8},
            second_pass_activated=False, processing_time_ms=1.) for _ in images]


def test_liveness_readiness_and_missing_artifacts(monkeypatch, tmp_path):
    monkeypatch.setenv('QSART_DR_CHECKPOINT', str(tmp_path/'missing.pt'))
    with TestClient(create_app()) as c:
        assert c.get('/health').status_code == 200
        assert c.head('/health').status_code == 200
        assert c.get('/ready').status_code == 503
        assert c.head('/ready').status_code == 503
        assert c.post('/predict', files={'file': ('x.png', image_bytes(), 'image/png')}).status_code == 503
        assert c.get('/docs').status_code == 200


def test_portal_readiness_is_independent_of_inference(monkeypatch):
    monkeypatch.delenv('QSART_PORTAL_DATABASE_URL', raising=False)
    with TestClient(create_app(SoftwareFixture())) as client:
        portal = client.get('/portal/ready')
        assert portal.status_code == 503
        assert portal.json() == {'ready': False, 'status': 'not_configured'}
        assert client.get('/ready').status_code == 200

    monkeypatch.setenv('QSART_PORTAL_DATABASE_URL', 'invalid-db-url-with-private-configuration')
    with TestClient(create_app(SoftwareFixture())) as client:
        portal = client.get('/portal/ready')
        assert portal.status_code == 503
        assert portal.json() == {'ready': False, 'status': 'unavailable'}
        assert client.get('/health').status_code == 200


def test_prediction_contract_and_batch():
    with TestClient(create_app(SoftwareFixture())) as c:
        assert c.get('/ready').json()['ready']
        r = c.post('/predict', files={'file': ('x.png', image_bytes(), 'image/png')})
        assert r.status_code == 200, r.text
        assert r.json()['model_version'] == 'synthetic-software-fixture'
        assert r.json()['policy_scope'] == 'single_image'
        assert r.json()['severe_probability'] == pytest.approx(.1)
        assert datetime.fromisoformat(r.json()['inference_timestamp']).utcoffset() == timedelta(0)
        files = [('files', ('x.png', image_bytes(), 'image/png'))]*10
        r = c.post('/predict/batch?review_budget=.2&recapture_budget=.1', files=files)
        assert r.status_code == 200, r.text
        assert len(r.json()['results']) == 10
        assert len({row['inference_timestamp'] for row in r.json()['results']}) == 1
        assert r.json()['review_rate'] <= .2
        assert r.json()['recapture_rate'] <= .1


def test_reject_corrupt_unsupported_and_mismatched_mime():
    with TestClient(create_app(SoftwareFixture())) as c:
        assert c.post('/predict', files={'file': ('x.txt', b'x', 'text/plain')}).status_code == 415
        assert c.post('/predict', files={'file': ('x.png', b'broken', 'image/png')}).status_code == 422
        assert c.post('/predict', files={'file': ('x.jpg', image_bytes(), 'image/jpeg')}).status_code == 415
        assert c.post('/predict').status_code == 422


def test_rejected_batch_releases_images_decoded_so_far(monkeypatch):
    original = Image.Image.convert
    decoded = []

    def remember(image, *args, **kwargs):
        converted = original(image, *args, **kwargs)
        decoded.append(converted)
        return converted

    monkeypatch.setattr(Image.Image, 'convert', remember)
    files = [('files', ('valid.png', image_bytes(), 'image/png')),
             ('files', ('corrupt.png', b'broken', 'image/png'))]
    with TestClient(create_app(SoftwareFixture())) as client:
        assert client.post('/predict/batch', files=files).status_code == 422
    assert len(decoded) == 1
    with pytest.raises(ValueError, match='closed image'):
        decoded[0].getpixel((0, 0))
