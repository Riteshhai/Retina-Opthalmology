from io import BytesIO
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
        assert c.get('/ready').status_code == 503
        assert c.post('/predict', files={'file': ('x.png', image_bytes(), 'image/png')}).status_code == 503
        assert c.get('/docs').status_code == 200


def test_prediction_contract_and_batch():
    with TestClient(create_app(SoftwareFixture())) as c:
        assert c.get('/ready').json()['ready']
        r = c.post('/predict', files={'file': ('x.png', image_bytes(), 'image/png')})
        assert r.status_code == 200, r.text
        assert r.json()['model_version'] == 'synthetic-software-fixture'
        assert r.json()['policy_scope'] == 'single_image'
        files = [('files', ('x.png', image_bytes(), 'image/png'))]*10
        r = c.post('/predict/batch?review_budget=.2&recapture_budget=.1', files=files)
        assert r.status_code == 200, r.text
        assert len(r.json()['results']) == 10
        assert r.json()['review_rate'] <= .2
        assert r.json()['recapture_rate'] <= .1


def test_reject_corrupt_unsupported_and_mismatched_mime():
    with TestClient(create_app(SoftwareFixture())) as c:
        assert c.post('/predict', files={'file': ('x.txt', b'x', 'text/plain')}).status_code == 415
        assert c.post('/predict', files={'file': ('x.png', b'broken', 'image/png')}).status_code == 422
        assert c.post('/predict', files={'file': ('x.jpg', image_bytes(), 'image/jpeg')}).status_code == 415
        assert c.post('/predict').status_code == 422
