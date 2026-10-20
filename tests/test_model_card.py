import json
from fastapi.testclient import TestClient
from api.main import create_app


class CardEngine:
    calibration = {'dr': {'checkpoint_sha256': 'dr-test'}, 'quality': {'checkpoint_sha256': 'q-test'}}

    def info(self):
        return {'model_version': 'software-contract-only'}


def test_model_card_only_exposes_metrics_for_matching_weights(monkeypatch, tmp_path):
    path = tmp_path/'card.json'
    monkeypatch.setenv('QSART_MODEL_CARD', str(path))
    path.write_text(json.dumps({'dr_checkpoint_sha256': 'wrong', 'quality_checkpoint_sha256': 'q-test',
                                'performance': {'accuracy': .99}}))
    with TestClient(create_app(CardEngine())) as client:
        result = client.get('/model/info').json()
        assert result['model_card_status'] == 'unavailable_or_mismatched'
        assert 'model_card' not in result
        path.write_text(json.dumps({'dr_checkpoint_sha256': 'dr-test', 'quality_checkpoint_sha256': 'q-test',
                                    'performance': {'accuracy': .5}}))
        result = client.get('/model/info').json()
        assert result['model_card_status'] == 'matched'
        assert result['model_card']['performance']['accuracy'] == .5
