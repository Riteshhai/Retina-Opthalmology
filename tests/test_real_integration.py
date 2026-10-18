"""Opt-in actual retinal-image integration. Never substitutes synthetic fixtures."""
from io import BytesIO
import json
import os
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image
import pytest

from api.main import create_app


@pytest.mark.integration
def test_real_trained_pipeline_and_batch_export():
    manifest = os.environ.get('QSART_REAL_MANIFEST')
    if not manifest:
        pytest.skip('Set QSART_REAL_MANIFEST to opt into local real-image verification')
    import pandas as pd
    from qsart.inference.engine import InferenceEngine
    frame = pd.read_csv(manifest)
    row = frame.loc[frame.split == 'holdout'].iloc[0]
    image_path = Path(row.image_path)
    assert image_path.is_file()
    with Image.open(image_path) as image:
        buffer = BytesIO()
        image.convert('RGB').save(buffer, format='JPEG')
    engine = InferenceEngine('checkpoints/dr/best.pt', 'checkpoints/quality/best.pt',
                             'checkpoints/calibration.json')
    assert engine.info()['data_purpose'] == 'real_research'
    with TestClient(create_app(engine)) as client:
        assert client.get('/ready').status_code == 200
        r = client.post('/predict?second_pass=true', files={
            'file': ('retina.jpg', buffer.getvalue(), 'image/jpeg')})
        assert r.status_code == 200, r.text
        result = r.json()
        assert len(result['grade_probabilities']) == 5
        assert abs(sum(result['grade_probabilities'])-1) < 1e-5
        assert result['second_pass_activated']
        batch = client.post('/predict/batch?review_budget=1&recapture_budget=1&second_pass_budget=.5',
            files=[('files', ('retina.jpg', buffer.getvalue(), 'image/jpeg'))]*2)
        assert batch.status_code == 200, batch.text
        results = batch.json()['results']
        # The same JSON/CSV serialization used by the dashboard exports.
        assert len(json.loads(json.dumps(results))) == 2
        assert len(pd.read_csv(BytesIO(pd.json_normalize(results).to_csv(index=False).encode()))) == 2
