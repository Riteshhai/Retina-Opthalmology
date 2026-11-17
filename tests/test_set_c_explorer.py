"""Exercise all displayed Set-C modes against the actual audited aggregate."""

import json
from pathlib import Path
from urllib.parse import urlsplit

from fastapi.testclient import TestClient
import httpx
from streamlit.testing.v1 import AppTest

from api.main import create_app


ROOT = Path(__file__).resolve().parents[1]


def test_explorer_keeps_every_allocation_and_comparison(monkeypatch):
    evaluation = json.loads((ROOT / 'reports/set_c_validation_v1/evaluation.json').read_bytes())
    monkeypatch.setenv('QSART_REPORTS_DIR', str(ROOT / 'reports'))
    monkeypatch.delenv('QSART_CONFORMAL', raising=False)
    with TestClient(create_app(object())) as client:
        def local_get(url, **kwargs):
            target = urlsplit(url)
            return client.get(target.path + (f'?{target.query}' if target.query else ''))

        monkeypatch.setattr(httpx, 'get', local_get)
        app = AppTest.from_file(str(ROOT / 'dashboard/app.py')).run(timeout=30)
        app.sidebar.radio[0].set_value('Experiment Explorer').run(timeout=30)
        next(box for box in app.selectbox if box.label == 'Experiment').set_value('set_c_validation_v1').run(timeout=30)

        for mode in ('ordinal', 'lac', 'no_cp'):
            next(box for box in app.selectbox if box.label == 'Set-C policy mode').set_value(mode).run(timeout=30)
            assert not app.exception
            scenarios = next(frame.value for frame in app.dataframe if 'Allocation' in frame.value.columns)
            comparisons = next(frame.value for frame in app.dataframe if 'Identical actions' in frame.value.columns)
            expected = [row for row in evaluation['scenarios'] if row['mode'] == mode]
            assert scenarios['Allocation'].tolist() == ['Evaluated' if x['outcomes'] else 'Overload' for x in expected]
            assert scenarios['Fully automatic'].fillna(-1).tolist() == [
                row['outcomes']['patient_actions']['AUTO_NONREFERABLE'] if row['outcomes'] else -1 for row in expected
            ]
            assert len(comparisons) == 28
