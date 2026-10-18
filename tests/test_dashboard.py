import importlib.util
from pathlib import Path
import pytest


@pytest.mark.skipif(importlib.util.find_spec('streamlit') is None, reason='Streamlit not installed')
def test_dashboard_pages_and_export_render_without_invented_records(monkeypatch):
    from streamlit.testing.v1 import AppTest
    monkeypatch.setenv('QSART_API_URL', 'http://127.0.0.1:1')
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1]/'dashboard/app.py'))
    app.run(timeout=25)
    assert not app.exception
    assert any('No batch' in item.value for item in app.info)
    for page in ('Retinal Image Analysis', 'Research Analytics', 'Batch Screening', 'Model Information'):
        app.sidebar.radio[0].set_value(page).run(timeout=25)
        assert not app.exception, page
