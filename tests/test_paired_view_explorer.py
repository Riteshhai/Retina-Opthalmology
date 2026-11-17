"""Verify the fixed-budget UI shows the audited paired-view outcomes."""

import json
from pathlib import Path
from urllib.parse import urlsplit

from fastapi.testclient import TestClient
import httpx
import pytest
from streamlit.testing.v1 import AppTest

from api.main import create_app


ROOT = Path(__file__).resolve().parents[1]


def test_paired_view_explorer_shows_every_budget_and_exact_comparison(monkeypatch):
    evaluation = json.loads((ROOT / 'reports/paired_view_v1/evaluation.json').read_bytes())
    monkeypatch.setenv('QSART_REPORTS_DIR', str(ROOT / 'reports'))
    monkeypatch.delenv('QSART_CONFORMAL', raising=False)
    with TestClient(create_app(object())) as client:
        def local_get(url, **kwargs):
            target = urlsplit(url)
            return client.get(target.path + (f'?{target.query}' if target.query else ''))

        monkeypatch.setattr(httpx, 'get', local_get)
        app = AppTest.from_file(str(ROOT / 'dashboard/app.py')).run(timeout=30)
        app.sidebar.radio[0].set_value('Experiment Explorer').run(timeout=30)
        next(box for box in app.selectbox if box.label == 'Experiment').set_value('paired_view_v1').run(timeout=30)
        for budget in evaluation['budgets']:
            next(slider for slider in app.select_slider if slider.label == 'Extra-view budget').set_value(budget).run(timeout=30)
            assert not app.exception
            results = next(frame.value for frame in app.dataframe if 'Extra views used' in frame.value.columns)
            expected = [row for row in evaluation['scenarios'] if row['budget'] == budget]
            assert results['Available slots'].tolist() == [row['slots'] for row in expected]
            assert results['Extra views used'].tolist() == [row['outcomes']['extra_views'] for row in expected]
            assert results['Research cost per eye'].tolist() == pytest.approx(
                [row['outcomes']['ledger_per_eye'] for row in expected])
            pairs = [frame.value for frame in app.dataframe if 'Extra views per policy' in frame.value.columns]
            expected_pairs = [pair for pair in evaluation['comparisons'] if pair['budget'] == budget]
            assert (len(pairs[0]) if pairs else 0) == len(expected_pairs)

        app.select_slider[0].set_value(.25).run(timeout=30)
        next(box for box in app.selectbox if box.label == 'Outcome').set_value('grade_accuracy').run(timeout=30)
        pairs = next(frame.value for frame in app.dataframe if 'Extra views per policy' in frame.value.columns)
        assert pairs['Predicted value minus comparator'].tolist() == pytest.approx([
            pair['intervals']['grade_accuracy']['difference_candidate_minus_reference']
            for pair in evaluation['comparisons'] if pair['budget'] == .25])
