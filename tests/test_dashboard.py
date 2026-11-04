import importlib.util
from pathlib import Path
from urllib.parse import parse_qs, urlparse
import httpx
import pytest


@pytest.mark.skipif(importlib.util.find_spec('streamlit') is None, reason='Streamlit not installed')
def test_dashboard_pages_and_export_render_without_invented_records(monkeypatch):
    from streamlit.testing.v1 import AppTest
    monkeypatch.setenv('QSART_API_URL', 'http://127.0.0.1:1')
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1]/'dashboard/app.py'))
    app.run(timeout=25)
    assert not app.exception
    assert any('No batch' in item.value for item in app.info)
    for page in ('Retinal Image Analysis', 'Research Analytics', 'Batch Screening',
                 'Triage Policy Simulator', 'Model Comparison', 'Experiment Explorer', 'Model Information'):
        app.sidebar.radio[0].set_value(page).run(timeout=25)
        assert not app.exception, page


@pytest.fixture
def dashboard_api(monkeypatch):
    """Explicit synthetic API fixtures verify UI plumbing, not scientific results."""
    requests = []
    records = [
        {'experiment_id': 'selection_dr_fixture_a', 'dataset': 'synthetic_fixture',
         'partition': 'validation', 'scope': 'software_test', 'model_version': 'fixture_a',
         'metrics': {'accuracy': .6, 'macro_f1': .5}, 'notes': ['Synthetic software-test record.']},
        {'experiment_id': 'selection_dr_fixture_b', 'dataset': 'synthetic_fixture',
         'partition': 'validation', 'scope': 'software_test', 'model_version': 'fixture_b',
         'metrics': {'accuracy': .5, 'macro_f1': .4}, 'notes': ['Synthetic software-test record.']},
        {'experiment_id': 'historical_fixture', 'dataset': 'other_fixture', 'partition': 'holdout',
         'scope': 'historical_software_test', 'model_version': 'fixture_old',
         'metrics': {'accuracy': .3, 'paired_comparisons': [{'left': ['ordinal', .9, .4, 'plain']}]},
         'notes': ['Synthetic historical record.']},
    ]

    def get(url, **kwargs):
        parsed = urlparse(url)
        if parsed.path == '/ready':
            data = {'ready': True}
        elif parsed.path == '/triage/config':
            data = {'policy_version': '2.0', 'defaults': {'review_budget': .3, 'recapture_budget': .2,
                    'uncertainty_threshold': .75, 'severity_costs': [0., 0., 3., 6., 10.]},
                    'methods': ['plain', 'entropy_only', 'dual_budget'],
                    'compute_profiles': {'single': 0., 'budget25': .25, 'all': 1.}}
        elif parsed.path == '/research/experiments':
            data = {'experiments': records}
        elif parsed.path == '/research/metrics':
            chosen = parse_qs(parsed.query)['experiment_id'][0]
            record = next(record for record in records if record['experiment_id'] == chosen)
            data = {'experiment_id': chosen, 'scope': record['scope'], 'metrics': record['metrics'],
                    'notes': record['notes']}
        else:
            data = {}
        return httpx.Response(200, json=data)

    def post(url, **kwargs):
        requests.append((urlparse(url).path, kwargs['json']))
        return httpx.Response(200, json={
            'status': 'evaluated', 'dataset_id': 'development_selection_v2', 'images': 3,
            'actions': ['AUTO_NONREFERABLE', 'RECAPTURE', 'SPECIALIST_REVIEW'],
            'metrics': {'auto_nonreferable_count': 1, 'review_count': 1, 'recapture_count': 1,
                        'missed_referable_per_1000_all': 0.},
            'scope': 'development_exploration', 'warnings': ['Synthetic software-test response.']})

    monkeypatch.setattr(httpx, 'get', get)
    monkeypatch.setattr(httpx, 'post', post)
    return requests


@pytest.mark.skipif(importlib.util.find_spec('streamlit') is None, reason='Streamlit not installed')
def test_new_pages_display_stored_metrics_and_send_development_only_simulation(dashboard_api):
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1]/'dashboard/app.py')).run(timeout=25)
    app.sidebar.radio[0].set_value('Model Comparison').run(timeout=25)
    assert not app.exception
    assert len(app.dataframe[0].value) == 2
    assert set(app.dataframe[0].value['experiment_id']) == {
        'selection_dr_fixture_a', 'selection_dr_fixture_b'}
    app.sidebar.radio[0].set_value('Experiment Explorer').run(timeout=25)
    assert not app.exception
    assert 'metrics.paired_comparisons' not in app.dataframe[0].value
    app.selectbox[0].set_value('other_fixture').run(timeout=25)
    assert list(app.dataframe[0].value['experiment_id']) == ['historical_fixture']
    app.sidebar.radio[0].set_value('Triage Policy Simulator').run(timeout=25)
    assert not app.exception
    app.selectbox[1].set_value('all')
    app.slider[0].set_value(.4)
    app.slider[2].set_value(.8)
    next(button for button in app.button if button.label == 'Run development simulation').click().run(timeout=25)
    assert not app.exception
    path, payload = dashboard_api[-1]
    assert path == '/triage/simulate'
    assert payload['dataset_id'] == 'development_selection_v2'
    assert payload['compute_profile'] == 'all'
    assert payload['config']['review_budget'] == .4
    assert payload['config']['uncertainty_threshold'] == .8
    assert 'compute_budget' not in payload['config']
    assert any(metric.label == 'Recommended reviews' and metric.value == '1' for metric in app.metric)


@pytest.mark.skipif(importlib.util.find_spec('streamlit') is None, reason='Streamlit not installed')
@pytest.mark.parametrize('status', ['overloaded', 'unavailable'])
def test_simulator_preserves_overload_and_unavailable_states(monkeypatch, dashboard_api, status):
    from streamlit.testing.v1 import AppTest

    def post(*args, **kwargs):
        return httpx.Response(200, json={'status': status, 'diagnostics': {'needed_review': 5,
            'review_capacity': 1}, 'warnings': ['Synthetic software-test response.']})

    monkeypatch.setattr(httpx, 'post', post)
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1]/'dashboard/app.py')).run(timeout=25)
    app.sidebar.radio[0].set_value('Triage Policy Simulator').run(timeout=25)
    next(button for button in app.button if button.label == 'Run development simulation').click().run(timeout=25)
    assert not app.exception
    if status == 'overloaded':
        assert any('cannot accommodate' in error.value for error in app.error)
    else:
        assert any('unavailable' in warning.value for warning in app.warning)
    assert not app.metric


@pytest.mark.skipif(importlib.util.find_spec('streamlit') is None, reason='Streamlit not installed')
def test_session_overview_and_batch_filters_use_actual_stored_rows(dashboard_api):
    from streamlit.testing.v1 import AppTest
    rows = [
        {'action': 'AUTO_NONREFERABLE', 'predicted_grade': 0, 'referable_probability': .02,
         'quality': {'poor_probability': .1}, 'uncertainty': {'entropy': .1}, 'processing_time_ms': 10.},
        {'action': 'SPECIALIST_REVIEW', 'predicted_grade': 3, 'referable_probability': .9,
         'quality': {'poor_probability': .9}, 'uncertainty': {'entropy': .5}, 'processing_time_ms': 30.},
        {'action': 'RECAPTURE', 'predicted_grade': 0, 'referable_probability': .05,
         'quality': {'poor_probability': .8}, 'uncertainty': {'entropy': .9}, 'processing_time_ms': 20.},
    ]
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1]/'dashboard/app.py'))
    app.session_state['batch_results'] = rows
    app.run(timeout=25)
    assert not app.exception
    assert any(metric.label == 'Automatic non-referable' and metric.value == '1' for metric in app.metric)
    assert any(metric.label == 'Session median processing time' and metric.value == '20.0 ms'
               for metric in app.metric)
    app.sidebar.radio[0].set_value('Batch Screening').run(timeout=25)
    assert not app.exception
    app.selectbox[0].set_value('Poor quality')
    app.selectbox[1].set_value('Highest referable risk first').run(timeout=25)
    displayed = app.dataframe[0].value
    assert len(displayed) == 2
    assert displayed.iloc[0]['referable_probability'] == .9


@pytest.mark.skipif(importlib.util.find_spec('streamlit') is None, reason='Streamlit not installed')
def test_policy_explorer_preserves_missing_metrics_and_labels_ablation(monkeypatch, dashboard_api):
    """Synthetic contract fixture distinguishes missing outcomes from measured zeros."""
    from streamlit.testing.v1 import AppTest
    original_get = httpx.get
    metrics = {'summary': {'fully_evaluated_scenarios': 1, 'incomplete_scenarios': 1, 'estimable_pairs': 0},
        'scenarios': [{'mode': 'no_cp', 'variant': 'tuned', 'method': 'dual_budget',
                       'status': 'incomplete', 'pooled_metrics': None},
                      {'mode': 'no_cp', 'variant': 'tuned', 'method': 'plain', 'status': 'evaluated',
                       'pooled_metrics': {'missed_referable_auto': 0, 'severe_deferred_to_recapture': 1,
                                          'proxy_total_cost_per_image': .4}}]}
    record = {'experiment_id': 'policy_comparison_v1', 'dataset': 'synthetic_fixture',
              'scope': 'development_out_of_fold_policy', 'metrics': metrics, 'notes': ['Synthetic fixture.']}

    def get(url, **kwargs):
        if urlparse(url).path == '/research/experiments':
            return httpx.Response(200, json={'experiments': [record]})
        if urlparse(url).path == '/research/metrics':
            return httpx.Response(200, json=record)
        return original_get(url, **kwargs)

    monkeypatch.setattr(httpx, 'get', get)
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1]/'dashboard/app.py')).run(timeout=25)
    app.sidebar.radio[0].set_value('Experiment Explorer').run(timeout=25)
    next(box for box in app.selectbox if box.label == 'Conformal mode').set_value('no_cp').run(timeout=25)
    assert not app.exception
    table = app.dataframe[-1].value
    assert table['Referable automatic misses'].isna().tolist() == [True, False]
    assert table.iloc[1]['Referable automatic misses'] == 0
    assert table.iloc[1]['Severe recaptures'] == 1
    assert any('Diagnostic ablation' in item.value for item in app.info)
    assert any('Blank pooled metrics' in item.value for item in app.caption)


@pytest.mark.skipif(importlib.util.find_spec('streamlit') is None, reason='Streamlit not installed')
def test_aptos_explorer_keeps_component_representatives_distinct(monkeypatch, dashboard_api):
    """Explicit fixture checks cohort display; these are not APTOS measurements."""
    from streamlit.testing.v1 import AppTest
    original_get = httpx.get
    metrics = {'images': 4, 'population': {'conformal_representative_images': 2, 'temperature_images': 3},
        'summary': {'selected_backbone': 'software_fixture'},
        'candidates': [{'backbone': 'software_fixture', 'selection_metrics': {'accuracy': .5}}],
        'comparisons': [{'method': 'lac', 'alpha': .1, 'randomization_seed': None,
            'raw': {'all_selection_images': {'n_images': 4, 'marginal_coverage': .5, 'empty_set_count': 1},
                    'selection_group_representatives': {'n_images': 2, 'marginal_coverage': 0., 'empty_set_count': 1}},
            'argmax_fallback': {'all_selection_images': {'n_images': 4, 'marginal_coverage': .75, 'mean_set_size': 2.},
                               'selection_group_representatives': {'n_images': 2, 'marginal_coverage': .5, 'mean_set_size': 1.}}}]}
    record = {'experiment_id': 'aptos_development_v1', 'dataset': 'APTOS_2019',
              'scope': 'development_selection_and_independent_calibration', 'metrics': metrics,
              'notes': ['Synthetic software-test fixture.']}

    def get(url, **kwargs):
        if urlparse(url).path == '/research/experiments':
            return httpx.Response(200, json={'experiments': [record]})
        if urlparse(url).path == '/research/metrics':
            return httpx.Response(200, json=record)
        return original_get(url, **kwargs)

    monkeypatch.setattr(httpx, 'get', get)
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1]/'dashboard/app.py')).run(timeout=25)
    app.sidebar.radio[0].set_value('Experiment Explorer').run(timeout=25)
    assert not app.exception
    assert app.dataframe[-1].value.iloc[0]['Images'] == 4
    next(box for box in app.selectbox if box.label == 'Reported population').set_value(
        'One representative per fingerprint group').run(timeout=25)
    assert not app.exception
    table = app.dataframe[-1].value
    assert table.iloc[0]['Images'] == 2
    assert table.iloc[0]['Raw coverage'] == 0.
    assert table.iloc[0]['Coverage with argmax fallback'] == .5
    assert any('Fingerprint groups' in item.value for item in app.warning)


@pytest.mark.skipif(importlib.util.find_spec('streamlit') is None, reason='Streamlit not installed')
def test_finetune_explorer_displays_seed_without_changing_primary(monkeypatch, dashboard_api):
    """Generated results make a later seed look better to test the display-only control."""
    from streamlit.testing.v1 import AppTest
    original_get = httpx.get
    metrics = {'images': 5, 'population': {'conformal_representative_images': 3, 'temperature_images': 4},
        'summary': {'primary_training_seed': 2026},
        'baseline': {'selection_metrics': {'accuracy': .5, 'macro_f1': .4, 'nll': 1.}},
        'runs': [{'training_seed': seed, 'grading_before_temperature': {'accuracy': value, 'macro_f1': value, 'nll': 1.}}
                 for seed, value in [(2026, .6), (2027, .9)]], 'comparisons': []}
    record = {'experiment_id': 'aptos_finetune_v1', 'dataset': 'APTOS_2019',
              'scope': 'development_finetuning_and_training_seed_diagnostics', 'metrics': metrics,
              'notes': ['Synthetic software-test fixture.']}

    def get(url, **kwargs):
        if urlparse(url).path == '/research/experiments':
            return httpx.Response(200, json={'experiments': [record]})
        if urlparse(url).path == '/research/metrics':
            return httpx.Response(200, json=record)
        return original_get(url, **kwargs)

    monkeypatch.setattr(httpx, 'get', get)
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1]/'dashboard/app.py')).run(timeout=25)
    app.sidebar.radio[0].set_value('Experiment Explorer').run(timeout=25)
    assert not app.exception
    table = app.dataframe[-1].value
    assert table.Run.tolist() == ['Frozen-feature baseline', 'Training seed 2026', 'Training seed 2027']
    next(box for box in app.selectbox if box.label == 'Recorded classifier training seed').set_value(2027).run(timeout=25)
    assert not app.exception
    assert any('fixed at 2026' in item.value for item in app.caption)
    assert any('does not select or deploy' in item.value for item in app.caption)


@pytest.mark.skipif(importlib.util.find_spec('streamlit') is None, reason='Streamlit not installed')
def test_external_data_readiness_keeps_reservation_and_overlap_visible(monkeypatch, dashboard_api):
    """Generated metadata verifies data-readiness display without external predictions."""
    from streamlit.testing.v1 import AppTest
    original_get = httpx.get
    metrics = {'images_decoded': 10, 'model_inference_performed': False,
        'source_provenance_status': 'local_package_download_origin_unconfirmed',
        'cross_source_candidate_pairs': 1, 'independence_claim_blocked_by_detected_overlap': True,
        'official_partitions': {name: {'images': 5, 'dr_grade_counts': {'0': 2, '1': 3}}
                                for name in ('official_train', 'official_test')}}
    record = {'experiment_id': 'idrid_data_v1', 'dataset': 'IDRiD',
        'scope': 'external_data_integrity_and_overlap_readiness_only', 'metrics': metrics,
        'notes': ['Generated software fixture; external evaluation remains reserved.']}

    def get(url, **kwargs):
        if urlparse(url).path == '/research/experiments':
            return httpx.Response(200, json={'experiments': [record]})
        if urlparse(url).path == '/research/metrics':
            return httpx.Response(200, json=record)
        return original_get(url, **kwargs)

    monkeypatch.setattr(httpx, 'get', get)
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1]/'dashboard/app.py')).run(timeout=25)
    app.sidebar.radio[0].set_value('Experiment Explorer').run(timeout=25)
    assert not app.exception
    assert app.dataframe[-1].value.Images.tolist() == [5, 5]
    assert any(item.label == 'External predictions' and item.value == '0' for item in app.metric)
    assert any('External cohort reserved' in item.value for item in app.info)
    assert any('download origin is not yet confirmed' in item.value for item in app.warning)
    assert any('overlap candidates require review' in item.value for item in app.warning)


@pytest.mark.skipif(importlib.util.find_spec('streamlit') is None, reason='Streamlit not installed')
def test_confirmed_idrid_origin_keeps_overlap_warning(monkeypatch, dashboard_api):
    """Generated receipt fixture updates current status without changing the reservation."""
    from streamlit.testing.v1 import AppTest
    original_get = httpx.get
    metrics = {'images_decoded': 10, 'model_inference_performed': False,
        'source_provenance_status': 'official_download_user_confirmed',
        'source_confirmation': {'status': 'official_origin_user_confirmed',
                                'confirmed_at_utc': '2026-11-04T13:00:00+00:00'},
        'cross_source_candidate_pairs': 1, 'independence_claim_blocked_by_detected_overlap': True,
        'official_partitions': {'official_test': {'images': 10, 'dr_grade_counts': {'0': 10}}}}
    record = {'experiment_id': 'idrid_data_v1', 'dataset': 'IDRiD',
        'scope': 'external_data_integrity_and_overlap_readiness_only', 'metrics': metrics, 'notes': []}

    def get(url, **kwargs):
        if urlparse(url).path == '/research/experiments':
            return httpx.Response(200, json={'experiments': [record]})
        if urlparse(url).path == '/research/metrics':
            return httpx.Response(200, json=record)
        return original_get(url, **kwargs)

    monkeypatch.setattr(httpx, 'get', get)
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1]/'dashboard/app.py')).run(timeout=25)
    app.sidebar.radio[0].set_value('Experiment Explorer').run(timeout=25)
    assert not app.exception
    assert not any('download origin is not yet confirmed' in item.value for item in app.warning)
    assert any('overlap candidates require review' in item.value for item in app.warning)
    assert any('origin confirmed by the user' in item.value for item in app.caption)
    assert any(item.label == 'External predictions' and item.value == '0' for item in app.metric)
