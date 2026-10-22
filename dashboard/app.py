"""Research interface. All predictions come from the local FastAPI service."""
from io import BytesIO
import hashlib
import json
import os
from pathlib import Path
from urllib.parse import urlencode

import httpx
import pandas as pd
from PIL import Image
import streamlit as st

from api.schemas import DISCLAIMER

st.set_page_config(page_title='QSART Retinal Screening', page_icon='◉', layout='wide')
st.title('QSART — AI-Assisted Retinal Screening Research Prototype')
st.warning(DISCLAIMER)
API_URL = os.environ.get('QSART_API_URL', 'http://127.0.0.1:8000').rstrip('/')


def api_get(path):
    try:
        r = httpx.get(API_URL+path, timeout=10)
        return r.json()
    except (httpx.HTTPError, ValueError):
        return {'ready': False, 'reason': 'API connection unavailable'}


def api_post(path, payload):
    try:
        response = httpx.post(API_URL + path, json=payload, timeout=60)
        data = response.json()
        if response.is_error:
            st.error(data.get('detail', 'The request could not be completed.'))
            return None
        return data
    except (httpx.HTTPError, ValueError):
        st.error('The API did not return a valid result. Check that the service is running.')
        return None


REASONS = {
    'PREDICTED_REFERABLE_GRADE': 'The predicted severity is moderate or worse and requires specialist review.',
    'HIGH_REFERABLE_PROBABILITY': 'The probability of referable disease meets the review threshold.',
    'HIGH_SEVERE_PROBABILITY': 'The probability of severe disease meets the conservative review threshold.',
    'CONFORMAL_SET_INCLUDES_REFERABLE': 'The prediction set includes a potentially referable grade.',
    'EMPTY_CONFORMAL_SET': 'The conformal method returned no eligible grade; automated clearance is withheld.',
    'HIGH_PREDICTIVE_UNCERTAINTY': 'Predictive uncertainty meets the review threshold.',
    'INSUFFICIENT_IMAGE_QUALITY': 'Image quality is insufficient for automated non-referable clearance.',
    'PROVISIONAL_NONREFERABLE': 'The configured non-referable eligibility checks were met.',
    'COHORT_BUDGET_ALLOCATION': 'The cohort policy selected this action within its configured capacities.',
}


def recorded_experiments():
    payload = api_get('/research/experiments')
    return [record for record in payload.get('experiments', []) if isinstance(record, dict)]


def experiment_table(records):
    """Display only supplied records; no default scientific metric is invented."""
    columns = ['experiment_id', 'dataset', 'partition', 'scope', 'model_version']
    table = pd.json_normalize(records)
    present = [column for column in columns if column in table]
    present += [column for column in table if column.startswith('metrics.')]
    return table[present]


def submit(files, params=None, batch=False):
    endpoint = '/predict/batch' if batch else '/predict'
    key = 'files' if batch else 'file'
    uploads = [(key, ('image', f.getvalue(), f.type)) for f in files]
    try:
        with st.spinner('Processing images on CPU…'):
            r = httpx.post(API_URL+endpoint, files=uploads, params=params, timeout=300)
        if r.is_error:
            st.error(r.json().get('detail', 'Prediction failed'))
            return None
        return r.json()
    except (httpx.HTTPError, ValueError):
        st.error('The API did not return a valid result. Check that the service is running.')
        return None


def exports(results):
    table = pd.json_normalize(results)
    st.dataframe(table, width='stretch')
    left, right = st.columns(2)
    left.download_button('Export CSV', table.to_csv(index=False), 'qsart_results.csv', 'text/csv')
    right.download_button('Export JSON', json.dumps(results, indent=2), 'qsart_results.json',
                          'application/json')


with st.sidebar:
    st.caption('LOCAL RESEARCH WORKSPACE')
    page = st.radio('Page', ['Screening Dashboard', 'Retinal Image Analysis',
                            'Batch Screening', 'Triage Policy Simulator', 'Research Analytics',
                            'Model Comparison', 'Experiment Explorer', 'Model Information'])
    ready = api_get('/ready')
    if ready.get('ready'):
        st.success('Model artifacts loaded')
        st.caption('Action policy '+str(ready.get('policy_version', 'Unreported'))+
                   ' · Conformal '+str(ready.get('conformal_status', 'Unreported')))
    else:
        st.error('Inference unavailable')
        st.caption(ready.get('reason', 'Unknown status'))
    st.caption('Uploads are processed temporarily and are not retained by the application. '
               'Exports are generated on request.')
    if st.button('Clear session results'):
        for key in ('single_result', 'single_result_digest', 'batch_results', 'simulation_result'):
            st.session_state.pop(key, None)
        st.rerun()

results = st.session_state.get('batch_results', [])
if page == 'Screening Dashboard':
    info = api_get('/model/info') if ready.get('ready') else {}
    card = info.get('model_card', {})
    confirmation = card.get('confirmation')
    if confirmation:
        st.subheader('Recorded internal confirmation')
        a, b, c = st.columns(3)
        a.metric('Five-grade DR accuracy', f"{confirmation['dr_accuracy']:.1%}")
        b.metric('DR macro-F1', f"{confirmation['dr_macro_f1']:.3f}")
        c.metric('Overall-quality accuracy', f"{confirmation['overall_quality']['accuracy']:.1%}")
        st.caption(f"{confirmation['images']} images from {confirmation['patients']} reserved patients. "
                   'Same-domain research evaluation; external and clinical validation remain outstanding.')
        st.caption('Loaded model: '+card['model_version'])
        st.caption('These recorded classification metrics describe the model evaluation. '
                   'They do not validate the current v2 action policy.')
    st.subheader('Current screening session')
    if not results:
        st.info('No batch has been processed in this session. Use Batch Screening to begin.')
    else:
        frame = pd.DataFrame(results)
        a, b, c, d = st.columns(4)
        a.metric('Processed images', len(results))
        b.metric('Specialist review', int((frame.action == 'SPECIALIST_REVIEW').sum()))
        c.metric('Recapture', int((frame.action == 'RECAPTURE').sum()))
        d.metric('Automatic non-referable', int((frame.action == 'AUTO_NONREFERABLE').sum()))
        latency = pd.to_numeric(frame.get('processing_time_ms', pd.Series(dtype=float)), errors='coerce').dropna()
        if not latency.empty:
            left, right = st.columns(2)
            left.metric('Session median processing time', f'{latency.median():.1f} ms')
            right.metric('Session P95 processing time', f'{latency.quantile(.95):.1f} ms')
            st.caption('Observed API processing times from this session; decode/network/display time '
                       'and independent benchmarking are outside these figures.')
        left, right = st.columns(2)
        left.bar_chart(frame.predicted_grade.value_counts().sort_index())
        right.bar_chart(frame.action.value_counts())
        poor = pd.Series([r['quality']['poor_probability'] >= .5 for r in results])
        st.subheader('Predicted image quality in this session')
        st.bar_chart(pd.Series({'Adequate': int((~poor).sum()), 'Poor': int(poor.sum())}))
        exports(results)
elif page == 'Retinal Image Analysis':
    st.subheader('Analyse a retinal image')
    file = st.file_uploader('Fundus image', type=['png', 'jpg', 'jpeg'])
    second = st.checkbox('Allow a conditional second pass for this image')
    if file:
        try:
            with Image.open(BytesIO(file.getvalue())) as im:
                st.image(im.copy(), caption='Original upload', width=400)
        except (OSError, ValueError):
            st.error('This image cannot be displayed.')
        if st.button('Analyse image', disabled=not ready.get('ready'), type='primary'):
            st.session_state.pop('single_result', None)
            st.session_state.pop('single_result_digest', None)
            result = submit([file], {'second_pass': second})
            if result:
                st.session_state.single_result = result
                st.session_state.single_result_digest = hashlib.sha256(file.getvalue()).hexdigest()
                try:
                    from qsart.preprocessing import preprocess_image
                    resolution = api_get('/model/info').get('image_size', 224)
                    with Image.open(BytesIO(file.getvalue())) as im:
                        preview = preprocess_image(im, image_size=resolution)
                    st.image(preview, caption=f'Preprocessed model input ({resolution} px)', width=300)
                except (ImportError, TypeError, ValueError):
                    pass
    current_digest = hashlib.sha256(file.getvalue()).hexdigest() if file else None
    row = (st.session_state.get('single_result')
           if current_digest == st.session_state.get('single_result_digest') and file else None)
    if row:
        st.caption('Processed with model: '+row['model_version'])
        st.caption('Action policy version: '+str(row.get('policy_version', 'unreported')))
        a, b, c = st.columns(3)
        a.metric('Provisional DR grade', row['predicted_grade'])
        b.metric('Referable DR probability', f"{row['referable_probability']:.1%}")
        c.metric('Processing time', f"{row['processing_time_ms']:.1f} ms")
        st.info(row['action'].replace('_', ' ').title())
        if row['quality']['poor_probability'] >= .5:
            st.warning('Insufficient predicted quality. This result cannot establish absence of disease.')
        st.bar_chart(pd.DataFrame({'Probability': row['grade_probabilities']},
                                 index=['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative']))
        st.write('Action reasons')
        for reason in row['reason_codes']:
            st.write('• ' + REASONS.get(reason, reason.replace('_', ' ').capitalize()))
        st.write('Quality:', row['quality'], 'Uncertainty:', row['uncertainty'])
        conformal = row.get('conformal')
        if conformal and conformal.get('status') in ('available', 'ready', 'calibrated'):
            st.subheader('Ordinal conformal prediction')
            st.write('Possible grades:', conformal.get('prediction_set', []))
            target = conformal.get('target_coverage')
            if isinstance(target, (int, float)):
                st.caption(f'Calibration target coverage: {target:.0%}. '
                           'This is a cohort-level target, not the probability that this image is correct.')
            st.caption('Coverage scope: '+str(conformal.get('coverage_scope', 'Unreported')))
            if 'baseline_lac_set' in conformal:
                st.write('Standard conformal comparator:', conformal['baseline_lac_set'])
        elif conformal and 'prediction_set' in conformal and conformal.get('status') != 'unavailable':
            st.write('Ordinal conformal possible grades:', conformal['prediction_set'])
            st.caption('Conformal status: '+str(conformal.get('status', 'Unreported')))
        else:
            st.info('Conformal prediction is unavailable for this result; no coverage guarantee is implied.')
        for warning in row.get('model_warnings', []):
            st.warning(str(warning))
        st.write('Second pass:', row['second_pass_activated'])
        exports([row])
elif page == 'Batch Screening':
    st.subheader('Screen a cohort')
    st.caption('Budgets apply to this uploaded cohort, with capacities rounded down to image counts. '
               'If mandatory reviews and quality deferrals exceed capacity, the API reports overload. '
               'High-risk cases are never cleared to make the budgets fit.')
    files = st.file_uploader('Fundus images (maximum 64)', accept_multiple_files=True,
                             type=['png', 'jpg', 'jpeg'])
    a, b, c = st.columns(3)
    review = a.slider('Review capacity', 0., 1., .30, .05)
    recapture = b.slider('Recapture capacity', 0., 1., .20, .05)
    compute = c.slider('Maximum second-pass fraction', 0., 1., .20, .05)
    if st.button('Process cohort', disabled=not files or not ready.get('ready'), type='primary'):
        st.session_state.pop('batch_results', None)
        payload = submit(files, {'review_budget': review, 'recapture_budget': recapture,
                                'second_pass_budget': compute}, batch=True)
        if payload:
            st.session_state.batch_results = payload['results']
            st.success(f"Processed {len(payload['results'])} images")
    rows = st.session_state.get('batch_results', [])
    if rows:
        st.caption('Results from the last successfully processed cohort in this session.')
        selected = st.multiselect('Filter action', sorted({r['action'] for r in rows}))
        grades = st.multiselect('Filter provisional grade', list(range(5)))
        quality = st.selectbox('Filter predicted quality', ['All quality levels', 'Adequate quality', 'Poor quality'])
        sort = st.selectbox('Sort results', ['Original upload order', 'Highest referable risk first',
                                           'Highest uncertainty first'])
        filtered = [r for r in rows if (not selected or r['action'] in selected)
                    and (not grades or r['predicted_grade'] in grades)
                    and (quality == 'All quality levels'
                         or (quality == 'Poor quality') == (r['quality']['poor_probability'] >= .5))]
        if sort == 'Highest referable risk first':
            filtered.sort(key=lambda row: row['referable_probability'], reverse=True)
        elif sort == 'Highest uncertainty first':
            filtered.sort(key=lambda row: row['uncertainty']['entropy'], reverse=True)
        st.caption(f'{len(filtered)} of {len(rows)} images match the filters. Exports use these rows.')
        exports(filtered)
elif page == 'Triage Policy Simulator':
    st.subheader('Explore a development policy')
    st.caption('Uses cached predictions from the development selection partition only. '
               'Changes are exploratory and never tune the consumed confirmation cohort.')
    settings = api_get('/triage/config')
    defaults = settings.get('defaults', {})
    methods = settings.get('methods', [])
    if not methods:
        st.info('Policy simulation is unavailable. Start the current API and prepare real selection scores.')
    else:
        st.caption('Action policy version: '+str(settings.get('policy_version', 'Unreported')))
        method = st.selectbox('Policy strategy', methods, index=methods.index('dual_budget')
                              if 'dual_budget' in methods else 0)
        profiles = settings.get('compute_profiles') or {'single': 0.}
        profile = st.selectbox('Precomputed inference profile', list(profiles))
        st.caption('Each profile uses fixed cached scores from real inference with compute activation '
                   'precomputed in eight-image requests. Changing review or recapture capacity '
                   'reallocates these scores and does not rerun the model.')
        left, middle, right = st.columns(3)
        review = left.slider('Maximum review fraction', 0., 1., float(defaults.get('review_budget', .3)), .01)
        recapture = middle.slider('Maximum recapture fraction', 0., 1.,
                                  float(defaults.get('recapture_budget', .2)), .01)
        entropy = right.slider('Uncertainty review threshold', 0., 1.,
                               float(defaults.get('uncertainty_threshold', .75)), .01)
        st.caption('Severity weights are research proxy costs, not measured clinical harms.')
        weights = list(defaults.get('severity_costs', [0., 0., 3., 6., 10.]))
        columns = st.columns(3)
        for grade, label, column in zip((2, 3, 4), ('Moderate DR cost', 'Severe DR cost',
                                                  'Proliferative DR cost'), columns):
            weights[grade] = column.slider(label, 0., max(20., float(weights[grade])),
                                           float(weights[grade]), .5)
        st.info('The selected profile chooses a stored inference experiment. '
                'Batch Screening applies the actual conditional second-pass budget to uploaded images.')
        if st.button('Run development simulation', type='primary'):
            payload = {'dataset_id': 'development_selection_v2', 'method': method,
                       'compute_profile': profile,
                       'config': {'review_budget': review, 'recapture_budget': recapture,
                                  'uncertainty_threshold': entropy, 'severity_costs': weights}}
            response = api_post('/triage/simulate', payload)
            if response:
                st.session_state.simulation_result = {'request': payload, 'response': response}
        simulation = st.session_state.get('simulation_result')
        if simulation:
            result = simulation['response']
            st.caption('Stored result from the last simulation request; rerun after changing controls.')
            status = result.get('status')
            if status == 'overloaded':
                st.error('The configured capacities cannot accommodate the required deferrals. '
                         'No unsafe automatic clearances were produced.')
                st.json(result.get('diagnostics', {}))
            elif status == 'evaluated':
                metrics = result.get('metrics', {})
                left, middle, right = st.columns(3)
                for column, key, label in ((left, 'auto_nonreferable_count', 'Automatic non-referable'),
                                           (middle, 'review_count', 'Recommended reviews'),
                                           (right, 'recapture_count', 'Recommended recaptures')):
                    if key in metrics:
                        column.metric(label, metrics[key])
                if 'missed_referable_per_1000_all' in metrics:
                    st.metric('Missed referable per 1,000 selection images',
                              f"{metrics['missed_referable_per_1000_all']:.1f}")
                st.caption('Development exploration only. Recommended deferral is not successful care.')
                st.json(metrics)
                if result.get('actions'):
                    st.bar_chart(pd.Series(result['actions']).value_counts())
            else:
                st.warning('Real selection scores are unavailable for this simulation.')
            for warning in result.get('warnings', []):
                st.warning(str(warning))
            st.download_button('Export simulation JSON', json.dumps(simulation, indent=2),
                               'qsart_development_simulation.json', 'application/json')
elif page == 'Research Analytics':
    st.subheader('Recorded experiments')
    root = Path(os.environ.get('QSART_REPORTS_DIR', 'reports'))
    files = sorted(root.rglob('*.json')) if root.exists() else []
    if not files:
        st.info('No experiment reports are available. No performance results are implied.')
    else:
        chosen = st.selectbox('Experiment report', files, format_func=lambda p: str(p.relative_to(root)))
        try:
            st.json(json.loads(chosen.read_text(encoding='utf-8')))
        except (OSError, ValueError):
            st.error('Report could not be read.')
    figures = sorted(root.rglob('*.png')) if root.exists() else []
    for figure in figures:
        st.image(str(figure), caption=figure.stem.replace('_', ' '))
elif page == 'Model Comparison':
    st.subheader('Recorded development model comparisons')
    st.caption('Checkpoint-selection results are development evidence. They are not external validation '
               'and must not be compared as paired improvements over different historical cohorts.')
    records = [record for record in recorded_experiments()
               if str(record.get('experiment_id', '')).startswith('selection_dr_')]
    if not records:
        st.info('No recorded DR selection comparisons are available from the API.')
    else:
        table = experiment_table(records)
        st.dataframe(table, width='stretch')
        numeric = [column for column in table if column.startswith('metrics.')
                   and pd.api.types.is_numeric_dtype(table[column])]
        if numeric:
            metric = st.selectbox('Recorded metric', numeric,
                                  index=numeric.index('metrics.macro_f1') if 'metrics.macro_f1' in numeric else 0)
            st.bar_chart(table.set_index('experiment_id')[[metric]])
        for record in records:
            for note in record.get('notes', []):
                st.caption(str(note))
        st.download_button('Export comparison CSV', table.to_csv(index=False),
                           'qsart_model_comparison.csv', 'text/csv')
elif page == 'Experiment Explorer':
    st.subheader('Explore recorded research evidence')
    records = recorded_experiments()
    if not records:
        st.info('No real experiment records are available from the API. No results are implied.')
    else:
        left, right = st.columns(2)
        datasets = sorted({str(record.get('dataset', 'Unreported')) for record in records})
        scopes = sorted({str(record.get('scope', 'Unreported')) for record in records})
        dataset = left.selectbox('Dataset', ['All datasets'] + datasets)
        scope = right.selectbox('Evidence scope', ['All scopes'] + scopes)
        records = [record for record in records
                   if (dataset == 'All datasets' or str(record.get('dataset', 'Unreported')) == dataset)
                   and (scope == 'All scopes' or str(record.get('scope', 'Unreported')) == scope)]
        if not records:
            st.info('No recorded experiments match these filters.')
        else:
            st.dataframe(experiment_table(records), width='stretch')
            ids = [record['experiment_id'] for record in records]
            chosen = st.selectbox('Experiment', ids)
            detail = api_get('/research/metrics?' + urlencode({'experiment_id': chosen}))
            if detail.get('experiment_id') == chosen:
                st.caption('Evidence scope: '+str(detail.get('scope', 'Unreported')))
                st.json(detail.get('metrics', {}))
                for note in detail.get('notes', []):
                    st.caption(str(note))
                st.download_button('Export experiment JSON', json.dumps(detail, indent=2),
                                   'qsart_experiment.json', 'application/json')
            else:
                st.warning('The selected experiment details could not be retrieved.')
else:
    st.subheader('Model information and limitations')
    st.json(api_get('/model/info'))
    st.write('Supported uploads: PNG and JPEG, up to 12 MiB each; batches up to 64 images.')
    st.write('Training provenance and the model card describe the measured evaluation scope. '
             'These research models are not clinically validated. Entropy is a predictive uncertainty score, not the probability '
             'that an individual result is correct. Single-image routing has no cohort budget guarantee.')
