"""Research interface. All predictions come from the local FastAPI service."""
from io import BytesIO
import hashlib
import json
import os
from pathlib import Path

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
                            'Research Analytics', 'Batch Screening', 'Model Information'])
    ready = api_get('/ready')
    if ready.get('ready'):
        st.success('Model artifacts loaded')
    else:
        st.error('Inference unavailable')
        st.caption(ready.get('reason', 'Unknown status'))
    st.caption('Uploads are processed temporarily and are not retained by the application. '
               'Exports are generated on request.')
    if st.button('Clear session results'):
        for key in ('single_result', 'batch_results'):
            st.session_state.pop(key, None)
        st.rerun()

results = st.session_state.get('batch_results', [])
if page == 'Screening Dashboard':
    st.subheader('Current screening session')
    if not results:
        st.info('No batch has been processed in this session. Use Batch Screening to begin.')
    else:
        frame = pd.DataFrame(results)
        a, b, c = st.columns(3)
        a.metric('Processed images', len(results))
        b.metric('Specialist review', int((frame.action == 'SPECIALIST_REVIEW').sum()))
        c.metric('Recapture', int((frame.action == 'RECAPTURE').sum()))
        left, right = st.columns(2)
        left.bar_chart(frame.predicted_grade.value_counts().sort_index())
        right.bar_chart(frame.action.value_counts())
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
        a, b, c = st.columns(3)
        a.metric('Provisional DR grade', row['predicted_grade'])
        b.metric('Referable DR probability', f"{row['referable_probability']:.1%}")
        c.metric('Processing time', f"{row['processing_time_ms']:.1f} ms")
        st.info(row['action'].replace('_', ' ').title())
        if row['quality']['poor_probability'] >= .5:
            st.warning('Insufficient predicted quality. This result cannot establish absence of disease.')
        st.bar_chart(pd.DataFrame({'Probability': row['grade_probabilities']},
                                 index=['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative']))
        st.write('Reason codes:', ', '.join(row['reason_codes']))
        st.write('Quality:', row['quality'], 'Uncertainty:', row['uncertainty'])
        st.write('Second pass:', row['second_pass_activated'])
        exports([row])
elif page == 'Batch Screening':
    st.subheader('Screen a cohort')
    st.caption('Budgets apply to this uploaded cohort, with capacities rounded down to image counts. '
               'If mandatory quality deferrals exceed capacity, the API reports infeasibility.')
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
        filtered = [r for r in rows if (not selected or r['action'] in selected)
                    and (not grades or r['predicted_grade'] in grades)]
        exports(filtered)
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
else:
    st.subheader('Model information and limitations')
    st.json(api_get('/model/info'))
    st.write('Supported uploads: PNG and JPEG, up to 12 MiB each; batches up to 64 images.')
    st.write('Training provenance and the model card describe the measured evaluation scope. '
             'These research models are not clinically validated. Entropy is a predictive uncertainty score, not the probability '
             'that an individual result is correct. Single-image routing has no cohort budget guarantee.')
