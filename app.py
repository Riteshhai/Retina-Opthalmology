"""Research benchmark dashboard. No diagnostic model, no clinical predictions."""
import json
import pandas as pd
import streamlit as st
from qsart.synthetic import make_mock
from qsart.cli import evaluate

st.set_page_config(page_title='QSART Policy Research Bench',layout='wide')
st.title('QSART — Retinal Triage Policy Research Bench')
st.warning('RESEARCH PROTOTYPE ONLY. This does not interpret fundus photographs, diagnose disease, or recommend actual patient care. All built-in data are synthetic mock scores.')
st.markdown('Upload two *separate* CSVs containing independently generated model scores and 0–4 reference grades, or explore mock data. Do not upload patient-identifying records.')
with st.expander('Required columns and meaning'):
    st.code('image_id, patient_id, grade, quality, p_rdr, p_severe, uncertainty')
    st.write('quality = estimated gradability (1=good), p_rdr = risk of grade≥2, p_severe = risk of grade≥3; uncertainty = 0–1 measure. Columns must be generated elsewhere from a validated model. The app does not produce them.')

with st.sidebar:
    st.header('Retrospective evaluation settings')
    max_review=st.slider('Maximum review fraction',.0,1.0,.50,.05)
    max_recapture=st.slider('Maximum recapture fraction',.0,1.0,.30,.05)
    min_routine=st.slider('Minimum routine fraction',.0,1.0,.18,.05)
    cal_file=st.file_uploader('Calibration CSV',type=['csv'],key='cal')
    test_file=st.file_uploader('Untouched test CSV',type=['csv'],key='test')
    run=st.button('Evaluate research policy',type='primary')

if run:
    try:
        if cal_file and test_file:
            cal,test=pd.read_csv(cal_file),pd.read_csv(test_file)
            mode='User-supplied research scores; provenance NOT verified'
        elif not cal_file and not test_file:
            cal,test=make_mock(1000,42,'cal'),make_mock(1000,43,'test')
            mode='SYNTHETIC DATA / NOT REAL RESULTS'
        else:
            st.error('Supply both calibration and test files, or neither for mock data.'); st.stop()
        result,routes=evaluate(cal,test,{'max_review':max_review,'max_recapture':max_recapture,'min_routine':min_routine})
        st.subheader(mode)
        st.json(result['fits']['qsart']['thresholds'])
        m=pd.DataFrame(result['test']).T
        st.dataframe(m, use_container_width=True)
        st.bar_chart(m[['routine_rate','review_rate','recapture_rate']])
        st.download_button('Download test routes CSV',routes.to_csv(index=False),'routes.csv','text/csv')
        st.download_button('Download metrics JSON',json.dumps(result,indent=2),'metrics.json','application/json')
        st.info(result['comparison_warning'])
        if result['infeasible']: st.warning('Some comparators were infeasible: '+str(result['infeasible']))
    except Exception as e:
        st.error(str(e))
else:
    st.info('Choose Evaluate research policy. Built-in mock data demonstrate the pipeline without any medical model.')
