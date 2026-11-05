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
    # Nested scenarios/pairs have their own detail view. Mixing their heterogeneous
    # lists with scalar summary cells is not representable by Arrow's table schema.
    present += [column for column in table if column.startswith('metrics.')
                and all(pd.api.types.is_scalar(value) for value in table[column])]
    return table[present]


def show_policy_comparison(metrics):
    """Keep failed allocations distinct from genuine zero-miss outcomes."""
    summary = metrics.get('summary', {})
    left, middle, right = st.columns(3)
    left.metric('Fully evaluated scenarios', summary.get('fully_evaluated_scenarios'))
    middle.metric('Incomplete scenarios', summary.get('incomplete_scenarios'))
    right.metric('Estimable paired comparisons', summary.get('estimable_pairs'))
    st.warning('Development policy cross-fitting on previously inspected selection patients. '
               'The production ordinal gate still requires full review. No validated triage benefit follows.')
    left, right = st.columns(2)
    mode = left.selectbox('Conformal mode', ['ordinal', 'lac', 'no_cp'])
    variant = right.selectbox('Policy variant', ['tuned', 'default'])
    if mode != 'ordinal':
        st.info('Diagnostic ablation only; this uncertainty setting is not deployed.')
    rows = [row for row in metrics.get('scenarios', [])
            if row.get('mode') == mode and row.get('variant') == variant]
    table = pd.json_normalize(rows)
    columns = ['method', 'review_budget', 'recapture_budget', 'status', 'evaluated_folds',
               'untunable_folds', 'evaluation_overload_folds',
               'pooled_metrics.auto_nonreferable_count', 'pooled_metrics.review_count',
               'pooled_metrics.recapture_count', 'pooled_metrics.missed_referable_auto',
               'pooled_metrics.severe_deferred_to_recapture', 'pooled_metrics.proxy_total_cost_per_image']
    if rows:
        labels = {'method': 'Policy', 'review_budget': 'Review budget', 'recapture_budget': 'Recapture budget',
                  'status': 'Status', 'evaluated_folds': 'Evaluated folds', 'untunable_folds': 'Untunable folds',
                  'evaluation_overload_folds': 'Overloaded evaluation folds',
                  'pooled_metrics.auto_nonreferable_count': 'Automatic',
                  'pooled_metrics.review_count': 'Reviews', 'pooled_metrics.recapture_count': 'Recaptures',
                  'pooled_metrics.missed_referable_auto': 'Referable automatic misses',
                  'pooled_metrics.severe_deferred_to_recapture': 'Severe recaptures',
                  'pooled_metrics.proxy_total_cost_per_image': 'Fixed proxy / image'}
        st.dataframe(table[[column for column in columns if column in table]].rename(columns=labels),
                     width='stretch', hide_index=True)
    st.caption('Blank pooled metrics mean incomplete allocation, never zero misses. '
               'Untunable means all six candidates overloaded on the tuning patients; '
               'evaluation overload is a separate held-fold failure. Fixed evaluation prices: '
               'review 0.25, recapture 0.20, plus declared severity penalties.')
    st.caption('Matching budget caps does not imply matching actual workloads. '
               'All 19 comparisons with matched per-fold workloads produced identical actions.')
    with st.expander('Aggregate paired comparisons and provenance'):
        st.json({key: metrics[key] for key in ('evaluation_ledger', 'source_report_sha256',
                                               'paired_bootstrap', 'paired_comparisons') if key in metrics})


def show_aptos_experiment(metrics):
    """Show the study's actual cohorts and all recorded randomization replicates."""
    population = metrics.get('population', {})
    left, middle, right = st.columns(3)
    left.metric('Selection images', metrics.get('images'))
    middle.metric('Conformal calibration representatives', population.get('conformal_representative_images'))
    right.metric('Temperature-fitting images', population.get('temperature_images'))
    st.write('Development candidate selected: '+str(metrics.get('summary', {}).get('selected_backbone', 'Unreported')))
    st.caption('The deployed DeepDRiD models remain selected. These APTOS results use '
               'selection images that also chose classifier checkpoints and architecture.')
    st.warning('Fingerprint groups do not establish patient independence. '
               'APTOS provides no quality annotations for this study; useful triage and external validity remain unproven.')
    candidates = pd.json_normalize(metrics.get('candidates', []))
    labels = {'backbone': 'Candidate', 'selected_epoch': 'Selected epoch',
              'selection_metrics.accuracy': 'Accuracy', 'selection_metrics.macro_f1': 'Macro-F1',
              'selection_metrics.quadratic_weighted_kappa': 'Quadratic kappa', 'selection_metrics.nll': 'NLL'}
    if not candidates.empty:
        st.dataframe(candidates[[key for key in labels if key in candidates]].rename(columns=labels),
                     width='stretch', hide_index=True)
    left, right = st.columns(2)
    alpha = left.selectbox('Reported alpha', [.10, .05])
    cohort = right.selectbox('Reported population', ['All selection images', 'One representative per fingerprint group'])
    key = ('all_selection_images' if cohort == 'All selection images' else 'selection_group_representatives')
    records = []
    for row in metrics.get('comparisons', []):
        if row.get('alpha') != alpha:
            continue
        raw = row.get('raw', {}).get(key, {})
        enlarged = row.get('argmax_fallback', {}).get(key, {})
        records.append({'Method': row.get('method'), 'Randomization seed': row.get('randomization_seed'),
            'Images': enlarged.get('n_images'), 'Raw coverage': raw.get('marginal_coverage'),
            'Raw empty sets': raw.get('empty_set_count'),
            'Coverage with argmax fallback': enlarged.get('marginal_coverage'),
            'Mean set size with fallback': enlarged.get('mean_set_size')})
    if records:
        st.dataframe(pd.DataFrame(records), width='stretch', hide_index=True)
    st.caption('Empirical inclusion on development images; alpha is a declared setting, '
               'not a verified coverage guarantee. All five randomization seeds are retained. '
               'Temperature fitting and conformal representatives come from separate calibration groups.')
    with st.expander('Grading and calibration details'):
        st.json({key: metrics[key] for key in ('grading', 'grading_before_temperature',
                                               'calibration', 'calibration_unit_definition') if key in metrics})


def show_aptos_finetune(metrics):
    runs = metrics.get('runs', [])
    if not runs:
        st.info('No completed fine-tuning runs are recorded.')
        return
    primary = metrics.get('summary', {}).get('primary_training_seed')
    st.caption(f'Primary training seed was fixed at {primary} before execution. '
               'Training-seed variation is conditional on a shared warm start and reused selection data.')
    rows = []
    baseline = metrics.get('baseline', {}).get('selection_metrics', {})
    rows.append({'Run': 'Frozen-feature baseline', **{key: baseline.get(key) for key in ('accuracy', 'macro_f1', 'nll')}})
    for run in runs:
        grading = run.get('grading_before_temperature', run.get('selection_metrics', {}))
        rows.append({'Run': f'Training seed {run["training_seed"]}',
                     **{key: grading.get(key) for key in ('accuracy', 'macro_f1', 'nll')}})
    st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)
    seed = st.selectbox('Recorded classifier training seed', [run['training_seed'] for run in runs])
    st.caption('This control displays a recorded run; it does not select or deploy a winning seed.')
    run = next(row for row in runs if row['training_seed'] == seed)
    view = {**run, 'images': metrics.get('images'), 'population': metrics.get('population', {}),
            'summary': {'selected_backbone': f'EfficientNet-B0, recorded training seed {seed}'},
            'comparisons': [row for row in metrics.get('comparisons', []) if row['model_training_seed'] == seed]}
    show_aptos_experiment(view)
    with st.expander('Across-seed summaries and descriptive paired changes'):
        st.json({'summary': metrics.get('summary', {}),
                 'paired_changes': {str(row['training_seed']): row.get('paired_baseline_changes') for row in runs}})


def show_aptos_optimization(metrics):
    runs = metrics.get('runs', [])
    if not runs:
        st.info('No completed source-optimization run is recorded.')
        return
    st.caption('Only the fixed primary seed 2026 was extended to a maximum of 15 epochs. '
               'No new across-seed variance is estimated. IDRiD is consumed and was not used in this run.')
    run = runs[0]
    rows = []
    for label, grading in [('Frozen-feature baseline', metrics.get('baseline', {}).get('selection_metrics', {})),
                           ('Four-epoch primary', metrics.get('prior_primary', {}).get('selection_metrics', {})),
                           ('Extended primary', run.get('grading_before_temperature', {}))]:
        rows.append({'Run': label, **{key: grading.get(key) for key in ('accuracy', 'macro_f1', 'nll')}})
    st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)
    st.caption('Selected epoch '+str(metrics.get('summary', {}).get('selected_epoch'))+
               '; stop reason: '+str(metrics.get('summary', {}).get('stop_reason'))+
               '. Selection/calibration data are reused; these results do not establish convergence or fresh-test performance.')
    view = {**metrics, 'summary': {'selected_backbone': 'EfficientNet-B0, extended primary seed 2026'}}
    show_aptos_experiment(view)
    with st.expander('Training history and paired development changes'):
        st.json({'history': run.get('history', []), 'paired_changes': metrics.get('paired_comparisons', {})})


def show_quality_adaptation(metrics):
    runs = metrics.get('runs', [])
    if not runs:
        st.info('No completed quality-adaptation study is recorded.')
        return
    st.caption('Fixed primary seed 2026; all three seeds and both matched arms are retained. '
               'Each run used 15 epochs. These reused development results do not select a production model.')
    labels = {'quality_grade': 'Overall quality', 'clarity_class': 'Clarity',
              'field_definition_class': 'Field definition', 'artifact_class': 'Artifact'}
    st.dataframe(pd.DataFrame([{'Seed': row['seed'], 'Arm': row['arm'],
        'Selected epoch': row['selected_epoch'], 'Mean four-head macro-F1': row['mean_head_macro_f1'],
        **{labels[head]+' F1': row['selected_metrics'][head]['macro_f1'] for head in labels}}
        for row in runs]), width='stretch', hide_index=True)
    seed = st.selectbox('Recorded quality seed', sorted({row['seed'] for row in runs}), index=0)
    head = st.selectbox('Recorded quality head', list(labels), format_func=labels.get)
    selected = [row for row in runs if row['seed'] == seed]
    rows = []
    warm = selected[0]['warm_metrics'][head]
    rows.append({'Reference / arm': 'Legacy warm start', **{key: warm[key] for key in ('accuracy', 'macro_f1', 'nll')}})
    for row in selected:
        values = row['selected_metrics'][head]
        rows.append({'Reference / arm': row['arm'], **{key: values[key] for key in ('accuracy', 'macro_f1', 'nll')}})
    majority = metrics.get('majority_references', {}).get(head, {})
    if majority:
        rows.append({'Reference / arm': 'Fitting-majority class '+str(majority['class']),
                     'accuracy': majority['accuracy'], 'macro_f1': majority['macro_f1'], 'nll': None})
    st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)
    supports = metrics.get('population', {}).get('class_supports', {}).get(head, {})
    classes = range(len(warm['support']))
    st.dataframe(pd.DataFrame([{'Class': index, 'Fitting images': supports.get('train', [])[index],
        'Selection images': warm['support'][index],
        **{row['arm']+' recall': row['selected_metrics'][head]['recall'][index] for row in selected}}
        for index in classes]), width='stretch', hide_index=True)
    st.caption('Poor quality is class 0. The lowest field-definition class has only one selection image. '
               'Warm-start changes also include a different selector and fitting procedure; '
               'the matched adapted-minus-frozen contrast measures added backbone adaptation.')
    with st.expander('All paired intervals and conditional seed summaries'):
        st.json({'summary': metrics.get('summary', {}), 'paired_changes': metrics.get('paired_changes', [])})
    st.caption('Intervals resample whole components and condition on selected checkpoints. '
               'They are descriptive, without reselection or multiplicity correction. Production weights are unchanged.')


def show_quality_calibration_stress(metrics):
    runs = metrics.get('runs', [])
    if not runs:
        st.info('No completed quality calibration/stress study is recorded.')
        return
    st.caption('Temperatures use a separate calibration role, historically used by older models. '
               'Selection data are reused development evidence. All six models stay fixed; production is unchanged.')
    heads = {'quality_grade': 'Overall quality', 'clarity_class': 'Clarity',
             'field_definition_class': 'Field definition', 'artifact_class': 'Artifact'}
    rows = []
    for run in runs:
        clean = run['clean_metrics']['validation']
        rows.append({'Model': run['model_id'], 'Training seed': run['training_seed'], 'Arm': run['arm'],
            'Raw mean head NLL': sum(clean['raw'][h]['nll'] for h in heads)/4,
            'Temperature mean head NLL': sum(clean['temperature'][h]['nll'] for h in heads)/4,
            'Paired NLL change': run['calibration_changes']['mean_head_nll']['difference']})
    st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)
    ids = [row['model_id'] for row in runs]
    chosen = st.selectbox('Recorded calibrated quality model', ids,
                          index=ids.index(metrics.get('primary_model_id', 'quality_seed2026_adapted')))
    run = next(row for row in runs if row['model_id'] == chosen)
    head = st.selectbox('Clean quality head', list(heads), format_func=heads.get)
    role = st.selectbox('Recorded clean role', ['validation', 'calibration'])
    st.metric('Recorded head temperature', f"{run['temperatures'][head]:.5f}")
    clean = run['clean_metrics'][role]
    st.dataframe(pd.DataFrame([{'Probabilities': state,
        **{key: clean[state][head][key] for key in ('accuracy', 'macro_f1', 'nll', 'ece_10', 'brier')}}
        for state in ('raw', 'temperature')]), width='stretch', hide_index=True)
    support = clean['raw'][head]['support']
    st.dataframe(pd.DataFrame([{'Class': index, 'Clean images': count,
        'Raw recall': clean['raw'][head]['recall'][index],
        'Temperature recall': clean['temperature'][head]['recall'][index]}
        for index, count in enumerate(support)]), width='stretch', hide_index=True)
    st.caption('Calibration-role metrics describe fitting data. Positive scalar temperatures preserve class '
               'predictions and do not repair missing rare classes. Poor quality is class 0.')
    state = st.selectbox('Recorded probability state', ['temperature', 'raw'])
    response = st.selectbox('Recorded stress response', ['poor_probability', 'expected_clarity',
                                                      'expected_field_definition', 'expected_artifact'])
    stress = []
    for row in run['stress']:
        interval = row[state]['changes'][response]
        stress.append({'Condition': row['condition'], 'Images': row['images'],
            'Mean score': row[state]['levels'][response], 'Change from clean': interval['difference'],
            'Lower 95%': interval['lower_95'], 'Upper 95%': interval['upper_95']})
    st.dataframe(pd.DataFrame(stress), width='stretch', hide_index=True)
    st.caption('Altered inputs have no new quality labels: this table measures score response, not '
               'defect-detection accuracy. Expected class indices are ordinal summaries, not physical quality scales.')
    with st.expander('All paired calibration and adaptation-response intervals'):
        st.json({'calibration_changes': run['calibration_changes'],
                 'response_contrasts': metrics.get('response_contrasts', []),
                 'bootstrap': metrics.get('bootstrap', {}), 'summary': metrics.get('summary', {})})
    st.caption('Intervals retain whole components and condition on fitted temperatures and fixed models. '
               'No refitting, reselection or multiplicity correction; no production replacement is selected here.')


def show_idrid_data_readiness(metrics):
    st.info('External cohort reserved at audit time. This data-integrity record contains '
            'no model predictions. Later model outcomes, if available, have a separate external-evaluation record.')
    provenance = metrics.get('source_provenance_status', 'Unreported')
    st.caption('Download provenance: '+provenance)
    if provenance != 'official_download_user_confirmed':
        st.warning('The local package download origin is not yet confirmed.')
    confirmation = metrics.get('source_confirmation', {})
    if confirmation.get('status') == 'official_origin_user_confirmed':
        st.caption('Official IDRiD/IEEE DataPort origin confirmed by the user on '+
                   confirmation['confirmed_at_utc']+'. This receipt does not verify an official archive checksum.')
    left, middle, right = st.columns(3)
    left.metric('Images decoded', metrics.get('images_decoded', 0))
    middle.metric('Cross-source candidate pairs', metrics.get('cross_source_candidate_pairs', 0))
    right.metric('External predictions', 0 if metrics.get('model_inference_performed') is False else 'Unreported')
    if metrics.get('independence_claim_blocked_by_detected_overlap'):
        st.warning('Detected overlap candidates require review before an independent-cohort claim.')
    rows = []
    for split, part in metrics.get('official_partitions', {}).items():
        rows.append({'Original partition': split, 'Images': part['images'],
                     **{f'DR grade {grade}': part['dr_grade_counts'].get(str(grade), 0) for grade in range(5)}})
    st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)
    st.caption('Both original partitions are external holdout for QSART. '
               'DME risk is a separate source annotation and is not an image-quality label. '
               'Serial names and fingerprint components are not patient identifiers.')
    with st.expander('Data-audit details'):
        st.json(metrics)


def show_idrid_external(metrics):
    models = metrics.get('models', [])
    summary = metrics.get('summary', {})
    populations = list(summary.get('population_counts', {}))
    if not models or not populations:
        st.info('No completed locked external evaluation is recorded.')
        return
    st.info('External outcomes are consumed. Later reuse is not fresh confirmation; '
            'these diagnostics do not establish patient independence or clinical validity.')
    st.caption('Primary classifier '+str(summary.get('primary_model'))+' and primary population '+
               str(summary.get('primary_population'))+' were fixed before external outcomes. '
               'Controls display saved results and do not fit or select a model.')
    population = st.selectbox('Recorded external population', populations,
                             index=populations.index(summary['primary_population']))
    table = []
    for model in models:
        grade = model['grading'][population]
        table.append({'Model': model['model_id'], 'Images': grade['n'], 'Accuracy': grade['accuracy'],
            'Macro-F1': grade['macro_f1'], 'QWK': grade['quadratic_weighted_kappa'],
            'Grade-3 recall': grade['per_class'][3]['recall'], 'Grade-4 recall': grade['per_class'][4]['recall']})
    st.dataframe(pd.DataFrame(table), width='stretch', hide_index=True)
    identifiers = [model['model_id'] for model in models]
    chosen = st.selectbox('Recorded external classifier', identifiers, index=identifiers.index(summary['primary_model']))
    alpha = st.selectbox('External reported alpha', [.1, .05])
    rows = []
    for row in metrics.get('comparisons', []):
        if row['model_id'] != chosen or row['alpha'] != alpha:
            continue
        raw, fallback = row['raw'][population], row['argmax_fallback'][population]
        rows.append({'Method': row['method'], 'Uniform seed': row['randomization_seed'], 'Images': raw['n_images'],
            'Raw inclusion': raw['marginal_coverage'], 'Fallback inclusion': fallback['marginal_coverage'],
            'Raw mean size': raw['mean_set_size'], 'Fallback mean size': fallback['mean_set_size'],
            'Raw empties': raw['empty_set_count'], 'Grade-3 fallback inclusion': fallback['per_class']['3']['coverage'],
            'Grade-4 fallback inclusion': fallback['per_class']['4']['coverage']})
    st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)
    st.caption('Temperatures and all thresholds were fitted on APTOS and reused unchanged. '
               'Uniform seeds are distinct from classifier-training seeds. Source-grade ambiguities remain; '
               'annotation envelopes are hypothetical sensitivity analyses, not corrected clinical truth.')
    model = next(row for row in models if row['model_id'] == chosen)
    with st.expander('Annotation sensitivity and descriptive paired changes'):
        st.json({'grading': model['grading'][population], 'before_temperature': model['grading_before_temperature'][population],
            'paired_changes': [row for row in metrics.get('paired_baseline_changes', [])
                              if row['model_id'] == chosen and row['population'] == population],
            'conditional_seed_summary': summary.get('grading_across_finetuning_seeds', {}).get(population)})


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
                if chosen == 'policy_comparison_v1':
                    show_policy_comparison(detail.get('metrics', {}))
                elif chosen == 'aptos_development_v1':
                    show_aptos_experiment(detail.get('metrics', {}))
                elif chosen == 'aptos_finetune_v1':
                    show_aptos_finetune(detail.get('metrics', {}))
                elif chosen == 'aptos_optimization_v2':
                    show_aptos_optimization(detail.get('metrics', {}))
                elif chosen == 'quality_adaptation_v1':
                    show_quality_adaptation(detail.get('metrics', {}))
                elif chosen == 'quality_calibration_stress_v1':
                    show_quality_calibration_stress(detail.get('metrics', {}))
                elif chosen == 'idrid_data_v1':
                    show_idrid_data_readiness(detail.get('metrics', {}))
                elif chosen == 'idrid_external_v1':
                    show_idrid_external(detail.get('metrics', {}))
                else:
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
