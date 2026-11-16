"""Explorer views for saved quality and DeepDRiD screening studies."""

import pandas as pd
import streamlit as st

from qsart.triage_v2 import capacity


SET_C_MODES = {'ordinal': 'Deployed ordinal', 'lac': 'LAC diagnostic', 'no_cp': 'No-conformal diagnostic'}
SET_C_ACTIONS = {'AUTO_NONREFERABLE': 'Fully automatic', 'RECAPTURE': 'Recapture',
                 'SPECIALIST_REVIEW': 'Specialist review'}


def show_set_c_validation(metrics):
    if not metrics.get('scenarios'):
        st.info('Audited Set-C results are unavailable.')
        return
    st.subheader('DeepDRiD Set-C patient validation')
    st.caption('This held-out split comes from the same release. Patient grades use the worse eye. '
               'Patient actions summarize the four photo recommendations.')
    prediction = metrics['prediction_metrics']
    left, middle, right = st.columns(3)
    left.metric('Patients', metrics['patients'])
    middle.metric('Photos', metrics['images'])
    right.metric('Patient referable AUROC', f"{prediction['patient_grading']['referable']['auroc']:.3f}")
    st.dataframe(pd.DataFrame([
        {'Prediction set': label, 'Photos covered': prediction['conformal'][mode]['covered_images'],
         'Patients with all views covered': prediction['conformal'][mode]['covered_patients'],
         'Mean set size': prediction['conformal'][mode]['mean_set_size']}
        for mode, label in [('ordinal', 'Ordinal'), ('lac', 'LAC')]
    ]), hide_index=True, width='stretch')

    mode = st.selectbox('Set-C policy mode', list(SET_C_MODES), format_func=SET_C_MODES.__getitem__)
    rows = []
    for row in metrics['scenarios']:
        if row['mode'] != mode:
            continue
        review, recapture = row['budget']
        outcomes = row['outcomes']
        counts = outcomes['patient_actions'] if outcomes else {}
        images = outcomes['image_actions'] if outcomes else {}
        rows.append({'Method': row['method'], 'Image review cap': capacity(review, metrics['images']),
                     'Image recapture cap': capacity(recapture, metrics['images']),
                     'Allocation': 'Evaluated' if outcomes else 'Overload',
                     'Additional image review slots': row['diagnostics']['review_shortfall_after_recapture']
                     if row['diagnostics'] else None, 'Image reviews': images.get('SPECIALIST_REVIEW'),
                     'Image recaptures': images.get('RECAPTURE'),
                     **{label: counts.get(action) for action, label in SET_C_ACTIONS.items()},
                     'Severe photo misses': outcomes['missed_severe_images'] if outcomes else None,
                     'Referable patients cleared': outcomes['fully_automatic_referable_patients'] if outcomes else None,
                     'Additional patient review slots': row['patient_capacity']['review_shortfall']})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width='stretch')
    st.caption('The patient shortfall uses pointwise recommendations and patient caps. '
               'Recommendations for review and recapture have no observed follow-up outcome.')

    pairs = [row for row in metrics['comparisons'] if row['mode'] == mode]
    exact = sum(row['matched_image_workload'] is True and row['matched_patient_workload'] is True for row in pairs)
    identical = sum(row['identical_image_actions'] is True for row in pairs)
    st.subheader('Observed workload comparisons')
    st.write(f'{exact} of {len(pairs)} pairs matched both image and patient workloads. '
             f'{identical} pairs made identical image recommendations.')
    with st.expander('Paired results'):
        pairs_table = []
        for row in pairs:
            review, recapture = row['budget']
            effect = (row['intervals'] or {}).get('patient_weighted_missed_referral_per_patient')
            pairs_table.append({'Comparator': row['reference'].split(':')[1], 'Caps': f'{review:.0%} / {recapture:.0%}',
                                'Both arms feasible': row['status'] == 'comparison',
                                'Image workload match': row['matched_image_workload'],
                                'Patient workload match': row['matched_patient_workload'],
                                'Identical actions': row['identical_image_actions'],
                                'Weighted miss difference': effect['difference_candidate_minus_reference'] if effect else None,
                                '95% lower': effect['ci95'][0] if effect else None,
                                '95% upper': effect['ci95'][1] if effect else None})
        st.dataframe(pd.DataFrame(pairs_table), hide_index=True, width='stretch')
    st.caption('Paired intervals describe the measured patients and their fixed actions.')


POPULATION_NAMES = {
    'main': 'Main cohort',
    'representative': 'One photo per fingerprint component',
    'unanimous': 'Unanimous ratings',
}
GATE_NAMES = {
    'ordinal_projected': 'Projected ordinal',
    'lac_projected': 'Projected LAC',
    'binary_lac': 'Binary LAC',
}


def show_mshf_quality(metrics):
    populations = metrics.get('populations', {})
    choices = [name for name in POPULATION_NAMES if name in populations]
    if not choices:
        st.info('MSHF population results are unavailable.')
        return

    st.subheader('MSHF overall image quality')
    population = st.selectbox('MSHF population', choices, format_func=POPULATION_NAMES.__getitem__)
    view = populations[population]
    model = metrics['model']
    st.caption(f"The model flags poor quality when P(poor) exceeds {model['poor_probability_threshold']:.2f}. "
               f"The probability comparison uses the stored DeepDRiD temperature of {model['temperature']:.6f}.")
    results = []
    calibration = []
    for source, cell in [('Pooled', view['overall']), *view['groups'].items()]:
        calibrated = cell['calibrated'] or {}
        confusion = calibrated.get('confusion_matrix')
        results.append({
            'Source': source,
            'Poor photos': cell['poor'],
            'Good photos': cell['good'],
            'Poor flagged': confusion[0][0] if confusion is not None else None,
            'Good flagged': confusion[1][0] if confusion is not None else None,
            'Poor sensitivity': calibrated.get('poor_sensitivity'),
            'Good specificity': calibrated.get('good_specificity'),
            'Accuracy': calibrated.get('accuracy'),
            'Always-good accuracy': cell['always_good_accuracy'],
            'Poor AUROC': calibrated.get('poor_auroc'),
            'Poor AP': calibrated.get('poor_average_precision'),
        })
        raw = cell['raw'] or {}
        calibration.append({
            'Source': source,
            'Raw log loss': raw.get('nll'),
            'Frozen log loss': calibrated.get('nll'),
            'Raw Brier': raw.get('brier'),
            'Frozen Brier': calibrated.get('brier'),
            'Raw ECE': raw.get('ece'),
            'Frozen ECE': calibrated.get('ece'),
        })
    st.dataframe(pd.DataFrame(results), hide_index=True, width='stretch', column_config={
        'Poor sensitivity': st.column_config.ProgressColumn(min_value=0., max_value=1., format='percent'),
        'Good specificity': st.column_config.ProgressColumn(min_value=0., max_value=1., format='percent'),
        'Accuracy': st.column_config.NumberColumn(format='percent'),
        'Always-good accuracy': st.column_config.NumberColumn(format='percent'),
        'Poor AUROC': st.column_config.NumberColumn(format='%.4f'),
        'Poor AP': st.column_config.NumberColumn(format='%.4f'),
    })
    st.caption('Single-class groups have blank cells where a metric requires the absent class.')
    st.subheader('Raw and temperature-scaled probabilities')
    st.dataframe(pd.DataFrame(calibration), hide_index=True, width='stretch',
                 column_config={key: st.column_config.NumberColumn(format='%.4f')
                                for key in calibration[0] if key != 'Source'})
    st.caption('ECE is expected calibration error based on the confidence of the predicted class.')

    with st.expander('Annotation audit and provenance'):
        inventory = metrics['inventory']
        overlap = metrics['overlap']
        st.caption(f"The release contains {inventory['images_decoded']} decoded photos. "
                   f"The overlap rules excluded {overlap['excluded_images']} from the scored populations.")
        annotations = [{'Source': source, 'Photos': group['images'], 'Unanimous': group['unanimous'],
                        'Cached summary disagreements': group['summary_disagreements']}
                       for source, group in inventory['groups'].items()]
        st.dataframe(pd.DataFrame(annotations), hide_index=True, width='stretch')
        st.json({key: metrics[key] for key in ('archive', 'model', 'overlap', 'source_commit', 'protocol_sha256')})


def show_referability_gate(metrics):
    fits = metrics.get('fits', [])
    pairs = [tuple(pair) for pair in metrics.get('budget_pairs', [])]
    if not fits or not pairs:
        st.info('Referability gate results are unavailable.')
        return

    st.subheader('DeepDRiD referability gates')
    st.caption(f"The comparison reuses {metrics['images']} development selection photos "
               f"from {metrics['patients']} patients.")
    alpha = st.selectbox('Conformal alpha', sorted({fit['alpha'] for fit in fits}, reverse=True))
    selected = [fit for fit in fits if fit['alpha'] == alpha]
    sets = []
    pointwise = []
    for fit in selected:
        method = GATE_NAMES[fit['method']]
        inclusion = fit['set_metrics']
        sets.append({
            'Method': method,
            'Image inclusion': inclusion['marginal_coverage'],
            'Patient all-views inclusion': inclusion['patient_all_views_coverage'],
            'Grade 3 referability inclusion': inclusion['by_true_grade']['3']['coverage'],
            'Mean set size': inclusion['mean_set_size'],
            'Non-referable singleton sets': inclusion['nonreferable_singleton_count'],
        })
        actions = fit['pointwise_metrics']
        pointwise.append({
            'Method': method,
            'Automatic': actions['auto_nonreferable_count'],
            'Recapture': actions['recapture_count'],
            'Review': actions['review_count'],
            'Severe recaptures': actions['severe_deferred_to_recapture'],
        })
    st.dataframe(pd.DataFrame(sets), hide_index=True, width='stretch', column_config={
        'Image inclusion': st.column_config.NumberColumn(format='percent'),
        'Patient all-views inclusion': st.column_config.NumberColumn(format='percent'),
        'Grade 3 referability inclusion': st.column_config.NumberColumn(format='percent'),
        'Mean set size': st.column_config.NumberColumn(format='%.3f'),
    })
    st.caption('Inclusion refers to the true binary referability label. '
               'Patient all-views inclusion requires every photo of that patient to contain its label.')
    st.markdown('**Pointwise recommendations before applying cohort budgets**')
    st.dataframe(pd.DataFrame(pointwise), hide_index=True, width='stretch')

    st.subheader('Budgeted allocations')
    budget = st.selectbox('Cohort capacity', pairs, index=pairs.index((.90, .40)) if (.90, .40) in pairs else 0,
                          format_func=lambda pair: f'{pair[0]:.0%} review / {pair[1]:.0%} recapture')
    rows = []
    for fit in selected:
        allocation = next((entry for entry in fit['allocations']
                           if (entry['review_budget'], entry['recapture_budget']) == budget), None)
        if allocation is None:
            continue
        diagnostics = allocation['diagnostics']
        outcomes = allocation['metrics'] or {}
        rows.append({
            'Method': GATE_NAMES[fit['method']],
            'Status': allocation['status'].capitalize(),
            'Mandatory review': diagnostics['needed_review'],
            'Review cap': diagnostics['review_capacity'],
            'Recapture cap': diagnostics['recapture_capacity'],
            'Automatic': outcomes.get('auto_nonreferable_count'),
            'Recapture': outcomes.get('recapture_count'),
            'Review': outcomes.get('review_count'),
            'Referable auto misses': outcomes.get('missed_referable_auto'),
            'Severe recaptures': outcomes.get('severe_deferred_to_recapture'),
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, width='stretch')
    st.caption('Overload leaves allocation outcomes blank. Automatic recommendations are provisional clearances.')
    with st.expander('Calibration and policy settings'):
        st.json({key: metrics[key] for key in ('calibration', 'policy', 'input_sha256',
                                               'source_commit', 'protocol_sha256')})
