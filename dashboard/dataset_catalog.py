"""Render dataset provenance and role boundaries from the local research API."""

import json

import pandas as pd
import streamlit as st


STATUS_LABELS = {'audited': 'Audited records', 'metadata_only': 'Metadata only', 'unavailable': 'Evidence unavailable'}
ROLE_LABELS = {
    'fitting': 'Model fitting', 'selection': 'Model selection', 'calibration': 'Calibration',
    'reserved': 'Reserved for evaluation', 'consumed': 'Evaluated', 'excluded': 'Excluded',
    'not_imported': 'Outside the import', 'metadata_only': 'External annotations only',
}


def population_table(entry, populations):
    table = pd.DataFrame([{
        'Population': row['name'], 'QSART use': ROLE_LABELS[row['role']],
        entry['observation_label']: row['observations'], entry['group_label']: row['groups'],
        'Transformed mirror matches': row['transformed_matches'],
    } for row in populations])
    return table.dropna(axis=1, how='all').convert_dtypes()


def label_table(populations, field, codes):
    return pd.DataFrame([{'Population': row['name'], **{label: row[field].get(code, 0) for code, label in codes.items()}}
                         for row in populations if row[field]])


def show_dataset_catalog(payload):
    st.subheader('Dataset Catalog')
    st.caption('These are saved audit records. The catalog works independently of model readiness and local image storage.')
    datasets = {entry['name']: entry for entry in payload.get('datasets', [])}
    if not datasets:
        st.info('Dataset records are unavailable from the research API.')
        return

    entry = datasets[st.selectbox('Dataset', list(datasets), key='dataset_catalog_name')]
    st.subheader(entry['name'])
    st.markdown(f"**{STATUS_LABELS[entry['status']]}** · {entry['purpose']}")
    st.write(entry['access'])
    st.link_button('Dataset source', entry['source_url'])
    for note in entry['notes']:
        st.caption(note)
    if not entry['partitions']:
        return

    st.markdown('#### Source and QSART splits')
    st.dataframe(population_table(entry, entry['partitions']), width='stretch', hide_index=True)
    if entry['study_populations']:
        st.markdown('#### Calibration and evaluation subsets')
        st.caption('These rows can overlap the splits or one another. Counts describe each named subset.')
        st.dataframe(population_table(entry, entry['study_populations']), width='stretch', hide_index=True)

    populations = entry['partitions'] + entry['study_populations']
    grades = label_table(populations, 'dr_grade_counts', {str(code): f'Grade {code}' for code in range(5)})
    quality = label_table(populations, 'quality_counts', entry['quality_codes'])
    if not grades.empty:
        st.markdown('#### DR grade counts')
        st.dataframe(grades, width='stretch', hide_index=True)
    if not quality.empty:
        st.markdown('#### Source quality counts')
        st.dataframe(quality, width='stretch', hide_index=True)

    with st.expander('Evidence files and SHA-256 digests'):
        st.dataframe(pd.DataFrame(entry['evidence']).rename(columns={'report': 'Report', 'sha256': 'SHA-256'}),
                     width='stretch', hide_index=True)
    st.download_button('Export dataset evidence', json.dumps(entry, indent=2),
                       f"qsart_{entry['dataset_id']}_data.json", 'application/json')
