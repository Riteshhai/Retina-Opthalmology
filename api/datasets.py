"""Expose saved dataset evidence and split roles from allowlisted aggregate reports."""

import hashlib
import json
from pathlib import Path

from api.schemas import DatasetCatalog, DatasetCatalogEntry, DatasetEvidence, DatasetPopulation
from qsart.data.idrid_provenance import idrid_source_confirmation


class Reports:
    def __init__(self, root):
        self.root = root
        self.evidence = {}

    def read(self, name):
        content = (self.root / name).read_bytes()
        data = json.loads(content)
        if not isinstance(data, dict):
            raise ValueError('Expected an aggregate report object')
        self.evidence[name] = hashlib.sha256(content).hexdigest()
        return data

    def audited(self, name, digest_field, *, status_field='audit_status'):
        data = self.read(name)
        receipt = self.read((Path(name).parent / 'independent_audit.json').as_posix())
        if receipt[status_field] != 'passed' or receipt[digest_field] != self.evidence[name]:
            raise ValueError('Dataset report does not match its independent audit')
        return data


def population(name, role, values, *, group_key=None, dr_key='dr_grade_counts', matches=None):
    row = DatasetPopulation(name=name, role=role, observations=values['images'],
                            groups=values[group_key] if group_key else None,
                            dr_grade_counts=values.get(dr_key, {}), quality_counts=values.get('quality_counts', {}),
                            transformed_matches=matches)
    if row.observations is None:
        raise ValueError('The audited population count is missing')
    for counts in (row.dr_grade_counts, row.quality_counts):
        if counts and sum(counts.values()) != row.observations:
            raise ValueError('Label counts do not cover the displayed population')
    if row.transformed_matches is not None and row.transformed_matches > row.observations:
        raise ValueError('Mirror coverage exceeds source annotation count')
    return row


def aptos(reports):
    report = reports.audited('aptos_v1/data_audit.json', 'data_audit_sha256')
    if (report['status'] != 'verified' or report['competition_test_extracted'] is not False
            or report['reserved_test_inference_performed'] is not False):
        raise ValueError('APTOS source or test reservation differs from the frozen audit')
    partitions = [population(name, role, report['partitions'][key], group_key='split_groups')
                  for key, name, role in (
                      ('train', 'QSART training', 'fitting'), ('validation', 'QSART selection', 'selection'),
                      ('calibration', 'QSART calibration', 'calibration'), ('test', 'QSART internal test', 'reserved'))]
    partitions.extend([
        population('Conflicting-label quarantine', 'excluded', {
            'images': report['quarantined_images'], 'split_groups': report['conflicting_components'],
            'dr_grade_counts': report['quarantined_dr_grade_counts'],
        }, group_key='split_groups'),
        DatasetPopulation(name='Kaggle competition test', role='not_imported'),
    ])
    calibration_roles = [population(name, 'calibration', report['calibration_roles'][key], group_key='groups')
                         for key, name in (('temperature', 'Temperature fitting'), ('conformal', 'Conformal calibration'))]
    return partitions, calibration_roles


def deepdrid(reports):
    summary = reports.read('development_v1/data_summary.json')
    confirmation = reports.read('confirmation_v1/evaluation.json')
    online = reports.audited('deepdrid_online_readiness_v1/evaluation.json', 'evaluation_sha256')
    coverage = online['prior_manifest_coverage']
    if (summary['cross_cohort_audit']['passed'] is not True or online['model_inference_performed'] is not False
            or any(summary[key]['manifest_sha256'] != coverage[key]['sha256'] for key in (
                'development_v1', 'confirmation_v1'))
            or confirmation['confirmation_manifest_sha256'] != coverage['confirmation_v1']['sha256']
            or confirmation['images'] != summary['confirmation_v1']['images']
            or confirmation['patients'] != summary['confirmation_v1']['patients']):
        raise ValueError('DeepDRiD cohort records do not match their frozen bindings')
    development = summary['development_v1']['partitions']
    partitions = [population(name, role, development[key], group_key='subjects')
                  for key, name, role in (
                      ('train', 'Set-A fitting', 'fitting'), ('validation', 'Set-A selection', 'selection'),
                      ('calibration', 'Set-A calibration', 'calibration'))]
    counts = online['population']
    partitions.extend([
        population('Set-B historical confirmation', 'consumed', summary['confirmation_v1']['partitions']['holdout'],
                   group_key='subjects'),
        population('Set-C online challenge', 'reserved', {
            'images': counts['images'], 'patients': counts['patients'],
            'dr_grade_counts': online['labels']['image_grade_counts'],
            'quality_counts': online['labels']['overall_quality_counts'],
        }, group_key='patients'),
    ])
    return partitions, [
        DatasetPopulation(name='Set-C eligible for locked evaluation', role='reserved',
                          observations=counts['eligible_images'], groups=counts['eligible_patients']),
        DatasetPopulation(name='Set-C overlap exclusion', role='excluded',
                          observations=counts['images'] - counts['eligible_images'], groups=counts['excluded_patients']),
    ]


def idrid(reports):
    source = 'idrid_data_v1/data_audit.json'
    review = 'idrid_review_v1/data_review.json'
    origin = 'idrid_data_v1/source_confirmation.json'
    report = reports.audited(source, 'data_audit_sha256')
    confirmation = idrid_source_confirmation(reports.root / source)
    reports.read(origin)
    reviewed = reports.audited(review, 'data_review_sha256')
    outcomes = reports.audited('idrid_external_v1/evaluation.json', 'evaluation_sha256')
    locked_inputs = outcomes['input_bindings']['config']['input_sha256']
    if (outcomes['external_outcomes_consumed'] is not True
            or confirmation.get('source_provenance_status') != 'official_download_user_confirmed'
            or confirmation['source_confirmation']['receipt_sha256'] != reports.evidence[origin]
            or any(locked_inputs[f'reports/{name}'] != reports.evidence[name] for name in (source, origin, review))):
        raise ValueError('IDRiD provenance, overlap review and consumption records do not agree')
    partitions = [population(name, 'consumed', report['official_partitions'][key]) for key, name in (
        ('official_train', 'Source training split · QSART external'), ('official_test', 'Source test split · QSART external'))]
    study_populations = [population(name, 'consumed', reviewed['summary']['populations'][key],
                                    group_key='fingerprint_components')
                         for key, name in (('full_source', 'Full release'), ('cross_source_filtered', 'Overlap-filtered'),
                                           ('cross_source_filtered_representatives', 'One representative per component'))]
    return partitions, study_populations


def eyeq(reports):
    report = reports.audited('eyeq_data_v1/data_audit.json', 'source_report_sha256', status_field='status')
    mirror = reports.audited('eyeq_hf_v1/acquisition.json', 'acquisition_report_sha256', status_field='status')
    if (report['readiness']['metadata_verified'] is not True or mirror['ready_for_model_evaluation'] is not False
            or report['readiness']['ready_for_model_evaluation'] is not False):
        raise ValueError('EyeQ readiness records do not describe the audited metadata-only study')
    partitions = [population(f'Author {key} · external annotations', 'metadata_only',
                             report['annotation_profile']['original_splits'][key], group_key='source_subject_keys',
                             dr_key='dr_counts', matches=mirror['original_author_splits'][key]['matched'])
                  for key in ('train', 'test')]
    return partitions, []


SOURCES = (
    (aptos, dict(dataset_id='aptos_2019', name='APTOS 2019', purpose='Five-grade severity model development',
                 source_url='https://www.kaggle.com/competitions/aptos2019-blindness-detection/data',
                 access='Official Kaggle competition download, confirmed by the user. Access follows Kaggle terms.',
                 group_label='Fingerprint groups', status='audited', notes=[
                     'The internal test is reserved within the annotated training release. The Kaggle competition test is separate.',
                     'Fingerprint grouping reduces detectable overlap. The source provides no patient identity or quality labels.',
                 ])),
    (deepdrid, dict(dataset_id='deepdrid', name='DeepDRiD', purpose='Severity, image quality and paired-view research',
                    source_url='https://github.com/deepdrdoc/DeepDRiD/tree/56d8af71319803150d0e7b90533e0918c7bf85ee',
                    access='Pinned author release under CC BY-SA 4.0. Set-C photo bytes matched the author Git tree.',
                    group_label='Patients', quality_codes={'0': 'Insufficient', '1': 'Sufficient'}, status='audited', notes=[
                        'Set-A roles use the frozen study roster. Set-B confirmation has already been evaluated.',
                        'Set-C is a same-release held-out cohort with two views per eye. The eligibility screen keeps whole patients.',
                        'Disease labels apply to the eye. Quality ratings apply to individual photos.',
                    ])),
    (idrid, dict(dataset_id='idrid', name='IDRiD', purpose='External grade and uncertainty diagnostics',
                 source_url='https://idrid.grand-challenge.org/Data/',
                 access='User-confirmed official download. The package includes CC BY 4.0 terms. Official archive checksum unavailable.',
                 group_label='Fingerprint groups', status='audited', notes=[
                     'Both author splits were external to QSART and the evaluation outcomes have been inspected. Keep them outside fitting and tuning.',
                     'The study populations overlap. Patient identity is unavailable and DME risk is a separate label.',
                 ])),
    (eyeq, dict(dataset_id='eyeq', name='EyeQ', purpose='Optional image-quality research',
                source_url='https://github.com/HzFu/EyeQ/tree/1dcfabb08cd31045de1a4a3ee82e48f6833697aa',
                access='The pinned author annotations are available. Original photographs require official EyePACS access terms.',
                observation_label='Annotations', group_label='Source subject keys',
                quality_codes={'0': 'Good', '1': 'Usable', '2': 'Reject'}, status='metadata_only', notes=[
                    'Mirror matches identify transformed images by original filename. Original bytes, quality labels after transformation and full author test coverage still need verification.',
                    'Filename subject keys group views. Pixel overlap checks and the quality-label harmonization protocol remain pending.',
                ])),
)


def dataset_catalog(root):
    datasets = []
    for project, metadata in SOURCES:
        reports = Reports(Path(root))
        try:
            partitions, studies = project(reports)
            datasets.append(DatasetCatalogEntry(**metadata, partitions=partitions, study_populations=studies,
                evidence=[DatasetEvidence(report=name, sha256=digest) for name, digest in reports.evidence.items()]))
        except (KeyError, OSError, TypeError, ValueError):
            datasets.append(DatasetCatalogEntry(**{**metadata, 'status': 'unavailable',
                'notes': ['The configured reports directory lacks complete, matching audit evidence for this dataset.']}))
    return DatasetCatalog(datasets=datasets)
