"""Artifact reporting fixtures validate accounting; they do not simulate research results."""
import json

import numpy as np
import pandas as pd
import pytest

from qsart.triage import Action
from scripts.run_ablations import (CPU_STRATEGIES, artifact_ablations, cpu_strategy_report,
                                  referable_metrics, safe_benchmark_copy, write_reports)


def fixtures():
    validation = pd.DataFrame({'dr_grade': [0, 2, 3, 4], 'quality_grade': [1, 1, 0, 0]})
    probabilities = [[.8, .05, .05, .05, .05], [.6, .1, .1, .1, .1],
                     [.05, .05, .1, .7, .1], [.05, .05, .1, .1, .7]]
    rows, measurements = [], {}
    for strategy in CPU_STRATEGIES:
        activation = ([True] * 4 if strategy == 'all' else
                      [False] * 4 if strategy == 'single' else [True, False, False, False])
        for index, p in enumerate(probabilities):
            rows.append({'strategy': strategy, 'record_index': index, 'dr_grade': validation.dr_grade[index],
                         'grade_probabilities': json.dumps(p), 'predicted_grade': int(np.argmax(p)),
                         'referable_probability': sum(p[2:]), 'poor_probability': .1,
                         'uncertainty': .3, 'second_pass_activated': activation[index]})
        measurements[strategy] = {'images': 4, 'repeats': 3, 'second_pass_fraction': np.mean(activation),
                                  'mean_ms_per_image': 12., 'median_ms_per_image': 10.,
                                  'p95_ms_per_image': 20., 'population_second_pass_budget': np.mean(activation)}
    benchmark = {'hardware': {'cpu': 'Software test machine', 'private_path': 'PRIVATE_TEST_VALUE'},
                 'configuration': {'split': 'validation', 'model_version': 'software-test', 'image_size': 32,
                                   'private_path': 'PRIVATE_TEST_VALUE'}, 'strategies': measurements,
                 'private_records': ['PRIVATE_TEST_VALUE']}
    return validation, pd.DataFrame(rows), benchmark, probabilities


def test_referable_metrics_keep_misses_separate_from_deferrals():
    validation, _, _, p = fixtures()
    metrics = referable_metrics(validation.dr_grade, p)
    assert metrics['sensitivity'] == pytest.approx(2 / 3)
    assert metrics['specificity'] == 1.
    assert metrics['false_negative'] == 1
    assert metrics['true_positive'] == 2


def test_comparable_cpu_populations_configs_and_identical_conditional_selection():
    validation, rows, benchmark, _ = fixtures()
    report = cpu_strategy_report(rows.sample(frac=1, random_state=8), benchmark, validation)
    assert report['images'] == 4
    assert report['conditional_activation_sets_identical']
    assert report['conditional_grade_probabilities_equal_within_tolerance']
    configurations = [entry['configuration'] for entry in report['strategies']['single']['triage']]
    for strategy in CPU_STRATEGIES:
        result = report['strategies'][strategy]
        assert result['grading']['accuracy'] == .75
        assert [entry['configuration'] for entry in result['triage']] == configurations
        for triage in result['triage']:
            assert triage['review_rate'] <= triage['configuration']['review_budget']
            assert triage['recapture_rate'] <= triage['configuration']['recapture_budget']


@pytest.mark.parametrize('problem', ['duplicate_index', 'wrong_label', 'wrong_class', 'wrong_probability', 'bad_activation'])
def test_cpu_artifact_misalignment_and_inconsistency_rejected(problem):
    validation, rows, benchmark, _ = fixtures()
    if problem == 'duplicate_index':
        rows.loc[1, 'record_index'] = rows.loc[0, 'record_index']
    elif problem == 'wrong_label':
        rows.loc[0, 'dr_grade'] = 4
    elif problem == 'wrong_class':
        rows.loc[0, 'predicted_grade'] = 4
    elif problem == 'wrong_probability':
        rows.loc[0, 'referable_probability'] = .9
    elif problem == 'bad_activation':
        rows.loc[0, 'second_pass_activated'] = True
    with pytest.raises(ValueError):
        cpu_strategy_report(rows, benchmark, validation)


def test_cpu_holdout_artifact_rejected_for_validation_comparison():
    validation, rows, benchmark, _ = fixtures()
    benchmark['configuration']['split'] = 'holdout'
    with pytest.raises(ValueError, match='validation'):
        cpu_strategy_report(rows, benchmark, validation)


def test_report_does_not_export_paths_or_individual_records():
    _, _, benchmark, _ = fixtures()
    safe = safe_benchmark_copy(benchmark)
    assert 'PRIVATE_TEST_VALUE' not in json.dumps(safe)
    assert 'private_records' not in safe
    assert safe['hardware']['cpu'] == 'Software test machine'


def test_ablation_scoring_weights_stay_fixed_when_allocation_weights_change(monkeypatch):
    import qsart.evaluation as evaluation

    configs = []
    def allocate_stub(probabilities, quality, uncertainty, config, method):
        configs.append((config, method))
        return [Action.AUTO_SCREEN] * len(probabilities)
    monkeypatch.setattr(evaluation, 'allocate', allocate_stub)
    validation, _, _, p = fixtures()
    records = [{'grade_probabilities': item, 'quality': {'poor_probability': .1},
                'uncertainty': {'entropy': .3}} for item in p]
    report = artifact_ablations(records, validation)
    values = [entry['weighted_auto_error_per_image'] for entry in report['ablations']]
    assert len(set(values)) == 1  # identical actions must have identical scoring costs
    assert values[0] == .75  # one missed grade-2 case, cost3, across4 images
    assert configs[2][0].severity_costs == (1.,) * 5
    assert configs[3][1] == 'combined_budget'


def test_report_writer_rejects_stale_manifest_before_creating_outputs(tmp_path):
    manifest = tmp_path / 'manifest.csv'
    pd.DataFrame({'split': ['holdout'], 'dr_grade': [0]}).to_csv(manifest, index=False)
    evaluation = tmp_path / 'evaluation.json'
    evaluation.write_text(json.dumps({'manifest_sha256': 'stale', 'split': 'holdout'}))
    with pytest.raises(ValueError, match='SHA256'):
        write_reports(manifest, tmp_path / 'missing.json', evaluation,
                      tmp_path / 'missing.csv', tmp_path / 'missing_benchmark.json', tmp_path / 'reports')
    assert not (tmp_path / 'reports').exists()
