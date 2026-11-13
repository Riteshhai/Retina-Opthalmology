"""Recompute aggregate comparisons from frozen predictions; never loads a model.

The locked evaluation's per-image rows follow the filtered manifest order. CPU
rows carry an explicit record_index within the validation partition. Reject
inconsistent labels, ordering contracts, and manifest provenance before reporting.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd

from qsart.data.manifests import sha256_file
from qsart.evaluation import ablations, grade_metrics
from qsart.triage import (Action, BudgetInfeasible, PolicyConfig, allocate,
                         screening_metrics, validate_scores)

CPU_STRATEGIES = ('single', 'all', 'uncertainty', 'severity', 'budget')
COMMON_POLICIES = (PolicyConfig(.4, .2), PolicyConfig(.6, .4))


def _grades(values, name='dr_grade') -> np.ndarray:
    numbers = pd.to_numeric(pd.Series(values), errors='raise').to_numpy(dtype=float)
    if not len(numbers) or not np.isfinite(numbers).all() or not np.isin(numbers, np.arange(5)).all():
        raise ValueError(f'{name} must be nonempty integer grades 0..4')
    return numbers.astype(int)


def _bools(values) -> np.ndarray:
    labels = pd.Series(values).astype(str).str.casefold()
    if not labels.isin(['true', 'false', '1', '0']).all():
        raise ValueError('Invalid second_pass_activated values')
    return labels.isin(['true', '1']).to_numpy()


def referable_metrics(grades, probabilities, referable_grade=2) -> dict:
    """Binary metrics use the fixed argmax-grade decision, not tuned thresholds."""
    grades = _grades(grades)
    p, _, _ = validate_scores(probabilities, np.ones(len(grades)), np.zeros(len(grades)))
    actual = grades >= referable_grade
    predicted = p.argmax(1) >= referable_grade
    tp, tn = int((actual & predicted).sum()), int((~actual & ~predicted).sum())
    fp, fn = int((~actual & predicted).sum()), int((actual & ~predicted).sum())
    def ratio(numerator, denominator):
        return float(numerator / denominator) if denominator else None
    return {'grade_threshold': referable_grade, 'decision': 'argmax_grade >= grade_threshold',
            'true_positive': tp, 'true_negative': tn, 'false_positive': fp, 'false_negative': fn,
            'sensitivity': ratio(tp, tp + fn), 'specificity': ratio(tn, tn + fp),
            'precision': ratio(tp, tp + fp), 'f1': ratio(2 * tp, 2 * tp + fp + fn)}


def artifact_ablations(predictions: list[dict], partition: pd.DataFrame) -> dict:
    if len(predictions) != len(partition):
        raise ValueError('Prediction count does not match the frozen manifest partition')
    grades = _grades(partition.dr_grade)
    p = np.asarray([row['grade_probabilities'] for row in predictions], dtype=float)
    q = np.asarray([1 - row['quality']['poor_probability'] for row in predictions], dtype=float)
    u = np.asarray([row['uncertainty']['entropy'] for row in predictions], dtype=float)
    validate_scores(p, q, u)
    return {'images': len(grades), 'grading': grade_metrics(grades, p),
            'referable': referable_metrics(grades, p),
            'common_scoring_policy': asdict(PolicyConfig(.4, .2)),
            'ablations': ablations(grades, p, q, u),
            'interpretation': ['Retrospective aggregate recomputation from frozen calibrated predictions.',
                               'All ablations retain the same severity weights for reported errors.',
                               'The combined-budget ablation deliberately removes separate caps while matching total slots.',
                               'Infeasibility is an observed outcome; no capacity is relaxed or diagnosis assumed.',
                               'No additional model fitting, inference, threshold search, or holdout selection occurred.']}


def _policy_report(grades, p, q, u, config: PolicyConfig) -> dict:
    result = {'configuration': asdict(config), 'method': 'dual_budget'}
    try:
        actions = allocate(p, q, u, config, 'dual_budget')
        result.update(status='evaluated', **screening_metrics(grades, p, actions, config))
        result['action_counts'] = {action.value: actions.count(action) for action in Action}
    except BudgetInfeasible:
        result.update(status='infeasible', reason='Required quality deferrals exceed fixed capacities')
    return result


def cpu_strategy_report(predictions: pd.DataFrame, benchmark: dict,
                        validation: pd.DataFrame) -> dict:
    if benchmark.get('configuration', {}).get('split') != 'validation':
        raise ValueError('CPU strategy comparisons must use the recorded validation partition')
    required = {'strategy', 'record_index', 'dr_grade', 'grade_probabilities', 'predicted_grade',
                'referable_probability', 'poor_probability', 'uncertainty', 'second_pass_activated'}
    if not required.issubset(predictions):
        raise ValueError(f'CPU prediction columns missing: {sorted(required - set(predictions))}')
    if set(predictions.strategy) != set(CPU_STRATEGIES):
        raise ValueError('Expected exactly the five recorded CPU strategies')
    n = len(validation)
    reference_grades = _grades(validation.dr_grade)
    reference_quality = pd.to_numeric(validation.quality_grade, errors='raise').to_numpy()
    if not np.isin(reference_quality, [0, 1]).all():
        raise ValueError('Validation quality labels must be binary official codes')
    reports, activation_vectors, probability_arrays = {}, {}, {}
    for strategy in CPU_STRATEGIES:
        rows = predictions.loc[predictions.strategy.eq(strategy)].copy()
        indices = pd.to_numeric(rows.record_index, errors='raise').to_numpy(dtype=float)
        if not np.isfinite(indices).all() or not np.equal(indices, np.floor(indices)).all():
            raise ValueError(f'{strategy}: record_index must be integer')
        rows['record_index'] = indices.astype(int)
        rows = rows.sort_values('record_index')
        if rows.record_index.tolist() != list(range(n)):
            raise ValueError(f'{strategy}: missing/duplicate record_index or mismatched validation population')
        if not np.array_equal(_grades(rows.dr_grade), reference_grades):
            raise ValueError(f'{strategy}: recorded labels differ from validation manifest order')
        p = np.asarray([json.loads(value) if isinstance(value, str) else value
                        for value in rows.grade_probabilities], dtype=float)
        q = 1 - rows.poor_probability.to_numpy(dtype=float)
        u = rows.uncertainty.to_numpy(dtype=float)
        validate_scores(p, q, u)
        if not np.array_equal(_grades(rows.predicted_grade, 'predicted_grade'), p.argmax(1)):
            raise ValueError(f'{strategy}: predicted_grade disagrees with probabilities')
        if not np.allclose(rows.referable_probability.to_numpy(dtype=float), p[:, 2:].sum(1), atol=1e-6):
            raise ValueError(f'{strategy}: referable probability disagrees with five-grade distribution')
        activated = _bools(rows.second_pass_activated)
        measurement = benchmark.get('strategies', {}).get(strategy, {})
        if measurement.get('images') != n:
            raise ValueError(f'{strategy}: benchmark image count differs from recorded predictions')
        if not np.isclose(float(activated.mean()), measurement.get('second_pass_fraction', np.nan)):
            raise ValueError(f'{strategy}: activation fraction differs from timing artifact')
        reports[strategy] = {'images': n, 'grading': grade_metrics(reference_grades, p),
                             'referable': referable_metrics(reference_grades, p),
                             'quality_accuracy': float(((q >= .5).astype(int) == reference_quality).mean()),
                             'predicted_insufficient_quality_count': int((q < .5).sum()),
                             'second_pass_count': int(activated.sum()),
                             'second_pass_fraction': float(activated.mean()),
                             'cpu_measurement': measurement,
                             'triage': [_policy_report(reference_grades, p, q, u, config)
                                        for config in COMMON_POLICIES]}
        activation_vectors[strategy], probability_arrays[strategy] = activated, p
    conditional = ('uncertainty', 'severity', 'budget')
    same_activation = all(np.array_equal(activation_vectors[conditional[0]], activation_vectors[name])
                          for name in conditional[1:])
    same_predictions = all(np.allclose(probability_arrays[conditional[0]], probability_arrays[name], atol=1e-7)
                           for name in conditional[1:])
    return {'split': 'validation', 'images': n, 'strategies': reports,
            'common_triage_configurations': [asdict(config) for config in COMMON_POLICIES],
            'conditional_activation_sets_identical': same_activation,
            'conditional_grade_probabilities_equal_within_tolerance': same_predictions,
            'limitations': [f'Only the {n}-image pilot validation partition is used in this recorded run; it is not a test superiority study.',
                            'Grading/triage metrics use the first recorded prediction repeat; latency summarizes all timed repeats.',
                            'Each strategy is evaluated on the same images and fixed triage configurations.',
                            'Latency varies with order, scheduling and preprocessing; same activation fraction is not evidence of a speed advantage.',
                            'Identical selected records/predictions across conditional strategies cannot support effectiveness superiority.',
                            'Deferred cases are recommendations, not completed review/recapture or correct diagnoses.',
                            'Exploratory severity weights are research proxies, not established clinical costs.']}


def safe_benchmark_copy(benchmark: dict) -> dict:
    """Explicit public aggregate fields; exclude paths/records/unknown additions."""
    hardware_fields = ('cpu', 'os', 'python', 'torch', 'logical_cpus')
    config_fields = ('model_version', 'backbone', 'image_size', 'device', 'cpu_threads',
                     'data_purpose', 'quality_model_version', 'calibration_method',
                     'second_pass_strategy', 'batch_size', 'model_forward_batch_size',
                     'cohort_chunk_size', 'split', 'warmup_images')
    timing_fields = ('mean_ms_per_image', 'median_ms_per_image', 'p95_ms_per_image',
                     'mean_batch_wall_ms', 'amortized_wall_ms_per_image', 'timing_note',
                     'second_pass_fraction', 'max_observed_rss_mb', 'memory_note',
                     'images', 'repeats', 'population_second_pass_budget')
    return {'hardware': {key: benchmark.get('hardware', {})[key] for key in hardware_fields
                          if key in benchmark.get('hardware', {})},
            'configuration': {key: benchmark.get('configuration', {})[key] for key in config_fields
                              if key in benchmark.get('configuration', {})},
            'strategies': {name: {key: values[key] for key in timing_fields if key in values}
                           for name, values in benchmark.get('strategies', {}).items()},
            'limitations': ['CPU pilot measurements; no clinical-performance or strategy-superiority claim.',
                            'Per-image times exclude image-file decode/HTTP transport; RSS is sampled at batch boundaries.']}


def save_cpu_plot(report: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    names = list(CPU_STRATEGIES)
    x = np.arange(len(names))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for offset, (key, label) in enumerate([('mean_ms_per_image', 'Mean'),
                                          ('median_ms_per_image', 'Median'), ('p95_ms_per_image', 'P95')]):
        axes[0].bar(x + (offset - 1) * .25,
                    [report['strategies'][name]['cpu_measurement'][key] for name in names], .25, label=label)
    repeats = report['strategies']['single']['cpu_measurement']['repeats']
    axes[0].set(ylabel='CPU milliseconds per image', xticks=x, xticklabels=names,
                title=f'Recorded pilot timings ({repeats} repeats)')
    axes[0].legend()
    axes[1].bar(x - .18, [report['strategies'][name]['grading']['accuracy'] for name in names],
                .36, label='Grade accuracy')
    axes[1].bar(x + .18, [report['strategies'][name]['second_pass_fraction'] for name in names],
                .36, label='Second-pass fraction')
    axes[1].set(ylim=(0, 1.05), xticks=x, xticklabels=names,
                title='Same validation cohort; no superiority claim')
    axes[1].legend()
    for axis in axes:
        axis.tick_params(axis='x', rotation=25)
    fig.suptitle(f'CPU strategy comparison — {report["images"]}-image engineering pilot', fontsize=12)
    config = report.get('inference_configuration', {})
    footer = (f'{config.get("image_size", "?")} px; {config.get("cpu_threads", "?")} CPU threads; '
              f'model forward batch {config.get("model_forward_batch_size", "?")}; '
              'excludes file decode/HTTP; timing variation is not a significance test')
    fig.tight_layout(rect=(0, .045, 1, 1))
    fig.text(.5, .01, footer, ha='center', fontsize=8)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_reports(manifest_path: Path, predictions_path: Path, evaluation_path: Path,
                   cpu_predictions_path: Path, benchmark_path: Path, output: Path) -> dict:
    manifest = pd.read_csv(manifest_path)
    manifest_hash = sha256_file(manifest_path)
    evaluation = json.loads(evaluation_path.read_text(encoding='utf-8'))
    if evaluation.get('manifest_sha256') != manifest_hash:
        raise ValueError('Frozen evaluation report does not match manifest SHA256')
    split = evaluation.get('split')
    if split not in {'holdout', 'validation'}:
        raise ValueError('Unrecognized frozen evaluation split')
    partition = manifest.loc[manifest.split.eq(split)].reset_index(drop=True)
    predictions = json.loads(predictions_path.read_text(encoding='utf-8'))
    expected_model = evaluation.get('model_info', {}).get('model_version')
    if not expected_model or any(row.get('model_version') != expected_model for row in predictions):
        raise ValueError('Frozen prediction model version differs from evaluation provenance')
    ablation_report = artifact_ablations(predictions, partition)
    if evaluation.get('grading', {}).get('n') != len(partition):
        raise ValueError('Frozen evaluation population count differs from predictions')
    ablation_report.update(split=split, manifest_sha256=manifest_hash,
                           predictions_sha256=sha256_file(predictions_path),
                           source_evaluation_sha256=sha256_file(evaluation_path),
                           alignment_contract='Frozen evaluate.py JSON row order equals its filtered manifest row order; count/hash validated.')
    # Preserve already measured ablations unavailable from calibrated rows alone.
    ablation_report['additional_previously_recorded_ablations'] = [
        row for row in evaluation.get('ablations', []) if row.get('ablation') == 'without_calibration']
    benchmark = json.loads(benchmark_path.read_text(encoding='utf-8'))
    if benchmark.get('configuration', {}).get('model_version') != expected_model:
        raise ValueError('CPU benchmark model version differs from frozen evaluation')
    for task in ('dr', 'quality'):
        source_hash = benchmark.get('configuration', {}).get('training_provenance', {}).get(task, {}).get('manifest_sha256')
        if source_hash != manifest_hash:
            raise ValueError(f'CPU {task} provenance does not match manifest SHA256')
    validation = manifest.loc[manifest.split.eq('validation')].reset_index(drop=True)
    cpu_report = cpu_strategy_report(pd.read_csv(cpu_predictions_path), benchmark, validation)
    cpu_report.update(manifest_sha256=manifest_hash,
                      prediction_artifact_sha256=sha256_file(cpu_predictions_path),
                      benchmark_artifact_sha256=sha256_file(benchmark_path))
    # Reuse only whitelisted aggregate benchmark fields in public reports.
    sanitized = safe_benchmark_copy(benchmark)
    sanitized['source_artifact_sha256'] = sha256_file(benchmark_path)
    cpu_report['hardware'] = sanitized['hardware']
    cpu_report['inference_configuration'] = sanitized['configuration']
    for name, result in cpu_report['strategies'].items():
        result['cpu_measurement'] = sanitized['strategies'][name]
    output.mkdir(parents=True, exist_ok=True)
    for name, data in [('artifact_ablations.json', ablation_report),
                        ('cpu_strategy_metrics.json', cpu_report), ('cpu_benchmark.json', sanitized)]:
        (output / name).write_text(json.dumps(data, indent=2, allow_nan=False), encoding='utf-8')
    pd.DataFrame(ablation_report['ablations']).to_csv(output / 'artifact_ablations.csv', index=False)
    save_cpu_plot(cpu_report, output / 'cpu_strategy_comparison.png')
    return {'ablations': ablation_report, 'cpu': cpu_report}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=Path('data/processed/deepdrid/pilot_manifest.csv'))
    parser.add_argument('--predictions', type=Path, default=Path('artifacts/pilot/predictions.json'))
    parser.add_argument('--evaluation', type=Path, default=Path('reports/pilot/evaluation.json'))
    parser.add_argument('--cpu-predictions', type=Path, default=Path('artifacts/pilot_cpu_benchmark.predictions.csv'))
    parser.add_argument('--benchmark', type=Path, default=Path('artifacts/pilot_cpu_benchmark.json'))
    parser.add_argument('--output', type=Path, default=Path('reports/pilot'))
    args = parser.parse_args()
    result = write_reports(args.manifest, args.predictions, args.evaluation,
                           args.cpu_predictions, args.benchmark, args.output)
    print(json.dumps({'output': str(args.output), 'retrospective_split': result['ablations']['split'],
                      'retrospective_images': result['ablations']['images'],
                      'cpu_validation_images': result['cpu']['images'],
                      'conditional_activation_sets_identical': result['cpu']['conditional_activation_sets_identical']}, indent=2))


if __name__ == '__main__':
    main()
