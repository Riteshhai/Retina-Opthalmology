"""Execute the declared selection-only patient-fold guarded-policy comparison."""
from importlib import metadata
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from qsart.evaluation_policy import (BUDGET_PAIRS, DECISION_REVIEW_PRICES, METHODS, MODES,
                                     evaluate_policy, patient_folds, prepare_mode_rows, tune_policy)
from qsart.evaluation_robustness import paired_patient_bootstrap
from qsart.inference.conformal_sidecar import ConformalSidecar
from qsart.models.checkpoint import sha256_file
from qsart.training.data import read_manifest
from qsart.triage_v2 import Action, PolicyConfig, capacity, screening_metrics


MANIFEST_SHA = '45e9f2d34ed54a12056b57a808d03cfaf29ccc970d148e9eb107b79cf5ac3e0d'
CACHE_SHA = '71013749c8b303104568e8115e0e78f13373ef951e9e5c5bfac1fea9469a0b57'
ARTIFACT_SHA = {
    'dr': 'f530056c27f83dbff8bba4ebb481cf7a9d82289947c4eda655a69ed22685a527',
    'quality': 'a600872ce8e2b256205350a5ef6c99b77d71f679d217c20c26810a3ed76eff43',
    'calibration': '674821e445490c209fab99e56b0e9915ac406ca4ec8694e5efdbed63e456a737',
    'conformal': '15a98f6bfe93f81156c5592701995d08796dd889e3a5e43bcbde7ab5aa61c221',
}
OBJECTIVE_FIELDS = ('proxy_total_cost_per_image', 'missed_severe_auto', 'missed_referable_auto',
                    'severe_deferred_to_recapture', 'review_count', 'auto_nonreferable_count',
                    'recapture_count', 'proxy_review_cost', 'proxy_recapture_cost')


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8', newline='\n')


def public_evaluation(value):
    return {key: item for key, item in value.items() if key != 'actions'}


def public_fit(value):
    result = {key: item for key, item in value.items() if key not in ('actions', 'candidates', 'metrics')}
    result['candidates'] = []
    for candidate in value.get('candidates', []):
        row = {key: item for key, item in candidate.items() if key not in ('actions', 'metrics')}
        if 'metrics' in candidate:
            row['fixed_ledger_metrics'] = {key: candidate['metrics'][key] for key in OBJECTIVE_FIELDS}
        result['candidates'].append(row)
    return result


def outcome_vectors(truth, actions):
    actions = np.array([Action(value).value for value in actions])
    auto = actions == Action.AUTO_NONREFERABLE.value
    review = actions == Action.SPECIALIST_REVIEW.value
    recap = actions == Action.RECAPTURE.value
    weights = np.array(PolicyConfig().severity_costs)[truth]
    miss = auto & (truth >= 2)
    return {'proxy_cost': weights*miss + weights*((truth >= 3) & recap)+.25*review+.20*recap,
            'automatic_indicator': auto.astype(float), 'referable_auto_miss_indicator': miss.astype(float)}


def pooled_metrics(truth, rows, actions, review, recapture, fold_sizes):
    config = PolicyConfig(review_budget=review, recapture_budget=recapture)
    result = screening_metrics(truth, [row['grade_probabilities'] for row in rows], actions, config)
    result['capacity_scope'] = 'Sum of three separately allocated excluded-fold batches'
    result['review_capacity'] = sum(capacity(review, n) for n in fold_sizes)
    result['recapture_capacity'] = sum(capacity(recapture, n) for n in fold_sizes)
    result['review_budget_violation'] = result['review_count'] > result['review_capacity']
    result['recapture_budget_violation'] = result['recapture_count'] > result['recapture_capacity']
    if result['review_budget_violation'] or result['recapture_budget_violation']:
        raise ValueError('Pooled action counts exceed summed fold capacities')
    if not np.isclose(result['proxy_total_cost_per_image'], outcome_vectors(truth, actions)['proxy_cost'].mean()):
        raise ValueError('Per-image fixed ledger differs from pooled scorer')
    return result


def make_plot(report, output):
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), constrained_layout=True)
    for axis, mode in zip(axes, MODES, strict=True):
        for variant, marker in (('tuned', 'o'), ('default', 's')):
            records = [r for r in report['scenarios'] if r['mode'] == mode and r['method'] == 'dual_budget'
                       and r['variant'] == variant]
            for i, record in enumerate(records):
                if record['status'] == 'evaluated':
                    axis.scatter(i, record['pooled_metrics']['coverage'], marker=marker,
                                 color='#35546c' if variant == 'tuned' else '#d58b28',
                                 label=variant if i == 0 else None)
                else:
                    axis.text(i, .03 if variant == 'tuned' else .09, variant+': unavailable',
                              rotation=85, ha='center', va='bottom', fontsize=7)
            axis.plot([], [], marker=marker, linestyle='', color='#35546c' if variant == 'tuned' else '#d58b28',
                      label=variant)
        axis.set_xticks(np.arange(len(BUDGET_PAIRS)), [f'{r:.2f}/{c:.2f}' for r, c in BUDGET_PAIRS], rotation=25)
        axis.set_xlabel('Review / recapture maximum budget')
        axis.set_ylim(0, 1)
        axis.set_title({'ordinal': 'Deployed ordinal gate', 'lac': 'LAC diagnostic', 'no_cp': 'No-CP ablation'}[mode])
        axis.grid(axis='y', alpha=.2)
        handles, labels = axis.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        axis.legend(unique.values(), unique.keys(), fontsize=8)
    axes[0].set_ylabel('Automatic non-referable fraction')
    fig.suptitle('Dual-budget held-fold automatic coverage | 168 development images / 42 patients\n'
                 'Unavailable means one or more folds cannot produce actions; not zero-error success')
    fig.savefig(output/'dual_budget_coverage.png', dpi=160)
    plt.close(fig)


def run(args):
    started = time.perf_counter()
    root = Path(__file__).resolve().parents[1]
    for path, parent in ((args.output, root/'reports'), (args.private, root/'artifacts')):
        if not path.resolve().is_relative_to(parent.resolve()) or path.resolve() == parent.resolve():
            raise ValueError('Outputs must remain in project-local report/artifact subdirectories')
        if path.exists():
            raise ValueError('Existing experiment output refused')
    if sha256_file(args.manifest) != MANIFEST_SHA or sha256_file(args.cache) != CACHE_SHA:
        raise ValueError('Frozen manifest or first-pass cache differs')
    frame = read_manifest(args.manifest)
    if set(frame.split) != {'train', 'validation', 'calibration'}:
        raise ValueError('Only the declared development population is permitted')
    selected = frame[frame.split == 'validation'].reset_index(drop=True)
    if len(selected) != 168 or selected.subject_id.nunique() != 42:
        raise ValueError('Selection population differs from 168 images / 42 patients')
    for row in selected.itertuples():
        if sha256_file(row.image_path) != row.sha256:
            raise ValueError('Selection image bytes differ from frozen manifest')
    cached = json.loads(args.cache.read_text(encoding='utf-8'))
    prior = json.loads(args.cache_report.read_text(encoding='utf-8'))
    paths = {'dr': args.selected/'dr.pt', 'quality': args.selected/'quality.pt',
             'calibration': args.selected/'calibration.json', 'conformal': args.conformal}
    actual_sha = {name: sha256_file(path) for name, path in paths.items()}
    if actual_sha != ARTIFACT_SHA:
        raise ValueError('Selected model/calibration/sidecar bytes changed')
    ConformalSidecar(paths['conformal'], dr_checkpoint=paths['dr'], quality_checkpoint=paths['quality'],
                     calibration=paths['calibration'])
    if (prior['cache_sha256'] != CACHE_SHA or cached['manifest_sha256'] != MANIFEST_SHA
            or cached['partition'] != 'validation' or cached['scope'] != 'development_exploration'
            or cached['dr_checkpoint_sha256'] != actual_sha['dr']
            or cached['quality_checkpoint_sha256'] != actual_sha['quality']
            or cached['base_calibration_sha256'] != actual_sha['calibration']
            or cached['conformal_sha256'] != actual_sha['conformal']):
        raise ValueError('Cache population/model/public-report binding differs')
    truth = selected.dr_grade.to_numpy(dtype=int)
    if cached['true_grades'] != truth.tolist():
        raise ValueError('Cache labels/order differ from selection manifest')
    original = cached['profiles']['single']['rows']
    if len(original) != 168 or any(row['second_pass_activated'] for row in original):
        raise ValueError('Only the complete first-pass profile is accepted')
    groups = selected.subject_id.to_numpy()
    folds = patient_folds(groups)
    fold_sizes = [int((folds == fold).sum()) for fold in range(3)]
    if fold_sizes != [56, 56, 56] or any(len(np.unique(groups[folds == fold])) != 14 for fold in range(3)):
        raise ValueError('Declared patient-fold counts differ')
    sources = ('scripts/run_policy_comparison.py', 'qsart/evaluation_policy.py', 'qsart/triage_v2.py',
               'qsart/evaluation_robustness.py', 'qsart/training/data.py')
    report = {'schema_version': 1, 'experiment_id': 'policy_comparison_v1', 'spec_version': '1.0',
        'dataset': 'DeepDRiD', 'partition': 'validation', 'scope': 'development_out_of_fold_policy',
        'images': 168, 'patients': 42, 'model_version': cached['model_version'],
        'manifest_sha256': MANIFEST_SHA, 'cache_sha256': CACHE_SHA, 'artifact_sha256': actual_sha,
        'cache_report_sha256': sha256_file(args.cache_report), 'protocol_sha256': sha256_file(args.protocol),
        'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'source_sha256': {name: sha256_file(name) for name in sources},
        'package_versions': {name: metadata.version(name) for name in ('numpy', 'scipy', 'pandas')},
        'folds': [{'fold': fold, 'tuning_images': 112, 'tuning_patients': 28,
                   'evaluation_images': 56, 'evaluation_patients': 14} for fold in range(3)],
        'private_fold_assignment_sha256': hashlib.sha256(json.dumps(
            {'groups': groups.tolist(), 'folds': folds.tolist()}, separators=(',', ':')).encode()).hexdigest(),
        'decision_review_prices': list(DECISION_REVIEW_PRICES),
        'fixed_evaluation_costs': {'review': .25, 'recapture': .20, 'severity_weights': [0, 0, 3, 6, 10]},
        'scenarios': [], 'paired_comparisons': [], 'limitations': [
            'All 42 patients selected the classifier and have been inspected; not fresh end-to-end validation.',
            'LAC and no-CP modes are diagnostic sensitivity/ablation, not replacement deployments.',
            'All methods share mandatory guards and severity-based recapture costs; not pure literature baselines.',
            'Allocation review price is tuned; outcome scoring always uses the same fixed evaluation ledger.',
            'No pooled outcome is generated unless every excluded fold has real actions.',
            'Equal-cost LP optima may depend on solver version or row ordering; both are recorded.',
            'Paired intervals condition on fitted overlapping-fold policies and fixed actions, with no refitting/reallocation.',
            'Intervals are descriptive, unadjusted for multiple comparisons and do not certify resampled cohort capacities.',
            'Matching budget caps is not matching realized workloads; no clinical safety or prospective benefit is established.']}
    private = {'subject_ids': groups.tolist(), 'true_grades': truth.tolist(), 'fold_ids': folds.tolist(),
               'scenarios': []}
    pooled_actions = {}
    fold_workloads = {}
    for mode in MODES:
        rows = prepare_mode_rows(original, mode)
        for review, recapture in BUDGET_PAIRS:
            for method in METHODS:
                per_variant = {'tuned': [], 'default': []}
                private_variants = {'tuned': [], 'default': []}
                actions_by_variant = {'tuned': [None]*168, 'default': [None]*168}
                for fold in range(3):
                    fit_indices = np.flatnonzero(folds != fold)
                    eval_indices = np.flatnonzero(folds == fold)
                    fitted = tune_policy([rows[i] for i in fit_indices], truth[fit_indices], method, review, recapture)
                    for variant in ('tuned', 'default'):
                        fit_record = fitted if variant == 'tuned' else {'status': 'not_fitted', 'decision_review_price': .25}
                        if variant == 'tuned' and fitted['status'] == 'untunable':
                            evaluated = {'status': 'untunable', 'diagnostics': fitted.get('diagnostics', {})}
                        else:
                            price = fitted['chosen_decision_review_price'] if variant == 'tuned' else .25
                            evaluated = evaluate_policy([rows[i] for i in eval_indices], truth[eval_indices],
                                                        method, review, recapture, price)
                            if evaluated['status'] == 'evaluated':
                                for index, action in zip(eval_indices, evaluated['actions'], strict=True):
                                    actions_by_variant[variant][index] = action
                        per_variant[variant].append({'fold': fold, 'fit': public_fit(fit_record),
                                                     'evaluation': public_evaluation(evaluated)})
                        private_variants[variant].append({'fold': fold, 'fit': fit_record, 'evaluation': evaluated})
                for variant in ('tuned', 'default'):
                    records = per_variant[variant]
                    statuses = [r['evaluation']['status'] for r in records]
                    key = (mode, review, recapture, method, variant)
                    complete = all(value == 'evaluated' for value in statuses)
                    result = {'mode': mode, 'review_budget': review, 'recapture_budget': recapture,
                        'method': method, 'variant': variant, 'status': 'evaluated' if complete else 'incomplete',
                        'evaluated_folds': statuses.count('evaluated'), 'untunable_folds': statuses.count('untunable'),
                        'evaluation_overload_folds': statuses.count('overloaded'),
                        'images_with_real_actions': 56*statuses.count('evaluated'),
                        'patients_with_real_actions': 14*statuses.count('evaluated'),
                        'fold_records': records, 'pooled_metrics': None}
                    if complete:
                        actions = actions_by_variant[variant]
                        result['pooled_metrics'] = pooled_metrics(truth, rows, actions, review, recapture, fold_sizes)
                        pooled_actions[key] = actions
                        fold_workloads[key] = [(r['evaluation']['metrics']['review_count'],
                                                r['evaluation']['metrics']['recapture_count']) for r in records]
                    report['scenarios'].append(result)
                    private['scenarios'].append({'key': list(key), 'folds': private_variants[variant],
                                                  'pooled_actions_or_missing': actions_by_variant[variant]})
            print(json.dumps({'mode': mode, 'review': review, 'recapture': recapture, 'methods': len(METHODS)}), flush=True)
    def compare(left, right, kind):
        record = {'kind': kind, 'left': list(left), 'right': list(right), 'status': 'not_estimable',
                  'matched_realized_workload_within_every_fold': None}
        if left in pooled_actions and right in pooled_actions:
            record['status'] = 'evaluated'
            record['images'] = 168; record['patients'] = 42
            record['matched_realized_workload_within_every_fold'] = fold_workloads[left] == fold_workloads[right]
            record['left_fold_review_recapture_counts'] = fold_workloads[left]
            record['right_fold_review_recapture_counts'] = fold_workloads[right]
            a, b = outcome_vectors(truth, pooled_actions[left]), outcome_vectors(truth, pooled_actions[right])
            record['paired_deltas'] = {}
            for metric in a:
                delta = paired_patient_bootstrap(b[metric], a[metric], groups)
                delta['direction'] = 'left_minus_right'
                record['paired_deltas'][metric] = delta
        return record
    for mode in MODES:
        for review, recapture in BUDGET_PAIRS:
            for method in METHODS:
                if method != 'dual_budget':
                    report['paired_comparisons'].append(compare((mode, review, recapture, 'dual_budget', 'tuned'),
                        (mode, review, recapture, method, 'tuned'), 'tuned_dual_vs_guarded_comparator'))
                report['paired_comparisons'].append(compare((mode, review, recapture, method, 'tuned'),
                    (mode, review, recapture, method, 'default'), 'independently_tuned_vs_default'))
    report['summary'] = {'scenarios': len(report['scenarios']),
        'fully_evaluated_scenarios': sum(r['status'] == 'evaluated' for r in report['scenarios']),
        'incomplete_scenarios': sum(r['status'] != 'evaluated' for r in report['scenarios']),
        'planned_tuning_candidates': len(MODES)*len(BUDGET_PAIRS)*len(METHODS)*3*len(DECISION_REVIEW_PRICES),
        'paired_comparisons': len(report['paired_comparisons']),
        'estimable_pairs': sum(r['status'] == 'evaluated' for r in report['paired_comparisons'])}
    write_json(args.private/'folds_candidates_actions.json', private)
    report['private_output_sha256'] = sha256_file(args.private/'folds_candidates_actions.json')
    report['elapsed_seconds'] = time.perf_counter()-started
    serialized = json.dumps(report, allow_nan=False)
    if any(token in serialized for token in ('subject_id', 'image_path', str(root), 'C:\\Users\\')):
        raise ValueError('Public report contains a private field/path')
    write_json(args.output/'evaluation.json', report)
    make_plot(report, args.output)
    print(json.dumps(report['summary'], indent=2), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, default=Path('data/processed/deepdrid/development_v1.csv'))
    p.add_argument('--cache', type=Path, default=Path('checkpoints/spec_v1/development_scores.json'))
    p.add_argument('--cache-report', type=Path, default=Path('reports/spec_v1/development_evaluation.json'))
    p.add_argument('--selected', type=Path, default=Path('checkpoints/development_v1/selected'))
    p.add_argument('--conformal', type=Path, default=Path('checkpoints/spec_v1/conformal_r1.json'))
    p.add_argument('--protocol', type=Path, default=Path('docs/research/POLICY_COMPARISON_V1_PROTOCOL.md'))
    p.add_argument('--private', type=Path, default=Path('artifacts/policy_comparison_v1'))
    p.add_argument('--output', type=Path, default=Path('reports/policy_comparison_v1'))
    run(p.parse_args())


if __name__ == '__main__':
    main()
