"""Compare saved flip inference rankings with uniform selection at equal slot counts."""

import argparse
from collections import Counter
import hashlib
from itertools import combinations
import json
from math import comb
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from qsart.models.checkpoint import sha256_file
from qsart.training.data import read_manifest
from qsart.triage_v2 import Action, pointwise_action


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = 'matched_flip_v1'
PROTOCOL = 'docs/research/MATCHED_FLIP_V1_PROTOCOL.md'
INPUTS = {
    'cache': ('checkpoints/spec_v1/development_scores.json',
              '71013749c8b303104568e8115e0e78f13373ef951e9e5c5bfac1fea9469a0b57'),
    'original_report': ('reports/spec_v1/development_evaluation.json',
                        '7ca66739ba4c6daf72438527e3a747bdbb11488930bbf187db5dac9f2417a58b'),
    'manifest': ('data/processed/deepdrid/development_v1.csv',
                 '45e9f2d34ed54a12056b57a808d03cfaf29ccc970d148e9eb107b79cf5ac3e0d'),
    'ordinal_witness': ('artifacts/ordinal_study_v1/scores_and_sets.json',
                        '0ef95338994b3a587dd38b3034f5c059f063a77901b5af3f17f39292d5f40b4c'),
    'dr': ('checkpoints/development_v1/selected/dr.pt',
           'f530056c27f83dbff8bba4ebb481cf7a9d82289947c4eda655a69ed22685a527'),
    'quality': ('checkpoints/development_v1/selected/quality.pt',
                'a600872ce8e2b256205350a5ef6c99b77d71f679d217c20c26810a3ed76eff43'),
    'calibration': ('checkpoints/development_v1/selected/calibration.json',
                    '674821e445490c209fab99e56b0e9915ac406ca4ec8694e5efdbed63e456a737'),
    'conformal': ('checkpoints/spec_v1/conformal_r1.json',
                  '15a98f6bfe93f81156c5592701995d08796dd889e3a5e43bcbde7ab5aa61c221'),
}
SOURCES = (
    PROTOCOL, 'scripts/run_matched_flip.py', 'scripts/audit_matched_flip.py',
    'scripts/prepare_research_scores.py', 'qsart/inference/engine.py',
    'qsart/inference/conformal_sidecar.py', 'qsart/models/checkpoint.py',
    'qsart/training/data.py', 'qsart/preprocessing/fundus.py', 'qsart/triage_v2.py',
)
CHUNK = 8
SLOTS = (2, 4)
PROFILES = {'single': 0., 'all': 1., 'budget25': .25}
SET_FIELDS = {'lac': 'baseline_lac_set', 'ordinal': 'prediction_set'}


def bind_source(commit):
    def git(*args):
        return subprocess.check_output(['git', *args], cwd=ROOT)

    if git('rev-parse', 'HEAD').decode().strip() != commit or git('status', '--porcelain'):
        raise ValueError('Execution requires a clean checkout of the declared source commit')
    bindings = {}
    for name in SOURCES:
        raw = (ROOT / name).read_bytes()
        if raw != git('show', f'{commit}:{name}'):
            raise ValueError(f'Execution file differs from its locked bytes: {name}')
        bindings[name] = hashlib.sha256(raw).hexdigest()
    return bindings


def check_profile(rows, version):
    p = np.array([row['grade_probabilities'] for row in rows])
    q = np.array([row['quality']['probabilities'] for row in rows])
    if p.shape != (168, 5) or q.shape != (168, 2):
        raise ValueError('Saved probability rows differ from the frozen cohort')
    if any(not np.isfinite(value).all() or (value < 0).any() or (value > 1).any()
           or not np.allclose(value.sum(axis=1), 1., rtol=0, atol=1e-6) for value in (p, q)):
        raise ValueError('Saved probabilities are invalid')
    for row, grade, quality, risk in zip(rows, p.argmax(1), q.argmax(1), p[:, 2:].sum(1), strict=True):
        cp = row['conformal']
        entropy = row['uncertainty']['entropy']
        if (
            row['model_version'] != version or row['data_purpose'] != 'real_research'
            or row['predicted_grade'] != grade or row['quality']['predicted_class'] != quality
            or row['quality']['poor_probability'] != row['quality']['probabilities'][0]
            or not 0. <= row['referable_probability'] <= 1. or abs(row['referable_probability'] - risk) > 1e-6
            or not 0. <= entropy <= 1.
            or cp['path'] != ('all_flip' if row['second_pass_activated'] else 'first_pass')
            or cp['target_coverage'] != .90 or cp['status'] != 'available'
            or cp['prediction_set'] != row['conformal_set']
        ):
            raise ValueError('Saved decision or probability path differs')


def load_evidence():
    paths = {name: ROOT / rel for name, (rel, _) in INPUTS.items()}
    hashes = {name: sha256_file(path) for name, path in paths.items()}
    if hashes != {name: expected for name, (_, expected) in INPUTS.items()}:
        raise ValueError('Frozen model, manifest, calibration or score bytes differ')
    cache, old, ordinal = (json.loads(paths[name].read_bytes()) for name in
                           ('cache', 'original_report', 'ordinal_witness'))
    cache_fields = {'dr_checkpoint_sha256': 'dr', 'quality_checkpoint_sha256': 'quality',
                    'base_calibration_sha256': 'calibration', 'conformal_sha256': 'conformal',
                    'manifest_sha256': 'manifest'}
    if (
        any(cache[field] != hashes[name] for field, name in cache_fields.items())
        or old['cache_sha256'] != hashes['cache'] or old['conformal_artifact_sha256'] != hashes['conformal']
        or old['model_info']['model_version'] != cache['model_version']
        or cache['dataset_id'] != 'development_selection_v2' or cache['partition'] != 'validation'
        or cache['scope'] != 'development_exploration' or set(cache['profiles']) != set(PROFILES)
    ):
        raise ValueError('Expected the original development score and model provenance')
    manifest = read_manifest(paths['manifest'])
    frame = manifest.loc[manifest.split.eq('validation')]
    grades, quality, subjects = (frame.dr_grade.to_numpy(dtype=int), frame.quality_grade.to_numpy(dtype=int),
                                 frame.subject_id.to_numpy())
    groups = [np.flatnonzero(subjects == subject) for subject in np.unique(subjects)]
    if (
        set(manifest.split) != {'train', 'calibration', 'validation'}
        or set(subjects) & set(manifest.loc[~manifest.split.eq('validation'), 'subject_id'])
        or len(grades) != 168 or len(groups) != 42 or any(len(group) != 4 for group in groups)
        or not frame.annotation_status.eq('verified').all()
        or not np.isin(grades, range(5)).all() or not np.isin(quality, (0, 1)).all()
        or grades.tolist() != cache['true_grades'] or grades.tolist() != ordinal['selection_grades']
        or subjects.tolist() != ordinal['selection_subject_ids']
    ):
        raise ValueError('Frozen development label and patient order differs')
    profiles = {}
    for name, budget in PROFILES.items():
        data = cache['profiles'][name]
        rows = data['rows']
        check_profile(rows, cache['model_version'])
        activations = sum(row['second_pass_activated'] for row in rows)
        if data['requested_second_pass_budget'] != budget or activations != budget * len(rows):
            raise ValueError('Archived profile workload differs')
        profiles[name] = rows
    return hashes, cache['model_version'], profiles, grades, quality, groups


def select_slots(scores, slots, chunk=CHUNK):
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1 or len(scores) % chunk or not np.isfinite(scores).all() or not 0 <= slots <= chunk:
        raise ValueError('Finite first-pass scores and complete requests required')
    chosen = np.zeros(len(scores), dtype=bool)
    for offset in range(0, len(scores), chunk):
        order = np.argsort(-scores[offset:offset + chunk], kind='stable')
        chosen[offset + order[:slots]] = True
    return chosen


def correctness(rows, grades):
    predicted = np.array([row['predicted_grade'] for row in rows])
    referable = np.array([row['referable_probability'] >= .50 for row in rows])
    return {'grade': predicted == grades, 'referability': referable == (grades >= 2)}


def paired_counts(result, first):
    return {'correct': int(result.sum()), 'corrected': int((result & ~first).sum()),
            'harmed': int((~result & first).sum())}


def outcome_metrics(rows, grades, quality, groups, first):
    correct = correctness(rows, grades)
    grade = paired_counts(correct['grade'], first['grade'])
    grade['by_grade'] = []
    for label in range(5):
        mask = grades == label
        counts = paired_counts(correct['grade'][mask], first['grade'][mask])
        grade['by_grade'].append({'grade': label, 'support': int(mask.sum()), **counts})
    referable = grades >= 2
    risk_positive = np.array([row['referable_probability'] >= .50 for row in rows])
    binary = {**paired_counts(correct['referability'], first['referability']),
              'false_negatives': int((referable & ~risk_positive).sum()),
              'false_positives': int((~referable & risk_positive).sum())}
    patient_truth = np.array([referable[group].any() for group in groups])
    patient_predicted = np.array([risk_positive[group].any() for group in groups])
    corrected, harmed = correct['grade'] & ~first['grade'], ~correct['grade'] & first['grade']
    patients = {
        'with_grade_correction': sum(bool(corrected[group].any()) for group in groups),
        'with_new_grade_error': sum(bool(harmed[group].any()) for group in groups),
        'referable_false_negatives': int((patient_truth & ~patient_predicted).sum()),
        'referable_false_positives': int((~patient_truth & patient_predicted).sum()),
    }
    conformal = {}
    for name, field in SET_FIELDS.items():
        sets = [row['conformal'][field] for row in rows]
        covered = np.array([g in labels for g, labels in zip(grades, sets, strict=True)])
        conformal[name] = {'covered_images': int(covered.sum()),
                           'all_views_covered_patients': sum(bool(covered[group].all()) for group in groups),
                           'mean_size': sum(map(len, sets)) / len(sets)}
    actions = [pointwise_action(row['grade_probabilities'], 1. - row['quality']['poor_probability'],
                                row['uncertainty']['entropy'], conformal_set=row['conformal_set'])[0] for row in rows]
    counts = Counter(actions)
    return {
        'extra_passes': sum(row['second_pass_activated'] for row in rows), 'grade': grade, 'referability': binary,
        'quality_correct': sum(int(row['quality']['predicted_class'] == value)
                               for row, value in zip(rows, quality, strict=True)),
        'patients': patients, 'conformal': conformal,
        'pointwise': {'actions': {action.value: counts[action] for action in Action},
                      'referable_automatic_misses': sum(action == Action.AUTO_NONREFERABLE and bool(label)
                                                        for action, label in zip(actions, referable, strict=True))},
    }


def random_delta_histogram(blocks, slots):
    """Count assignments exactly, treating positions with equal gains as distinct."""
    counts = Counter({0: 1})
    for block in blocks:
        if not 0 <= slots <= len(block):
            raise ValueError('Extra passes exceed a request size')
        within = Counter(sum(chosen) for chosen in combinations(map(int, block), slots))
        combined = Counter()
        for left, left_count in counts.items():
            for right, right_count in within.items():
                combined[left + right] += left_count * right_count
        counts = combined
    return counts


def summarize_random(counts, blocks, slots):
    total = sum(counts.values())
    minimum = sum(sum(sorted(map(int, block))[:slots]) for block in blocks)
    maximum = sum(sum(sorted(map(int, block), reverse=True)[:slots]) for block in blocks)
    if total != comb(CHUNK, slots) ** len(blocks) or (minimum, maximum) != (min(counts), max(counts)):
        raise ValueError('Random assignment accounting disagrees with the deterministic bounds')

    def quantile(numerator):
        cumulative = 0
        for change, frequency in sorted(counts.items()):
            cumulative += frequency
            if 40 * cumulative >= numerator * total:
                return change
        raise ValueError('Invalid random quantile')

    return {'assignments': total, 'mean_change': sum(delta * n for delta, n in counts.items()) / total,
            'central_95_changes': [quantile(1), quantile(39)], 'minimum_change': minimum, 'maximum_change': maximum,
            'histogram': [{'change': delta, 'assignments': n} for delta, n in sorted(counts.items())]}


def compare_workloads(profiles, grades, quality, groups):
    single, flipped = profiles['single'], profiles['all']
    first = correctness(single, grades)
    last = correctness(flipped, grades)
    gains = {metric: (last[metric].astype(int) - first[metric]).reshape(-1, CHUNK) for metric in first}
    entropy = np.array([row['uncertainty']['entropy'] for row in single])
    risk = np.array([row['referable_probability'] for row in single])
    scores = {'entropy': entropy, 'referable_probability': risk, 'combined': (entropy + risk) / 2}
    anchors = {name: outcome_metrics(rows, grades, quality, groups, first)
               for name, rows in (('single', single), ('all', flipped))}
    budgets = []
    for slots in SLOTS:
        histograms = {metric: random_delta_histogram(blocks, slots) for metric, blocks in gains.items()}
        reference = {metric: summarize_random(histograms[metric], blocks, slots) for metric, blocks in gains.items()}
        rankings = {}
        for name, values in scores.items():
            selected = select_slots(values, slots)
            rows = [flip if extra else base for base, flip, extra in zip(single, flipped, selected, strict=True)]
            if name == 'combined' and slots == 2:
                for row, prior in zip(rows, profiles['budget25'], strict=True):
                    if {key: val for key, val in row.items() if key != 'processing_time_ms'} != {
                        key: val for key, val in prior.items() if key != 'processing_time_ms'
                    }:
                        raise ValueError('The archived 25% controller does not replay')
            metrics = outcome_metrics(rows, grades, quality, groups, first)
            if metrics['extra_passes'] != slots * len(gains['grade']):
                raise ValueError('The extra-pass count differs from the matched budget')
            fraction = {}
            for metric, histogram in histograms.items():
                change = metrics[metric]['corrected'] - metrics[metric]['harmed']
                fraction[metric] = sum(n for delta, n in histogram.items() if delta >= change) / sum(histogram.values())
            rankings[name] = {'metrics': metrics, 'random_probability_matching_or_exceeding': fraction}
        budgets.append({'budget': slots / CHUNK, 'slots_per_request': slots,
                        'random': reference, 'rankings': rankings})
    return anchors, budgets


def run(commit, output):
    output = (ROOT / output).resolve()
    if output == ROOT / 'reports' or not output.is_relative_to(ROOT / 'reports') or output.exists():
        raise ValueError('Use a new aggregate report directory under reports')
    source = bind_source(commit)
    hashes, version, profiles, grades, quality, groups = load_evidence()
    anchors, budgets = compare_workloads(profiles, grades, quality, groups)
    report = {
        'schema_version': 1, 'experiment_id': EXPERIMENT, 'dataset': 'DeepDRiD', 'partition': 'validation',
        'scope': 'development_matched_workload_flip_diagnostic', 'source_commit': commit,
        'source_sha256': source, 'protocol_sha256': source[PROTOCOL], 'input_sha256': hashes,
        'model_version': version, 'images': len(grades), 'patients': len(groups), 'request_size': CHUNK,
        'requests': len(grades) // CHUNK, 'referable_images': int((grades >= 2).sum()),
        'referable_patients': sum(bool((grades[group] >= 2).any()) for group in groups),
        'budget25_rows_replayed': len(grades), 'anchors': anchors, 'budgets': budgets,
        'limitations': [
            'Selection patients already informed model choice.',
            'Uniform selection distributions describe assignment variability on these fixed requests.',
            'Minimum and maximum changes use known reference labels to describe retrospective opportunity.',
            'Mixed-path set inclusion is empirical. Extra-forward counts describe compute workload.',
        ],
    }
    output.mkdir(parents=True)
    with (output / 'evaluation.json').open('xb') as stream:
        stream.write((json.dumps(report, indent=2, allow_nan=False) + '\n').encode('utf-8'))
    print(json.dumps({'experiment_id': EXPERIMENT, 'scored_pairs': len(grades), 'matched_rankings': 6,
                      'budget25_rows_replayed': len(grades)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--output', type=Path, default=Path('reports') / EXPERIMENT)
    args = parser.parse_args()
    run(args.source_commit, args.output)
