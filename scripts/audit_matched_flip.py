"""Independently reconstruct the matched flip study and replay its first request."""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from math import comb
from pathlib import Path
import re
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image

from qsart.inference.conformal_sidecar import ConformalSidecar
from qsart.inference.engine import InferenceEngine


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = 'matched_flip_v1'
PROTOCOL = 'docs/research/MATCHED_FLIP_V1_PROTOCOL.md'
PATHS = {
    'cache': 'checkpoints/spec_v1/development_scores.json',
    'original_report': 'reports/spec_v1/development_evaluation.json',
    'manifest': 'data/processed/deepdrid/development_v1.csv',
    'ordinal_witness': 'artifacts/ordinal_study_v1/scores_and_sets.json',
    'dr': 'checkpoints/development_v1/selected/dr.pt',
    'quality': 'checkpoints/development_v1/selected/quality.pt',
    'calibration': 'checkpoints/development_v1/selected/calibration.json',
    'conformal': 'checkpoints/spec_v1/conformal_r1.json',
}
EXPECTED_SOURCES = {
    PROTOCOL, 'scripts/run_matched_flip.py', 'scripts/audit_matched_flip.py',
    'scripts/prepare_research_scores.py', 'qsart/inference/engine.py',
    'qsart/inference/conformal_sidecar.py', 'qsart/models/checkpoint.py',
    'qsart/training/data.py', 'qsart/preprocessing/fundus.py', 'qsart/triage_v2.py',
}
AUTO, RECAPTURE, REVIEW = 'AUTO_NONREFERABLE', 'RECAPTURE', 'SPECIALIST_REVIEW'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def verify_inputs(report):
    source = report['source_commit']
    require(re.fullmatch(r'[0-9a-f]{40}', source) and set(report['source_sha256']) == EXPECTED_SOURCES,
            'Unexpected source inventory')
    for name in EXPECTED_SOURCES:
        bound = subprocess.check_output(['git', 'show', f'{source}:{name}'], cwd=ROOT)
        require(bound == (ROOT / name).read_bytes()
                and hashlib.sha256(bound).hexdigest() == report['source_sha256'][name], 'Source binding differs')
    require(report['source_sha256'][PROTOCOL] == report['protocol_sha256'], 'Protocol binding differs')
    paths = {name: ROOT / rel for name, rel in PATHS.items()}
    actual = {name: digest(path) for name, path in paths.items()}
    require(actual == report['input_sha256'], 'Evidence digest differs')
    cache, prior, witness = (read(paths[key]) for key in ('cache', 'original_report', 'ordinal_witness'))
    require(cache['model_version'] == report['model_version'] == prior['model_info']['model_version']
            and prior['cache_sha256'] == actual['cache'] and cache['partition'] == 'validation'
            and cache['manifest_sha256'] == actual['manifest'] and cache['conformal_sha256'] == actual['conformal']
            and cache['base_calibration_sha256'] == actual['calibration']
            and cache['dr_checkpoint_sha256'] == actual['dr'] and cache['quality_checkpoint_sha256'] == actual['quality'],
            'Archived score provenance differs')
    with paths['manifest'].open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    selected = [row for row in rows if row['split'] == 'validation']
    grades = [int(row['dr_grade']) for row in selected]
    subjects = [row['subject_id'] for row in selected]
    groups = defaultdict(list)
    for index, subject in enumerate(subjects):
        groups[subject].append(index)
    require(len(selected) == 168 and len(groups) == 42 and set(map(len, groups.values())) == {4}
            and all(row['annotation_status'] == 'verified' for row in selected)
            and subjects == witness['selection_subject_ids']
            and grades == witness['selection_grades'] == cache['true_grades'],
            'Manifest selection rows or patient grouping differ')
    require(report['images'] == 168 and report['patients'] == 42
            and report['request_size'] == 8 and report['requests'] == 21
            and report['referable_images'] == sum(grade >= 2 for grade in grades)
            and report['referable_patients'] == sum(any(grades[i] >= 2 for i in group) for group in groups.values()),
            'Reported cohort totals differ')
    return paths, selected, cache, prior, list(groups.values())


def predict_grade(row):
    return max(range(5), key=row['grade_probabilities'].__getitem__)


def binary_positive(row):
    return row['referable_probability'] >= .50


def flags(rows, truth):
    return {
        'grade': [predict_grade(row) == grade for row, grade in zip(rows, truth, strict=True)],
        'referability': [binary_positive(row) == (grade >= 2) for row, grade in zip(rows, truth, strict=True)],
    }


def paired_counts(indices, first, current):
    return {
        'correct': sum(current[i] for i in indices),
        'corrected': sum(current[i] and not first[i] for i in indices),
        'harmed': sum(first[i] and not current[i] for i in indices),
    }


def outcome_reference(rows, grades, quality, groups, baseline):
    current = flags(rows, grades)
    indices = range(len(grades))
    grade = paired_counts(indices, baseline['grade'], current['grade'])
    grade['by_grade'] = []
    for label in range(5):
        selected = [i for i in indices if grades[i] == label]
        grade['by_grade'].append({'grade': label, 'support': len(selected),
                                  **paired_counts(selected, baseline['grade'], current['grade'])})
    binary = paired_counts(indices, baseline['referability'], current['referability'])
    positive = {i for i in indices if binary_positive(rows[i])}
    disease = {i for i in indices if grades[i] >= 2}
    binary.update(false_negatives=len(disease - positive), false_positives=len(positive - disease))
    corrected = {i for i in indices if current['grade'][i] and not baseline['grade'][i]}
    harmed = {i for i in indices if baseline['grade'][i] and not current['grade'][i]}
    patients = {
        'with_grade_correction': sum(bool(corrected.intersection(group)) for group in groups),
        'with_new_grade_error': sum(bool(harmed.intersection(group)) for group in groups),
        'referable_false_negatives': sum(bool(disease.intersection(group)) and not positive.intersection(group)
                                         for group in groups),
        'referable_false_positives': sum(bool(positive.intersection(group)) and not disease.intersection(group)
                                         for group in groups),
    }
    conformal = {}
    for name, field in (('lac', 'baseline_lac_set'), ('ordinal', 'prediction_set')):
        sets = [row['conformal'][field] for row in rows]
        covered = {i for i in indices if grades[i] in sets[i]}
        conformal[name] = {'covered_images': len(covered),
                           'all_views_covered_patients': sum(covered.issuperset(group) for group in groups),
                           'mean_size': sum(len(labels) for labels in sets) / len(rows)}
    actions = Counter()
    auto_misses = 0
    quality_hits = 0
    for i, row in enumerate(rows):
        p = row['grade_probabilities']
        poor = 1 - row['quality']['poor_probability'] < .5
        ordinal = row['conformal_set']
        mandatory = (
            predict_grade(row) >= 2 or sum(p[2:]) >= .5 or sum(p[3:]) >= .25
            or not ordinal or any(label >= 2 for label in ordinal)
            or (row['uncertainty']['entropy'] >= .75 and not poor)
        )
        route = REVIEW if mandatory else RECAPTURE if poor else AUTO
        actions[route] += 1
        auto_misses += route == AUTO and i in disease
        q = row['quality']['probabilities']
        quality_hits += int(q[1] > q[0]) == quality[i]
    return {
        'extra_passes': sum(row['second_pass_activated'] for row in rows), 'grade': grade,
        'referability': binary, 'quality_correct': quality_hits, 'patients': patients, 'conformal': conformal,
        'pointwise': {'actions': {name: actions[name] for name in (AUTO, RECAPTURE, REVIEW)},
                      'referable_automatic_misses': auto_misses},
    }


def distribution_reference(changes, slots):
    """Count request gains with the multivariate hypergeometric formula."""
    count = np.array([1], dtype=object)
    for start in range(0, len(changes), 8):
        category = Counter(changes[start:start + 8])
        local = np.zeros(2 * slots + 1, dtype=object)
        for plus in range(slots + 1):
            for minus in range(slots - plus + 1):
                same = slots - plus - minus
                ways = comb(category[1], plus) * comb(category[-1], minus) * comb(category[0], same)
                local[slots + plus - minus] += ways
        count = np.convolve(count, local)
    origin = slots * (len(changes) // 8)
    return {index - origin: ways for index, ways in enumerate(count.tolist()) if ways}


def verify_distribution(saved, deltas, slots):
    histogram = distribution_reference(deltas, slots)
    total = comb(8, slots) ** 21
    require(sum(histogram.values()) == total == saved['assignments']
            and saved['histogram'] == [{'change': delta, 'assignments': n} for delta, n in histogram.items()]
            and saved['mean_change'] == sum(deltas) * slots / 8
            and (saved['minimum_change'], saved['maximum_change']) == (min(histogram), max(histogram)),
            'Exact random distribution or attainable bounds differ')
    for factor, observed in zip((1, 39), saved['central_95_changes'], strict=True):
        before = sum(n for delta, n in histogram.items() if delta < observed)
        through = sum(n for delta, n in histogram.items() if delta <= observed)
        require(40 * before < factor * total <= 40 * through, 'Discrete random percentile differs')
    return histogram


def chosen_indices(rows, slots, ranking):
    def score(index):
        row = rows[index]
        entropy, risk = row['uncertainty']['entropy'], row['referable_probability']
        return {'entropy': entropy, 'referable_probability': risk, 'combined': (entropy + risk) / 2}[ranking]

    return {i for start in range(0, len(rows), 8)
            for i in sorted(range(start, start + 8), key=lambda i: (-score(i), i))[:slots]}


def verify_workloads(report, cache, selected, prior, groups):
    single, full, archived = [cache['profiles'][name]['rows'] for name in ('single', 'all', 'budget25')]
    truth = [int(row['dr_grade']) for row in selected]
    quality = [int(row['quality_grade']) for row in selected]
    baseline, last = flags(single, truth), flags(full, truth)
    gains = {metric: [int(b) - int(a) for a, b in zip(baseline[metric], last[metric], strict=True)]
             for metric in baseline}
    require(report['anchors']['single']['grade']['correct'] / len(truth) == prior['grading']['accuracy'],
            'First-pass report does not match originating accuracy')
    for name, rows in (('single', single), ('all', full)):
        require(report['anchors'][name] == outcome_reference(rows, truth, quality, groups, baseline),
                'Workload anchor counts differ')
    checked = 0
    for slots, entry in zip((2, 4), report['budgets'], strict=True):
        require(entry['budget'] == slots / 8 and entry['slots_per_request'] == slots,
                'Extra-pass budget differs')
        reference = {name: verify_distribution(entry['random'][name], changes, slots)
                     for name, changes in gains.items()}
        methods = {'entropy', 'referable_probability', 'combined'}
        require(set(entry['rankings']) == methods, 'Ranking inventory differs')
        for name in methods:
            active = chosen_indices(single, slots, name)
            require(len(active) == slots * 21, 'Matched activation count differs')
            rows = [full[i] if i in active else row for i, row in enumerate(single)]
            saved = entry['rankings'][name]
            metrics = outcome_reference(rows, truth, quality, groups, baseline)
            require(saved['metrics'] == metrics, 'Paired outcome or routing counts differ')
            if slots == 2 and name == 'combined':
                require(report['budget25_rows_replayed'] == len(rows)
                        and {i for i, row in enumerate(archived) if row['second_pass_activated']} == active,
                        'Historical controller selection differs')
                for row, expected in zip(rows, archived, strict=True):
                    require(all(value == expected[key] for key, value in row.items() if key != 'processing_time_ms'),
                            'Historical controller prediction fields differ')
            for metric, histogram in reference.items():
                net = sum(gains[metric][i] for i in active)
                upper = sum(n for change, n in histogram.items() if change >= net) / sum(histogram.values())
                require(net == metrics[metric]['corrected'] - metrics[metric]['harmed']
                        and upper == saved['random_probability_matching_or_exceeding'][metric],
                        'Matched random comparison differs')
            checked += 1
    return checked


def replay_request(paths, selection, cache):
    engine = InferenceEngine(paths['dr'], paths['quality'], paths['calibration'], cpu_threads=2)
    cp = ConformalSidecar(paths['conformal'], dr_checkpoint=paths['dr'],
                          quality_checkpoint=paths['quality'], calibration=paths['calibration'])
    images = []
    for row in selection[:8]:
        path = Path(row['image_path'])
        require(digest(path) == row['sha256'], 'Replay source image hash differs')
        with Image.open(path) as image:
            images.append(image.convert('RGB').copy())
    deltas = {}
    for profile, policy, budget in (('single', 'none', 0.), ('all', 'all', 1.)):
        outputs = engine.predict_images(images, second_pass_policy=policy, second_pass_budget=budget,
                                        include_uncalibrated=True)
        cp.enrich(outputs)
        errors = []
        for row, prior in zip(outputs, cache['profiles'][profile]['rows'][:8], strict=True):
            require(all(row[key] == prior[key] for key in ('model_version', 'data_purpose', 'predicted_grade',
                                                            'second_pass_activated', 'conformal')),
                    'Replay decision or conformal sets differ')
            require(row['quality']['predicted_class'] == prior['quality']['predicted_class'], 'Replay quality differs')
            scores = [row['grade_probabilities'], row['quality']['probabilities'],
                      [row['referable_probability'], row['uncertainty']['entropy'], row['uncertainty']['confidence']],
                      *row['quality']['defects'].values()]
            archived = [prior['grade_probabilities'], prior['quality']['probabilities'],
                        [prior['referable_probability'], prior['uncertainty']['entropy'], prior['uncertainty']['confidence']],
                        *(prior['quality']['defects'][name] for name in row['quality']['defects'])]
            errors.extend(float(np.abs(np.asarray(left) - right).max()) for left, right in zip(scores, archived, strict=True))
        maximum = max(errors)
        require(maximum <= 1e-6, 'Replay probability difference exceeds the fixed tolerance')
        deltas[profile] = maximum
    return deltas


def run(output):
    output = (ROOT / output).resolve()
    require(output.is_relative_to(ROOT / 'reports') and output != ROOT / 'reports', 'Invalid output directory')
    target = output / 'independent_audit.json'
    require(not target.exists(), 'Preserve the existing audit receipt')
    result = output / 'evaluation.json'
    report = read(result)
    require(report['experiment_id'] == EXPERIMENT and report['partition'] == 'validation', 'Unexpected report identity')
    paths, selected, cache, prior, groups = verify_inputs(report)
    checked = verify_workloads(report, cache, selected, prior, groups)
    replay = replay_request(paths, selected, cache)
    receipt = {
        'audit_status': 'passed', 'experiment_id': EXPERIMENT, 'source_commit': report['source_commit'],
        'evaluation_sha256': digest(result), 'auditor_sha256': digest(Path(__file__)),
        'selection_images_verified': len(selected), 'selection_patients_verified': len(groups),
        'matched_rankings_verified': checked, 'workload_anchors_verified': 2, 'exact_distributions_verified': 4,
        'controller_rows_replayed_from_saved_paths': len(selected), 'source_photos_replayed': 8,
        'path_predictions_replayed': 16, 'max_absolute_score_delta_by_path': replay,
    }
    with target.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(receipt, indent=2, allow_nan=False) + '\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('reports') / EXPERIMENT)
    run(parser.parse_args().output)
