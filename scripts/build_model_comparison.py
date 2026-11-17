"""Build aggregate grading comparisons from genuine saved selection predictions.

No inference, fitting, threshold tuning or held-out evaluation occurs here.
Private prediction rows and local filesystem paths are never exported.
"""
from __future__ import annotations

import argparse
import hashlib
from importlib import metadata as package_metadata
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, recall_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qsart.evaluation import grade_metrics


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def verified_selection(folder: Path) -> tuple[dict, pd.DataFrame, np.ndarray, dict]:
    """Check provenance, row identity, labels and probability consistency first."""
    metadata_path = folder / 'training_metadata.json'
    prediction_path = folder / 'selection_predictions.csv'
    checkpoint_path = folder / 'best.pt'
    if not all(path.is_file() for path in (metadata_path, prediction_path, checkpoint_path)):
        raise ValueError('A required real candidate artifact is missing; no comparison was generated')
    metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
    if (metadata.get('data_purpose') != 'real_research' or metadata.get('trained') is not True or
            metadata.get('task') != 'dr' or metadata.get('confirmation_used') is not False):
        raise ValueError('Comparison requires genuine development-only DR training metadata')
    columns = ['manifest_row', 'true_dr_grade', 'predicted_dr_grade'] + [f'dr_grade_p{i}' for i in range(5)]
    frame = pd.read_csv(prediction_path, usecols=columns).sort_values('manifest_row').reset_index(drop=True)
    if frame.empty or frame.isna().any().any() or frame.manifest_row.duplicated().any():
        raise ValueError('Selection predictions need nonempty unique verified row identities')
    for column in ('manifest_row', 'true_dr_grade', 'predicted_dr_grade'):
        values = pd.to_numeric(frame[column], errors='coerce').to_numpy(dtype=float)
        if not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all():
            raise ValueError('Selection row identities and grades must be finite integers')
        frame[column] = values.astype(int)
    if not frame.true_dr_grade.isin(range(5)).all() or not frame.predicted_dr_grade.isin(range(5)).all():
        raise ValueError('Selection grade values must lie in [0, 4]')
    probabilities = frame[[f'dr_grade_p{i}' for i in range(5)]].to_numpy(dtype=float)
    if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError('Selection probabilities must be finite and within [0, 1]')
    sums = probabilities.sum(axis=1)
    if not np.allclose(sums, 1, rtol=0, atol=1e-5):
        raise ValueError('Selection probability rows must sum to one within serialization tolerance')
    probabilities /= sums[:, None]
    if not np.array_equal(probabilities.argmax(1), frame.predicted_dr_grade.to_numpy()):
        raise ValueError('Saved selection predictions disagree with probability argmax')
    if len(frame) != metadata.get('split_counts', {}).get('validation'):
        raise ValueError('Selection predictions and recorded validation count differ')
    recorded = metadata.get('selection_metrics', {})
    recorded = recorded.get('dr_grade', recorded)
    computed = grade_metrics(frame.true_dr_grade.to_numpy(), probabilities, referable_grade=2)
    if not all(np.isclose(computed[name], recorded.get(name, np.nan), rtol=0, atol=1e-6)
               for name in ('accuracy', 'macro_f1')):
        raise ValueError('Saved selection predictions disagree with recorded checkpoint metrics')
    return metadata, frame, probabilities, computed


def build_comparison(project_root: Path) -> dict:
    project_root = project_root.resolve()
    candidates = [
        ('selection_dr_mobilenet', 'checkpoints/development_v1/dr_mobilenet'),
        ('selection_dr_resnet18', 'checkpoints/development_v1/dr_resnet18'),
        ('selection_dr_efficientnet', 'checkpoints/spec_v1/dr_efficientnet_b0'),
    ]
    reference_identity = None
    reference_manifest = None
    reference_fingerprint = None
    experiments = []
    benchmark_path = project_root / 'reports/development_v1/cpu_benchmark.json'
    benchmark = json.loads(benchmark_path.read_text(encoding='utf-8')) if benchmark_path.is_file() else None
    for experiment_id, relative in candidates:
        folder = project_root / relative
        metadata, frame, probabilities, computed = verified_selection(folder)
        identity = frame[['manifest_row', 'true_dr_grade']]
        fingerprint = metadata.get('split_fingerprints', {}).get('validation')
        manifest_sha256 = metadata.get('manifest_sha256')
        if not fingerprint or not manifest_sha256:
            raise ValueError('Candidate selection manifest/fingerprint provenance is missing')
        if reference_identity is not None:
            if (not identity.equals(reference_identity) or reference_manifest != manifest_sha256 or
                    reference_fingerprint != fingerprint):
                raise ValueError('Candidates do not share the same verified selection population')
        else:
            reference_identity, reference_manifest, reference_fingerprint = identity, manifest_sha256, fingerprint
        truth = frame.true_dr_grade.to_numpy(dtype=int)
        positive = truth >= 2
        predicted_positive = probabilities.argmax(1) >= 2
        sensitivity = float(recall_score(positive, predicted_positive, zero_division=0)) if positive.any() else None
        flat = {name: computed[name] for name in ('n', 'accuracy', 'macro_f1', 'quadratic_weighted_kappa',
                                                 'referable_auroc', 'referable_auprc', 'ece', 'brier')}
        flat.update(referable_sensitivity=sensitivity,
                    nll=float(log_loss(truth, probabilities, labels=list(range(5)))),
                    model_size_mib=(folder / 'best.pt').stat().st_size / 1024**2,
                    selected_epoch=metadata['epoch'],
                    latency_ms_per_image=None)
        notes = [
            'Genuine frozen-ImageNet candidate on the same 168-image/42-patient development selection population; '
            'not independent test evidence.',
            'Metrics use saved uncalibrated first-pass five-class softmax probabilities; no temperature or conformal sidecar.',
            'Referable sensitivity: argmax DR grade >=2 versus verified grade >=2. AUROC/AUPRC use summed grade2-4 risk.',
            'ECE uses ten equal-width confidence bins; multiclass Brier sums squared error across all five classes.',
            'Model size is actual serialized best.pt MiB, including checkpoint metadata and optimizer state; not a weights-only export.',
        ]
        latency = None
        if experiment_id == 'selection_dr_mobilenet' and benchmark is not None:
            configuration = benchmark.get('configuration', {})
            single = benchmark.get('strategies', {}).get('single')
            if configuration.get('model_version') != metadata['model_version'] or single is None:
                raise ValueError('Recorded MobileNet CPU timing does not match this candidate model version')
            flat['latency_ms_per_image'] = float(single['mean_ms_per_image'])
            latency = {
                'scope': 'historical_v1_preprocessing_plus_DR_and_quality_model_pipeline_no_conformal',
                'partition': 'validation', 'images': int(single['images']), 'repeats': int(single['repeats']),
                'sampling_rule': configuration.get('timing_sample_rule'),
                'cpu_threads': int(configuration['cpu_threads']),
                'image_size': int(configuration['image_size']),
                'model_forward_batch_size': int(configuration['model_forward_batch_size']),
                'timing_note': single['timing_note'],
                'source_sha256': file_sha256(benchmark_path),
                'architecture_comparison_valid': False,
            }
            notes.append('Latency is a separate historical 48-image selection-prefix benchmark with the v1 DR+quality '
                         'pipeline, no conformal, decode or HTTP. It does not time this 168-image comparison or provide '
                         'a matched architecture speed comparison. Live-workstation/sequential-order limitations apply.')
        else:
            notes.append('Candidate latency is unmeasured and explicitly null; no speed estimate is inferred from training time.')
        source_paths = ['training_metadata.json', 'selection_predictions.csv', 'best.pt']
        experiments.append({
            'experiment_id': experiment_id, 'dataset': 'DeepDRiD', 'partition': 'validation',
            'scope': 'development_selection', 'model_version': metadata['model_version'],
            'architecture': metadata['backbone'], 'metrics': flat, 'notes': notes,
            'latency_measurement': latency,
            'input_sha256': {name: file_sha256(folder / name) for name in source_paths},
            'training_source_provenance': metadata.get('code_provenance'),
        })
    report = {
        'schema_version': 1, 'scope': 'development_selection',
        'development_manifest_sha256': reference_manifest,
        'selection_split_fingerprint': reference_fingerprint,
        'selection_rule': 'macro_f1_then_unweighted_nll; do not select models using consumed confirmation',
        'probability_handling': 'Finite/range checks and row-sum tolerance1e-5; normalize CSV-rounding residues before metrics.',
        'metric_implementation': 'qsart.evaluation.grade_metrics plus sklearn.metrics.recall_score and log_loss',
        'metric_source_sha256': {
            'scripts/build_model_comparison.py': file_sha256(Path(__file__).resolve()),
            'qsart/evaluation.py': file_sha256(project_root / 'qsart/evaluation.py'),
        },
        'package_versions': {name: package_metadata.version(name) for name in ('numpy', 'pandas', 'scikit-learn')},
        'experiments': experiments,
        'limitations': ['Same development selection set and one seed; no confidence/clinical-generalization claim.',
                        'Only MobileNet has the separately scoped historical pipeline timing; architecture latencies are not comparable.',
                        'APTOS reproduction, repeated seeds, independent external IDRiD evaluation and prospective utility remain open.'],
    }
    serialized = json.dumps(report, allow_nan=False)
    if str(project_root) in serialized or any(key in serialized for key in ('image_path', 'subject_id', 'manifest_row')):
        raise ValueError('Comparison report contains forbidden private fields')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', type=Path, default=Path('reports/spec_v1/model_comparison.json'))
    args = parser.parse_args()
    report = build_comparison(args.project_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps([{'experiment_id': row['experiment_id'], **row['metrics']}
                      for row in report['experiments']], indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
