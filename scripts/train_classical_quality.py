"""Fit a fixed quality comparator using training images only; report selection results."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from qsart.models.checkpoint import sha256_file
from qsart.quality_baselines import (artifact_digest, binary_quality_metrics, feature_matrix,
                                     fit_statistical_quality, predict_statistical_quality, write_artifact)
from qsart.training.data import read_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--output', default=Path('checkpoints/development_v1/classical_quality.json'), type=Path)
    parser.add_argument('--report', default=Path('reports/development_v1/classical_quality.json'), type=Path)
    args = parser.parse_args()
    frame = read_manifest(args.manifest)
    if not set(frame.split).issubset({'train', 'validation', 'calibration'}):
        parser.error('Use development-only manifest; confirmation/holdout images are not accepted')
    training, selection = frame[frame.split == 'train'], frame[frame.split == 'validation']
    x_train, x_selection = feature_matrix(training), feature_matrix(selection)
    artifact = fit_statistical_quality(x_train, training.quality_grade.to_numpy())
    artifact['manifest_sha256'] = sha256_file(args.manifest)
    write_artifact(artifact, args.output)
    p = predict_statistical_quality(x_selection, artifact)
    prior = np.bincount(training.quality_grade.astype(int), minlength=2)/len(training)
    report = {'experiment': 'development_v1_classical_quality', 'split': 'validation',
              'manifest_sha256': artifact['manifest_sha256'], 'artifact_sha256': artifact_digest(artifact),
              'feature_description': 'Fixed 128px RGB/green contrast, exposure, gradient and border statistics',
              'fit_configuration': {'C': 1., 'class_weight': 'balanced', 'max_iter': 1000, 'seed': 2026},
              'logistic_regression': binary_quality_metrics(selection.quality_grade.to_numpy(), p),
              'training_prior_baseline': binary_quality_metrics(selection.quality_grade.to_numpy(),
                                                                            np.tile(prior, (len(selection), 1))),
              'limitation': 'Selection-partition comparator, not independent confirmation or clinical certification.'}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
