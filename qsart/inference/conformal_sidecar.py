"""Independent-temperature conformal outputs alongside unchanged grading scores."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

from qsart.models.checkpoint import sha256_file
from qsart.uncertainty.conformal import predict_sets, validate_artifact


def content_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def temperature_probabilities(probabilities, temperature):
    p = np.asarray(probabilities, dtype=float)
    if (p.ndim != 2 or p.shape[1] != 5 or not np.isfinite(p).all()
            or (p < 0).any() or (p > 1).any() or not np.allclose(p.sum(1), 1., atol=1e-5)
            or not math.isfinite(temperature) or temperature <= 0):
        raise ValueError('Valid raw DR probabilities and positive temperature required')
    logits = np.log(np.clip(p, 1e-30, 1)) / temperature
    logits -= logits.max(1, keepdims=True)
    exp = np.exp(logits)
    return exp/exp.sum(1, keepdims=True)


class ConformalSidecar:
    def __init__(self, path, *, dr_checkpoint, quality_checkpoint, calibration):
        self.sha256 = sha256_file(path)
        self.bundle = json.loads(Path(path).read_text(encoding='utf-8'))
        expected = {'dr_checkpoint_sha256': sha256_file(dr_checkpoint),
                    'quality_checkpoint_sha256': sha256_file(quality_checkpoint),
                    'base_calibration_sha256': sha256_file(calibration)}
        if (self.bundle.get('format_version') != 1
                or self.bundle.get('data_purpose') != 'real_research'
                or any(self.bundle.get(key) != value for key, value in expected.items())):
            raise ValueError('Conformal sidecar is incompatible with loaded research artifacts')
        temperature = self.bundle['independent_temperature']
        if content_hash(temperature) != self.bundle['temperature_sha256']:
            raise ValueError('Conformal temperature binding differs')
        if (self.bundle['temperature_fit_split_sha256'] == self.bundle['split_sha256']
                or not self.bundle.get('patient_groups_disjoint')):
            raise ValueError('Temperature/conformal patient groups are not independently reserved')
        for path_name in ('first_pass', 'all_flip'):
            value = temperature[path_name]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError('Invalid conformal temperature')
            for alpha_key in ('0.1', '0.05'):
                for method in ('lac', 'ordinal_greedy'):
                    artifact = self.bundle['models'][path_name][alpha_key][method]
                    validate_artifact(artifact, expected_metadata={**expected,
                        'temperature_sha256': self.bundle['temperature_sha256'],
                        'manifest_sha256': self.bundle['manifest_sha256'],
                        'split_sha256': self.bundle['split_sha256'],
                        'temperature_fit_split_sha256': self.bundle['temperature_fit_split_sha256'],
                        'probability_path': path_name, 'data_purpose': 'real_research'})
                    if (artifact['method'] != method or artifact['alpha'] != float(alpha_key)
                            or artifact['num_classes'] != 5 or artifact['unit'] != 'patient_max'
                            or artifact['n_images'] != self.bundle['conformal_images']
                            or artifact['n_calibration_units'] != self.bundle['conformal_patients']):
                        raise ValueError('Conformal method, class count or calibration unit mismatch')

    def enrich(self, rows, *, alpha=.1):
        key = str(alpha)
        if key not in ('0.1', '0.05'):
            raise ValueError('Conformal alpha must be .1 or .05')
        for row in rows:
            path = 'all_flip' if row['second_pass_activated'] else 'first_pass'
            raw = row['uncalibrated']['grade_probabilities']
            probabilities = temperature_probabilities([raw], self.bundle['independent_temperature'][path])
            artifacts = self.bundle['models'][path][key]
            sets = {method: predict_sets(probabilities, artifact)[0] for method, artifact in artifacts.items()}
            row['conformal_set'] = sets['ordinal_greedy']
            row['conformal'] = {
                'status': 'available', 'method': 'ordinal_greedy',
                'prediction_set': sets['ordinal_greedy'], 'baseline_lac_set': sets['lac'],
                'target_coverage': 1-alpha, 'probability_source': 'independent_temperature_for_conformal',
                'path': path, 'calibration_patients': artifacts['ordinal_greedy']['n_calibration_units'],
                'coverage_scope': 'Empirical research output; no shifted or conditional-routing coverage guarantee'}
        return rows

    def info(self):
        return {'status': 'available', 'artifact_sha256': self.sha256,
                'methods': ['lac', 'ordinal_greedy'], 'alpha_values': [.1, .05],
                'temperature_images': self.bundle['temperature_images'],
                'conformal_images': self.bundle['conformal_images'],
                'conformal_patients': self.bundle['conformal_patients'],
                'independent_temperature': True, 'formal_coverage_claim': False}
