"""Software-only bindings and probability-transform checks for the CP adapter."""
import json

import numpy as np
import pytest

from qsart.inference.conformal_sidecar import (ConformalSidecar, content_hash,
                                              temperature_probabilities)
from qsart.models.checkpoint import sha256_file
from qsart.uncertainty.conformal import fit_conformal


def fixture_bundle(tmp_path):
    paths = {name: tmp_path/name for name in ('dr', 'quality', 'calibration')}
    for name, path in paths.items():
        path.write_text('software-binding-'+name)
    temperatures = {'first_pass': 1.2, 'all_flip': .9}
    metadata = {f'{name}_checkpoint_sha256': sha256_file(paths[name]) for name in ('dr', 'quality')}
    metadata.update(base_calibration_sha256=sha256_file(paths['calibration']),
        temperature_sha256=content_hash(temperatures), manifest_sha256='software-manifest',
        split_sha256='software-cp-group', temperature_fit_split_sha256='software-temperature-group',
        data_purpose='real_research')
    p = np.tile([.8, .05, .05, .05, .05], (21, 1))
    bundle = {**metadata, 'format_version': 1, 'patient_groups_disjoint': True,
              'independent_temperature': temperatures, 'conformal_images': 21, 'conformal_patients': 21,
              'temperature_images': 21, 'models': {}}
    for path_name in temperatures:
        bundle['models'][path_name] = {str(alpha): {method: fit_conformal(p, np.zeros(21, dtype=int),
            alpha=alpha, method=method, groups=np.arange(21),
            metadata={**metadata, 'probability_path': path_name})
            for method in ('lac', 'ordinal_greedy')} for alpha in (.1, .05)}
    path = tmp_path/'sidecar.json'; path.write_text(json.dumps(bundle))
    return path, paths, bundle


def test_temperature_transform_matches_logit_temperature_and_rejects_invalid():
    logits = np.array([[2., 1., 0., -1., -2.]])
    raw = np.exp(logits)/np.exp(logits).sum(1, keepdims=True)
    expected = np.exp(logits/1.4)/np.exp(logits/1.4).sum(1, keepdims=True)
    np.testing.assert_allclose(temperature_probabilities(raw, 1.4), expected)
    with pytest.raises(ValueError):
        temperature_probabilities([[1, 1, 0, 0, 0]], 1.)


def test_bound_sidecar_preserves_grading_probabilities_and_adds_separate_sets(tmp_path):
    path, paths, bundle = fixture_bundle(tmp_path)
    sidecar = ConformalSidecar(path, dr_checkpoint=paths['dr'], quality_checkpoint=paths['quality'],
                               calibration=paths['calibration'])
    row = {'grade_probabilities': [.7, .1, .1, .05, .05], 'second_pass_activated': False,
           'uncalibrated': {'grade_probabilities': [.8, .05, .05, .05, .05]}}
    result = sidecar.enrich([row])[0]
    assert result['grade_probabilities'] == [.7, .1, .1, .05, .05]
    assert result['conformal_set'] == result['conformal']['prediction_set']
    assert result['conformal']['probability_source'] == 'independent_temperature_for_conformal'
    paths['dr'].write_text('changed-software-weights')
    with pytest.raises(ValueError, match='incompatible'):
        ConformalSidecar(path, dr_checkpoint=paths['dr'], quality_checkpoint=paths['quality'],
                         calibration=paths['calibration'])


def test_sidecar_rejects_temperature_tampering_and_calibration_group_overlap(tmp_path):
    path, paths, bundle = fixture_bundle(tmp_path)
    bundle['independent_temperature']['first_pass'] = 20.
    path.write_text(json.dumps(bundle))
    with pytest.raises(ValueError, match='temperature'):
        ConformalSidecar(path, dr_checkpoint=paths['dr'], quality_checkpoint=paths['quality'],
                         calibration=paths['calibration'])
    bundle['independent_temperature']['first_pass'] = 1.2
    bundle['temperature_fit_split_sha256'] = bundle['split_sha256']
    path.write_text(json.dumps(bundle))
    with pytest.raises(ValueError, match='independently'):
        ConformalSidecar(path, dr_checkpoint=paths['dr'], quality_checkpoint=paths['quality'],
                         calibration=paths['calibration'])
