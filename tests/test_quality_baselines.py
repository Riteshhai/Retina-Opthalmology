"""Software-only statistical-feature fixtures, not retinal evidence."""
import json
import numpy as np
from PIL import Image
import pytest

from qsart.quality_baselines import (FEATURE_NAMES, binary_quality_metrics, fit_statistical_quality,
                                     predict_statistical_quality, quality_features)


def test_features_are_finite_on_dark_and_saturated_images():
    for colour in ('black', 'white', 'red'):
        x = quality_features(Image.new('RGB', (48, 64), colour))
        assert x.shape == (len(FEATURE_NAMES),) and np.isfinite(x).all()


def test_artifact_roundtrip_and_fitting_does_not_depend_on_selection_labels():
    rng = np.random.default_rng(2026)
    x = rng.normal(size=(20, len(FEATURE_NAMES)))
    y = np.arange(20) % 2
    artifact = fit_statistical_quality(x, y)
    p = predict_statistical_quality(x, artifact)
    assert np.allclose(p, predict_statistical_quality(x, json.loads(json.dumps(artifact))))
    assert np.allclose(np.asarray(artifact['mean']), x.mean(0))
    assert np.allclose(p.sum(1), 1.)
    assert binary_quality_metrics(y, p)['n'] == 20


def test_invalid_artifact_and_training_labels_rejected():
    x = np.ones((10, len(FEATURE_NAMES)))
    with pytest.raises(ValueError):
        fit_statistical_quality(x, np.zeros(10))
    artifact = fit_statistical_quality(x, np.arange(10) % 2)
    artifact['scale'][0] = 0
    with pytest.raises(ValueError):
        predict_statistical_quality(x, artifact)
