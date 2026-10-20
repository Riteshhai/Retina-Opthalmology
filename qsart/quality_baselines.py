"""Fixed, interpretable image-statistic baselines for research comparisons."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import laplace
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from qsart.preprocessing import preprocess_image

FEATURE_NAMES = ('red_mean', 'green_mean', 'blue_mean', 'green_std', 'green_p05',
                 'green_p95', 'green_laplacian_variance', 'green_gradient_energy',
                 'cropped_dark_fraction', 'cropped_saturated_fraction', 'original_dark_fraction')
FEATURE_VERSION = 'fundus_statistics_v1_128px'


def quality_features(image: Image.Image) -> np.ndarray:
    original = np.asarray(image.convert('RGB').resize((128, 128)), dtype=np.float64)/255
    rgb = np.asarray(preprocess_image(image, image_size=128), dtype=np.float64)/255
    green = rgb[:, :, 1]
    gy, gx = np.gradient(green)
    values = [*rgb.mean(axis=(0, 1)), green.std(), *np.quantile(green, [.05, .95]),
              laplace(green).var(), np.mean(gx**2+gy**2),
              np.mean(rgb.max(axis=2) < .05), np.mean(rgb.max(axis=2) > .98),
              np.mean(original.max(axis=2) < .05)]
    result = np.asarray(values, dtype=float)
    if result.shape != (len(FEATURE_NAMES),) or not np.isfinite(result).all():
        raise ValueError('Invalid image statistic features')
    return result


def feature_matrix(frame):
    rows = []
    for path in frame.image_path:
        with Image.open(path) as image:
            rows.append(quality_features(image))
    if not rows:
        raise ValueError('No images in requested partition')
    return np.stack(rows)


def fit_statistical_quality(features, labels, seed=2026):
    x, y = np.asarray(features, dtype=float), np.asarray(labels)
    if x.ndim != 2 or x.shape[1] != len(FEATURE_NAMES) or len(x) != len(y):
        raise ValueError('Feature/label shape mismatch')
    if not np.isfinite(x).all() or set(y.tolist()) != {0, 1}:
        raise ValueError('Finite features and both binary training labels required')
    scaler = StandardScaler().fit(x)
    model = LogisticRegression(C=1., class_weight='balanced', max_iter=1000,
                               random_state=seed, solver='lbfgs').fit(scaler.transform(x), y)
    return {'format_version': 1, 'feature_version': FEATURE_VERSION, 'features': list(FEATURE_NAMES),
            'classes': [0, 1], 'mean': scaler.mean_.tolist(), 'scale': scaler.scale_.tolist(),
            'coefficient': model.coef_[0].tolist(), 'intercept': float(model.intercept_[0]),
            'training_count': len(y), 'seed': seed, 'C': 1., 'class_weight': 'balanced',
            'label_semantics': {'0': 'poor', '1': 'good'},
            'note': 'Research comparator. Image statistics do not certify medical image quality.'}


def predict_statistical_quality(features, artifact):
    from scipy.special import expit
    if artifact.get('feature_version') != FEATURE_VERSION or artifact.get('features') != list(FEATURE_NAMES):
        raise ValueError('Incompatible feature artifact')
    x = np.asarray(features, dtype=float)
    mean, scale = np.asarray(artifact['mean']), np.asarray(artifact['scale'])
    coefficient = np.asarray(artifact['coefficient'])
    if (x.ndim != 2 or x.shape[1] != len(FEATURE_NAMES) or not np.isfinite(x).all()
            or mean.shape != (len(FEATURE_NAMES),) or scale.shape != mean.shape
            or coefficient.shape != mean.shape or not (scale > 0).all()
            or not all(np.isfinite(v).all() for v in (mean, scale, coefficient))
            or not np.isfinite(artifact['intercept'])):
        raise ValueError('Invalid feature matrix or artifact parameters')
    good = expit(((x-mean)/scale) @ coefficient + artifact['intercept'])
    return np.column_stack((1-good, good))


def binary_quality_metrics(labels, probabilities):
    y, p = np.asarray(labels), np.asarray(probabilities, dtype=float)
    if (p.shape != (len(y), 2) or not np.isin(y, [0, 1]).all() or not len(y)
            or not np.isfinite(p).all() or (p < 0).any() or (p > 1).any()
            or not np.allclose(p.sum(1), 1)):
        raise ValueError('Invalid binary labels or probabilities')
    pred = p.argmax(1)
    return {'n': len(y), 'accuracy': float(accuracy_score(y, pred)),
            'macro_f1': float(f1_score(y, pred, labels=[0, 1], average='macro', zero_division=0)),
            'confusion_matrix': confusion_matrix(y, pred, labels=[0, 1]).tolist(),
            'poor_auroc': float(roc_auc_score(y == 0, p[:, 0])) if len(set(y)) == 2 else None,
            'brier': float(((p-np.eye(2)[y.astype(int)])**2).sum(1).mean())}


def artifact_digest(artifact):
    return hashlib.sha256(json.dumps(artifact, sort_keys=True).encode()).hexdigest()


def write_artifact(artifact, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2), encoding='utf-8')
