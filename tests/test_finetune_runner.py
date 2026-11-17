"""Explicit metadata/probability fixtures test isolation, never retinal results."""
from dataclasses import asdict
import hashlib

import numpy as np
import pandas as pd
import pytest
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score
import yaml

from qsart.training.finetune import FinetuneConfig
from qsart.training.data import image_to_tensor
from scripts.run_aptos_finetune import (PRIMARY_SEED, TRAINING_SEEDS, _metric_pair,
    cached_logits, fixed_configs, paired_fingerprint_bootstrap, runtime_mapping, scoring_cache, seed_summary)
from scripts.run_aptos_development import representative_mask, ROLE_PREFIX, _order


@pytest.fixture
def development():
    rows = []
    for split, count in [('train', 2), ('validation', 2), ('calibration', 4)]:
        for group in range(count):
            for photo in range(2):
                rows.append({'image_id': f'fixture_{split}_{group}_{photo}',
                    'image_path': f'/generated_fixture/{split}_{group}_{photo}.png', 'split': split,
                    'split_group_id': f'component_{split}_{group}', 'dr_grade': group % 5,
                    'dataset_source': 'APTOS_2019', 'official_split': 'official_train',
                    'sha256': f'{len(rows):064x}'})
    frame = pd.DataFrame(rows)
    frame['is_group_representative'] = representative_mask(frame)
    roles = frame.loc[frame.split.eq('calibration')].copy()
    groups = sorted(roles.split_group_id.unique(), key=lambda group: _order(group, ROLE_PREFIX))
    first = set(groups[:len(groups)//2])
    roles['calibration_role'] = roles.split_group_id.map(lambda group: 'temperature' if group in first else 'conformal')
    return frame, roles


def test_runtime_mapping_keeps_frozen_identities_labels_rows_and_disjoint_roles(development, tmp_path):
    frame, roles = development
    mapped, mapped_roles = runtime_mapping(frame, roles, tmp_path)
    pd.testing.assert_frame_equal(mapped.drop(columns='image_path'), frame.drop(columns='image_path'))
    pd.testing.assert_frame_equal(mapped_roles.drop(columns='image_path'), roles.drop(columns='image_path'))
    assert mapped.index.equals(frame.index)
    assert all(path.startswith(str(tmp_path)) for path in mapped.image_path)
    lookup = mapped.set_index('image_id').image_path
    assert mapped_roles.image_path.tolist() == mapped_roles.image_id.map(lookup).tolist()
    assert frame.image_path.str.startswith('/generated_fixture/').all()


@pytest.mark.parametrize('forbidden', ['test', 'holdout', 'confirmation'])
def test_runtime_mapping_rejects_reserved_rows_before_image_access(development, tmp_path, forbidden):
    frame, roles = development
    frame.loc[0, 'split'] = forbidden
    with pytest.raises(ValueError, match='no test or holdout'):
        runtime_mapping(frame, roles, tmp_path)


def test_unsafe_source_identity_is_rejected(development, tmp_path):
    frame, roles = development
    frame.loc[0, 'image_id'] = '../../outside'
    with pytest.raises(ValueError, match='Safe source image IDs'):
        runtime_mapping(frame, roles, tmp_path)


def test_bootstrap_resamples_entire_paired_components_deterministically():
    truth = np.array([0, 0, 2, 4, 1])
    groups = np.array(['a', 'a', 'b', 'c', 'c'])
    baseline = np.eye(5)[[0, 1, 2, 3, 1]]
    candidate = np.eye(5)[[0, 0, 2, 4, 1]]
    first = paired_fingerprint_bootstrap(truth, baseline, candidate, groups, repeats=100, seed=17)
    assert first == paired_fingerprint_bootstrap(truth, baseline, candidate, groups, repeats=100, seed=17)
    assert first['fingerprint_groups'] == 3 and first['images'] == 5
    assert first['accuracy']['delta'] == pytest.approx(.4)
    # Independently replay each group draw, including both a/c images per draw.
    rng = np.random.default_rng(17)
    members = [np.flatnonzero(groups == group) for group in np.unique(groups)]
    changes = []
    for draw in rng.integers(0, 3, size=(100, 3)):
        indices = np.concatenate([members[index] for index in draw])
        a, b = baseline[indices].argmax(1), candidate[indices].argmax(1)
        changes.append([accuracy_score(truth[indices], b)-accuracy_score(truth[indices], a),
            f1_score(truth[indices], b, labels=range(5), average='macro', zero_division=0)-
            f1_score(truth[indices], a, labels=range(5), average='macro', zero_division=0)])
    bounds = np.quantile(changes, [.025, .975], axis=0)
    for index, name in enumerate(('accuracy', 'macro_f1')):
        assert first[name]['lower'] == pytest.approx(bounds[0, index])
        assert first[name]['upper'] == pytest.approx(bounds[1, index])


def test_fast_metric_matches_fixed_five_class_scorer_even_if_grade_missing():
    truth, predicted = [0, 0, 2], [0, 1, 2]
    accuracy, macro = _metric_pair(truth, predicted)
    assert accuracy == pytest.approx(accuracy_score(truth, predicted))
    assert macro == pytest.approx(f1_score(truth, predicted, labels=range(5), average='macro', zero_division=0))


def test_primary_seed_and_summary_cannot_be_changed_to_best_observed_seed():
    assert PRIMARY_SEED == 2026
    runs = []
    for index, seed in enumerate(TRAINING_SEEDS):
        metrics = {name: .5+.1*index for name in ('accuracy', 'macro_f1', 'quadratic_weighted_kappa', 'nll', 'ece', 'brier')}
        metrics['per_class'] = [{'recall': .2+.1*index} for _ in range(5)]
        runs.append({'training_seed': seed, 'grading': metrics})
    summary = seed_summary(runs)
    assert summary['macro_f1']['mean'] == pytest.approx(.6)
    assert summary['macro_f1']['sample_standard_deviation'] == pytest.approx(.1)
    with pytest.raises(ValueError, match='All declared training seeds'):
        seed_summary(runs[::-1])


def test_declared_configuration_and_cache_inference_guards(tmp_path):
    path = tmp_path/'config.yaml'
    values = asdict(FinetuneConfig())
    path.write_text(yaml.safe_dump(values))
    assert tuple(fixed_configs(path)) == TRAINING_SEEDS
    values['epochs'] = 99
    path.write_text(yaml.safe_dump(values))
    with pytest.raises(ValueError, match='declared fixed settings'):
        fixed_configs(path)
    with pytest.raises(ValueError, match='fitting/selection'):
        cached_logits(None, pd.DataFrame({'split': ['calibration']}), None, None, FinetuneConfig())


def test_posthoc_cache_is_calibration_only_exact_and_bound_to_source_bytes(tmp_path):
    from qsart.training.finetune import normalize_cached
    pixels = np.random.default_rng(7).integers(0, 255, (45, 67, 3), dtype=np.uint8)
    image = tmp_path/'software_fixture.png'
    Image.fromarray(pixels).save(image)
    digest = hashlib.sha256(image.read_bytes()).hexdigest()
    frame = pd.DataFrame({'image_path': [str(image)], 'split': ['calibration'], 'sha256': [digest]})
    manifest = tmp_path/'runtime.csv'
    frame.to_csv(manifest, index=False)
    config = FinetuneConfig(image_size=32)
    values, proof = scoring_cache(frame, manifest, tmp_path/'pixels.pt', config)
    with Image.open(image) as handle:
        expected = image_to_tensor(handle, 32).numpy()
    np.testing.assert_array_equal(normalize_cached(values)[0].numpy(), expected)
    assert proof['scope'] == 'posthoc_calibration_scoring_only'
    changed = frame.assign(split='test')
    with pytest.raises(ValueError, match='calibration observations only'):
        scoring_cache(changed, manifest, tmp_path/'pixels.pt', config)
    Image.new('RGB', (45, 67), 'blue').save(image)
    with pytest.raises(ValueError, match='Input image changed'):
        scoring_cache(frame, manifest, tmp_path/'pixels.pt', config)
