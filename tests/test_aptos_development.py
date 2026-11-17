"""Generated metadata/probabilities test research isolation, never retinal performance."""
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml

from scripts.run_aptos_development import (BACKBONES, ROLE_PREFIX, REPRESENTATIVE_PREFIX,
    choose_candidate, conformal_comparisons, fixed_configs, infer_logits, representative_mask,
    set_metrics, validate_development, validate_inputs, write_json)
from qsart.models.checkpoint import sha256_file
from qsart.training.transfer import TransferConfig


def ordered(value, prefix):
    return hashlib.sha256((prefix+str(value)).encode()).hexdigest(), str(value)


@pytest.fixture
def development():
    records = []
    for split, groups in [('train', 2), ('validation', 2), ('calibration', 4)]:
        for group in range(groups):
            for image in range(2):
                identity = f'generated_{split}_{group}_{image}'
                records.append({'image_id': identity, 'image_path': f'/software_fixture/{identity}.png',
                    'split': split, 'split_group_id': f'{split}_{group}', 'dr_grade': group % 5,
                    'sha256': hashlib.sha256(identity.encode()).hexdigest(),
                    'dataset_source': 'APTOS_2019', 'official_split': 'official_train'})
    frame = pd.DataFrame(records)
    frame['is_group_representative'] = False
    for _, group in frame.groupby('split_group_id'):
        representative = min(group.image_id, key=lambda value: ordered(value, REPRESENTATIVE_PREFIX))
        frame.loc[frame.image_id.eq(representative), 'is_group_representative'] = True
    roles = frame.loc[frame.split.eq('calibration')].copy()
    groups = sorted(roles.split_group_id.unique(), key=lambda value: ordered(value, ROLE_PREFIX))
    first = set(groups[:len(groups)//2])
    roles['calibration_role'] = roles.split_group_id.map(lambda group: 'temperature' if group in first else 'conformal')
    return frame, roles


def test_representatives_are_hash_defined_label_independent_and_roles_disjoint(development):
    frame, roles = development
    actual = validate_development(frame, roles)
    assert actual.sum() == 8
    altered = frame.copy()
    altered['dr_grade'] = 4-altered.dr_grade
    assert representative_mask(altered).equals(actual)
    assert set(roles.loc[roles.calibration_role.eq('temperature'), 'split_group_id']).isdisjoint(
        roles.loc[roles.calibration_role.eq('conformal'), 'split_group_id'])
    shuffled = frame.sample(frac=1, random_state=10)
    assert representative_mask(shuffled).sort_index().equals(actual)


@pytest.mark.parametrize('forbidden', ['test', 'holdout', 'confirmation'])
def test_development_rejects_reserved_rows_before_any_training(development, forbidden):
    frame, roles = development
    frame.loc[0, 'split'] = forbidden
    with pytest.raises(ValueError, match='no test or holdout'):
        validate_development(frame, roles)


def test_temperature_and_conformal_cannot_share_a_group_or_use_selection(development):
    frame, roles = development
    altered = roles.copy()
    changed = altered.index[0]
    altered.loc[changed, 'calibration_role'] = ('conformal' if altered.loc[changed, 'calibration_role'] ==
                                              'temperature' else 'temperature')
    with pytest.raises(ValueError, match='group hash partition'):
        validate_development(frame, altered)
    altered = pd.concat([roles, frame.loc[frame.split.eq('validation')].assign(calibration_role='conformal')])
    with pytest.raises(ValueError, match='exactly calibration'):
        validate_development(frame, altered)


def test_nonrepresentative_or_conflicting_labels_never_silently_accepted(development):
    frame, roles = development
    altered = frame.copy()
    altered.loc[0, 'is_group_representative'] = not bool(altered.loc[0, 'is_group_representative'])
    with pytest.raises(ValueError, match='Representative flags'):
        validate_development(altered, roles)
    altered = frame.copy()
    altered.loc[0, 'dr_grade'] = 4
    with pytest.raises(ValueError, match='quarantined'):
        validate_development(altered, roles)


def test_classifier_selection_ignores_conformal_results_and_has_declared_tie_break():
    candidates = [{'backbone': backbone, 'selection_metrics': {'macro_f1': .6, 'nll': .9},
                   'conformal_mean_size': 5 if index == 0 else 1} for index, backbone in enumerate(BACKBONES)]
    assert choose_candidate(candidates)['backbone'] == 'efficientnet_b0'
    changed = deepcopy(candidates)
    changed[1]['selection_metrics']['nll'] = .8
    assert choose_candidate(changed)['backbone'] == 'mobilenet_v3_small'
    changed[0]['selection_metrics']['macro_f1'] = .7
    assert choose_candidate(changed)['backbone'] == 'efficientnet_b0'


def test_fixed_configuration_rejects_unplanned_hyperparameters(tmp_path):
    path = tmp_path/'config.yaml'
    values = asdict(TransferConfig(backbone='efficientnet_b0'))
    path.write_text(yaml.safe_dump(values))
    configurations = fixed_configs(path)
    assert list(configurations) == list(BACKBONES)
    values['epochs'] = 99
    path.write_text(yaml.safe_dump(values))
    with pytest.raises(ValueError, match='fixed declared settings'):
        fixed_configs(path)


def test_all_fourteen_cp_rows_keep_image_units_raw_and_common_fallback():
    rng = np.random.default_rng(20261103)
    calibration = rng.dirichlet(np.ones(5), size=40)
    selection = rng.dirichlet(np.ones(5), size=12)
    truth = np.arange(40, dtype=int) % 5
    selected_truth = np.arange(12, dtype=int) % 5
    reps = np.arange(12) % 2 == 0
    first, private_first = conformal_comparisons(calibration, truth, selection, selected_truth, reps, {})
    second, private_second = conformal_comparisons(calibration, truth, selection, selected_truth, reps, {})
    assert first == second and private_first == private_second
    assert len(first) == 14
    for row in first:
        assert row['artifact']['unit'] == 'image'
        assert 'patient' not in row['raw']['all_selection_images']
        assert row['raw']['all_selection_images']['n_images'] == 12
        assert row['raw']['selection_group_representatives']['n_images'] == 6
        assert row['argmax_fallback']['all_selection_images']['empty_set_count'] == 0
        assert row['argmax_fallback']['all_selection_images']['marginal_coverage'] >= row['raw']['all_selection_images']['marginal_coverage']
    random = [row for row in private_first if row['randomization_seed'] is not None]
    for seed in range(2026, 2031):
        lo, hi = [row for row in random if row['randomization_seed'] == seed]
        assert lo['calibration_uniforms'] == hi['calibration_uniforms']
        assert lo['selection_uniforms'] == hi['selection_uniforms']
        assert lo['calibration_uniforms'][:12] != lo['selection_uniforms']


def test_coverage_reporting_does_not_invent_patient_or_quality_outcomes():
    result = set_metrics([[], [0], [0, 1, 2, 3, 4]], np.array([0, 1, 4]), np.array([True, False, True]))
    assert result['all_selection_images']['empty_set_count'] == 1
    assert result['all_selection_images']['full_five_grade_set_count'] == 1
    assert 'n_patients' not in result['all_selection_images']
    assert 'patient_all_views_coverage' not in result['all_selection_images']
    assert 'quality' not in result


def test_reserved_inference_is_rejected_before_model_or_image_access():
    with pytest.raises(ValueError, match='calibration and development selection'):
        infer_logits(None, pd.DataFrame({'split': ['test']}), {})


def test_existing_research_outputs_are_preserved(tmp_path):
    output = tmp_path/'aggregate.json'
    write_json(output, {'status': 'fixture'})
    with pytest.raises(ValueError, match='Preserve existing'):
        write_json(output, {'status': 'replacement'})


def test_input_binding_compares_columns_by_name_without_reading_reserved_grade(development, tmp_path):
    frame, roles = development
    frame['image_path'] = frame.image_id.map(lambda value: str((tmp_path/f'{value}.png').resolve()))
    roles['image_path'] = roles.image_id.map(frame.set_index('image_id').image_path)
    reserved = frame.iloc[:1].copy()
    reserved['image_id'] = 'generated_reserved'
    reserved['image_path'] = str((tmp_path/'reserved.png').resolve())
    reserved['split'] = 'test'
    reserved['split_group_id'] = 'generated_reserved_component'
    reserved['sha256'] = 'f'*64
    # The runner is allowed to compare reserved metadata, but must not require or
    # inspect its grade values as part of this development experiment.
    reserved['dr_grade'] = 'reserved_label_not_needed_by_development'
    full = pd.concat([frame, reserved], ignore_index=True)
    # Deliberately use CSV field order different from the comparison field order.
    full = full[sorted(full.columns, reverse=True)]
    args = SimpleNamespace(manifest=tmp_path/'development.csv', full_manifest=tmp_path/'full.csv',
                           calibration_roles=tmp_path/'roles.csv', data_audit=tmp_path/'audit.json')
    frame.to_csv(args.manifest, index=False)
    roles.to_csv(args.calibration_roles, index=False)
    full.to_csv(args.full_manifest, index=False)
    args.data_audit.write_text(json.dumps({'manifest_sha256': sha256_file(args.full_manifest),
        'development_manifest_sha256': sha256_file(args.manifest),
        'calibration_roles_sha256': sha256_file(args.calibration_roles), 'archive_sha256': 'a'*64}))
    actual, _, _, bindings = validate_inputs(args)
    assert len(actual) == len(frame)
    assert set(actual.split) == {'train', 'validation', 'calibration'}
    assert bindings['archive_sha256'] == 'a'*64
