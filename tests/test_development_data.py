"""Generated metadata tests verify cohort boundaries and natural-code mappings."""
import hashlib
import numpy as np
import pandas as pd
import pytest

from qsart.data.development import (add_defect_targets, check_confirmation_independence,
                                    duplicate_safe_splits, select_development_and_confirmation)


def cohort(ids, official_split):
    rng = np.random.default_rng(2026)
    return pd.DataFrame([{'subject_id': f'p{i}', 'image_id': f'{i}_l1',
                          'dr_grade': i % 5, 'official_split': official_split,
                          'annotation_status': 'verified', 'clarity': 10,
                          'field_definition': 8, 'artifact': 0,
                          'sha256': hashlib.sha256(str(i).encode()).hexdigest(),
                          'phash': f'{int(rng.integers(0, 2**63)):016x}'} for i in ids])


def test_whole_ambiguous_subject_and_old_holdout_excluded():
    train = cohort(range(20), 'official_train')
    train.loc[0, 'annotation_status'] = 'quarantined'
    validation = cohort(range(30, 40), 'official_validation')
    pilot = pd.DataFrame({'subject_id': ['p30', 'p31'], 'split': ['holdout', 'holdout']})
    development, confirmation, report = select_development_and_confirmation(train, validation, pilot)
    assert 'p0' not in set(development.subject_id)
    assert not {'p30', 'p31'} & set(confirmation.subject_id)
    assert report['confirmation_subject_count'] == 8
    check_confirmation_independence(confirmation, development, {'p30', 'p31'})


def test_connected_fingerprints_cannot_cross_development_partitions():
    frame = cohort(range(30), 'official_train')
    frame.loc[1, 'phash'] = frame.loc[0, 'phash']
    result, report = duplicate_safe_splits(frame)
    assert result.loc[0, 'split'] == result.loc[1, 'split']
    assert result.loc[0, 'split_group_id'] == result.loc[1, 'split_group_id']
    assert report['merged_subject_components']
    assert set(result.split) == {'train', 'validation', 'calibration'}


def test_natural_codes_preserved_and_unknown_code_rejected():
    frame = cohort([1], 'official_train')
    mapped = add_defect_targets(frame)
    assert mapped.loc[0, 'clarity'] == 10 and mapped.loc[0, 'clarity_class'] == 4
    assert mapped.loc[0, 'artifact'] == 0 and mapped.loc[0, 'artifact_class'] == 0
    frame.loc[0, 'clarity'] = 5
    with pytest.raises(ValueError, match='Unrecognized'):
        add_defect_targets(frame)
