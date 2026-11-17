"""Generated image/data fixtures check pair proof and label-independent masking."""
import json

import numpy as np
import pandas as pd
from PIL import Image
import pytest

from qsart.data.idrid import IDRID_SOURCE
from qsart.data.idrid_provenance import idrid_source_confirmation
from qsart.data.idrid_review import compare_images, freeze_populations
from qsart.data.manifests import sha256_file


def test_decoded_identity_is_separate_from_bytes_and_canonical_geometry(tmp_path):
    original = tmp_path/'source.png'
    encoded = tmp_path/'different_bytes.png'
    pixels = np.random.default_rng(4).integers(20, 240, (40, 40, 3), dtype=np.uint8)
    Image.fromarray(pixels).save(original, compress_level=0)
    Image.fromarray(pixels).save(encoded, compress_level=9)
    proof = compare_images(original, encoded, sha256_file(original), sha256_file(encoded))
    assert proof['evidence_category'] == 'decoded_rgb_exact_different_bytes'
    assert proof['decoded_rgb_exact'] and proof['canonical_224_rgb_exact'] and not proof['byte_exact']
    assert proof['canonical_224_mae_0_255'] == 0 and proof['canonical_224_rmse_0_255'] == 0
    proof = compare_images(original, original, sha256_file(original), sha256_file(original))
    assert proof['evidence_category'] == 'byte_exact'
    resized = tmp_path/'different_original_size.png'
    Image.fromarray(pixels).resize((80, 80)).save(resized)
    proof = compare_images(original, resized, sha256_file(original), sha256_file(resized))
    assert not proof['decoded_rgb_exact'] and proof['evidence_category'] == 'fingerprint_candidate_unresolved'
    with pytest.raises(ValueError, match='image bytes changed'):
        compare_images(original, encoded, 'a'*64, sha256_file(encoded))


def fixture_population():
    return pd.DataFrame({'image_id': ['a', 'b', 'c', 'd', 'e'],
        'split_group_id': ['component1', 'component1', 'component2', 'component2', 'component3'],
        'dr_grade': [0, 4, 2, 2, 3], 'component_label_conflict': [True, True, False, False, False],
        'split': ['external_holdout']*5, 'eligible_for_training': [False]*5})


def test_masks_propagate_by_component_and_ignore_labels_and_row_order():
    original = fixture_population()
    candidates = pd.DataFrame({'external_image_id': ['a'], 'reference_dataset': ['APTOS_2019']})
    frame, summary = freeze_populations(original, candidates)
    assert frame.in_full_source.all()
    assert frame.in_cross_source_filtered.tolist() == [False, False, True, True, True]
    assert summary['cross_source_flagged_observations'] == 2
    assert summary['populations']['cross_source_filtered_representatives']['images'] == 2
    labels_changed = original.sample(frac=1, random_state=3).assign(dr_grade=0, component_label_conflict=False)
    altered, _ = freeze_populations(labels_changed, candidates)
    columns = ['in_full_source', 'in_cross_source_filtered', 'in_cross_source_filtered_representatives']
    pd.testing.assert_frame_equal(frame.set_index('image_id')[columns].sort_index(),
                                  altered.set_index('image_id')[columns].sort_index())
    pd.testing.assert_frame_equal(original, frame[original.columns])


@pytest.mark.parametrize('mutation', ['training', 'outside_candidate', 'other_reference', 'duplicate_id'])
def test_population_scope_rejects_invalid_inputs(mutation):
    frame = fixture_population()
    candidates = pd.DataFrame({'external_image_id': ['a'], 'reference_dataset': ['APTOS_2019']})
    if mutation == 'training': frame.loc[0, 'eligible_for_training'] = True
    elif mutation == 'outside_candidate': candidates.loc[0, 'external_image_id'] = 'outside'
    elif mutation == 'other_reference': candidates.loc[0, 'reference_dataset'] = 'unexpected'
    else: frame.loc[0, 'image_id'] = 'b'
    with pytest.raises(ValueError):
        freeze_populations(frame, candidates)


def make_confirmation(tmp_path):
    audit = tmp_path/'data_audit.json'
    raw = {'source_provenance_status': 'local_package_download_origin_unconfirmed',
        'private_output_sha256': {'external_manifest.csv': 'b'*64}, 'image_content_sha256': 'c'*64}
    audit.write_text(json.dumps(raw))
    receipt = {'schema_version': 1, 'dataset': 'IDRiD', 'confirmation_kind': 'user_statement',
        'source_provenance_status': 'official_download_user_confirmed', 'confirmation_statement': 'Official release confirmed.',
        'confirmed_at_utc': '2026-11-04T13:00:00+00:00', 'official_archive_checksum_verified': False,
        'source_urls': [IDRID_SOURCE], 'data_audit_sha256': sha256_file(audit),
        'manifest_sha256': 'b'*64, 'image_content_sha256': 'c'*64}
    return audit, receipt


def test_valid_confirmation_overlays_status_without_rewriting_history(tmp_path):
    audit, receipt = make_confirmation(tmp_path)
    before = audit.read_bytes()
    assert idrid_source_confirmation(audit) == {}
    (tmp_path/'source_confirmation.json').write_text(json.dumps(receipt))
    result = idrid_source_confirmation(audit)
    assert result['source_provenance_status'] == 'official_download_user_confirmed'
    assert result['audit_source_provenance_status'] == 'local_package_download_origin_unconfirmed'
    assert result['source_confirmation']['official_archive_checksum_verified'] is False
    assert audit.read_bytes() == before


@pytest.mark.parametrize(('key', 'value'), [
    ('data_audit_sha256', 'a'*64), ('manifest_sha256', 'a'*64), ('image_content_sha256', 'a'*64),
    ('official_archive_checksum_verified', True), ('confirmed_at_utc', '2026-11-04'),
    ('source_urls', []), ('confirmation_statement', ''), ('schema_version', 2),
])
def test_stale_or_unsupported_confirmation_cannot_upgrade_provenance(tmp_path, key, value):
    audit, receipt = make_confirmation(tmp_path)
    receipt[key] = value
    (tmp_path/'source_confirmation.json').write_text(json.dumps(receipt))
    result = idrid_source_confirmation(audit)
    assert result['source_confirmation']['status'] == 'invalid_or_stale'
    assert 'source_provenance_status' not in result


def test_unreadable_confirmation_fails_closed(tmp_path):
    audit, _ = make_confirmation(tmp_path)
    (tmp_path/'source_confirmation.json').write_text('{malformed')
    assert idrid_source_confirmation(audit)['source_confirmation']['status'] == 'invalid_or_stale'
