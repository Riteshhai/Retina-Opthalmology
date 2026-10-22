"""Inventory must omit local identities, paths and credential contents."""
import json

import pandas as pd

from scripts.inspect_datasets import inspect_local_datasets


def test_inventory_missing_aptos_and_private_data_sanitized(tmp_path, monkeypatch):
    manifest = tmp_path / 'private_source.csv'
    pd.DataFrame([{'image_id': 'PRIVATE_IMAGE_TOKEN', 'subject_id': 'PRIVATE_PATIENT_TOKEN',
                   'image_path': str(tmp_path / 'PRIVATE_IMAGE_TOKEN.png'), 'split': 'train',
                   'dr_grade': 2, 'quality_grade': 0, 'dataset_source': 'DeepDRiD_regular_fundus'}]).to_csv(
                       manifest, index=False)
    monkeypatch.setenv('KAGGLE_API_TOKEN', 'PRIVATE_CREDENTIAL_TOKEN')
    report = inspect_local_datasets(aptos_root=tmp_path / 'missing', deepdrid_development=manifest,
                                    deepdrid_confirmation=tmp_path / 'missing.csv')
    text = json.dumps(report)
    assert all(value not in text for value in ('PRIVATE_IMAGE_TOKEN', 'PRIVATE_PATIENT_TOKEN',
                                               'PRIVATE_CREDENTIAL_TOKEN', str(tmp_path)))
    assert report['credential_presence_only']['kaggle_api_token_environment_present'] is True
    assert report['aptos']['status'] == 'official_labeled_export_missing'
    assert report['deepdrid_development']['partitions']['train']['subjects'] == 1
    assert report['deepdrid_confirmation']['status'] == 'missing'
