"""Inventory must omit local identities, paths and credential contents."""
import json

import pandas as pd

from scripts.inspect_datasets import default_aptos_root, inspect_local_datasets


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


def test_local_dataset_setting_is_resolved_from_project_not_current_directory(tmp_path, monkeypatch):
    monkeypatch.delenv('QSART_APTOS_ROOT', raising=False)
    project = tmp_path / 'project'
    project.mkdir()
    (project / '.env').write_text('UNRELATED_SECRET=private\nQSART_APTOS_ROOT="../relocated source"\n')
    monkeypatch.chdir(tmp_path)
    assert default_aptos_root(project) == tmp_path / 'relocated source'


def test_environment_dataset_setting_takes_precedence(tmp_path, monkeypatch):
    (tmp_path / '.env').write_text('QSART_APTOS_ROOT=local_setting\n')
    monkeypatch.setenv('QSART_APTOS_ROOT', 'environment_setting')
    assert default_aptos_root(tmp_path) == tmp_path / 'environment_setting'


def test_missing_local_setting_preserves_original_default(tmp_path, monkeypatch):
    monkeypatch.delenv('QSART_APTOS_ROOT', raising=False)
    (tmp_path / '.env').write_text('UNRELATED_SECRET=private\n')
    assert default_aptos_root(tmp_path) == tmp_path / 'data/raw/aptos2019'
