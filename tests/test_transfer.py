"""Generated image fixtures verify transfer plumbing, not retinal performance."""
from dataclasses import replace
import numpy as np
import pandas as pd
from PIL import Image
import pytest
import torch

from qsart.models.checkpoint import ArtifactError, load_checkpoint
from qsart.training.transfer import (TransferConfig, _state_hash, extract_features,
                                     initialize_transfer, source_provenance, train_transfer)
from qsart.training.trainer import set_seed


@pytest.fixture
def manifest(tmp_path):
    rows = []
    for index in range(18):
        path = tmp_path/f'{index}.png'
        Image.fromarray(np.random.default_rng(index).integers(0, 255, (40, 40, 3), dtype=np.uint8)).save(path)
        rows.append({'image_path': str(path), 'split': 'train' if index < 10 else
                     'validation' if index < 16 else 'calibration', 'subject_id': f'test_{index}',
                     'official_split': 'official_train', 'dr_grade': index % 5, 'quality_grade': index % 2})
    path = tmp_path/'manifest.csv'; pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_frozen_cache_roundtrip_and_changed_input_rejection(manifest, tmp_path):
    config = TransferConfig(image_size=32, pretrained=False, num_workers=0, epochs=2, cpu_threads=1)
    set_seed(config.seed, 1)
    model, info = initialize_transfer(config, tmp_path/'cache', data_purpose='software_test')
    before = _state_hash(model.backbone)
    frame = pd.read_csv(manifest); frame = frame[frame.split != 'calibration']
    cached, first = extract_features(model, frame, manifest, config, tmp_path/'cache', info, source_provenance())
    again, second = extract_features(model, frame, manifest, config, tmp_path/'cache', info, source_provenance())
    assert not first['reused'] and second['reused']
    assert torch.equal(cached['features'], again['features'])
    assert before == _state_hash(model.backbone)
    # A manifest hash alone must not hide modified source pixels.
    frame['sha256'] = ['incorrect'] * len(frame)
    with pytest.raises(ValueError, match='changed'):
        extract_features(model, frame, manifest, config, tmp_path/'cache', info, source_provenance())


def test_trained_head_exports_cpu_checkpoint_without_test_fallback(manifest, tmp_path):
    config = TransferConfig(image_size=32, pretrained=False, num_workers=0, epochs=2,
                            batch_size=8, cpu_threads=1, patience=2)
    result = train_transfer(manifest, tmp_path/'model', tmp_path/'cache', config=config,
                             data_purpose='software_test')
    assert result['total_optimizer_steps'] > 0 and result['backbone_unchanged']
    with pytest.raises(ArtifactError, match='Software-test'):
        load_checkpoint(tmp_path/'model/best.pt')
    _, checkpoint = load_checkpoint(tmp_path/'model/best.pt', allow_software_test=True)
    assert checkpoint['metadata']['selection_metric'] == 'macro_f1_then_unweighted_nll'
    assert checkpoint['metadata']['confirmation_used'] is False
    assert checkpoint['metadata']['code_provenance']['source_file_sha256']


def test_confirmation_rejected_before_any_weights_download(manifest, tmp_path):
    frame = pd.read_csv(manifest); frame.loc[0, 'split'] = 'holdout'
    frame.to_csv(manifest, index=False)
    with pytest.raises(ValueError, match='holdout'):
        train_transfer(manifest, tmp_path/'model', tmp_path/'cache')
    with pytest.raises(ValueError, match='finite'):
        replace(TransferConfig(), learning_rate=float('nan')).validate()
