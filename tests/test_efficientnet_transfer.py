"""Generated fixtures verify EfficientNet mechanics, not scientific accuracy."""
import numpy as np
import pandas as pd
from PIL import Image
import pytest
import torch
from torch import nn
from torchvision import models

from qsart.models import build_model
from qsart.models.checkpoint import ArtifactError, load_checkpoint
from qsart.training.transfer import TransferConfig, initialize_transfer, pretrained_spec, train_transfer


def test_efficientnet_head_shape_and_official_pretrained_provenance(tmp_path):
    torch.set_num_threads(1)
    model = build_model('efficientnet_b0').eval()
    assert isinstance(model.backbone.classifier, nn.Identity)
    assert model.heads['dr_grade'].in_features == 1280
    with torch.inference_mode():
        assert model(torch.zeros(2, 3, 32, 32))['dr_grade'].shape == (2, 5)
    info = pretrained_spec('efficientnet_b0')
    assert info['enum'] == str(models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    assert info['url'] == models.EfficientNet_B0_Weights.IMAGENET1K_V1.url
    assert info['url'].startswith('https://download.pytorch.org/models/efficientnet_b0_rwightman-')
    assert len(info['published_sha256_prefix']) == 8
    with pytest.raises(ValueError, match='explicitly allow'):
        initialize_transfer(TransferConfig(backbone='efficientnet_b0'), tmp_path / 'no_download')
    bad_weights = tmp_path / 'no_download' / 'torchhub' / 'checkpoints' / info['filename']
    bad_weights.parent.mkdir(parents=True)
    bad_weights.write_bytes(b'generated invalid weight fixture')
    with pytest.raises(ValueError, match='hash does not match'):
        initialize_transfer(TransferConfig(backbone='efficientnet_b0'), tmp_path / 'no_download')


@pytest.mark.parametrize('backbone', ['mobilenet_v3_small', 'resnet18'])
def test_previous_backbone_state_names_and_dimensions_unchanged(backbone):
    retinal = build_model(backbone)
    reference = getattr(models, backbone)(weights=None)
    if backbone == 'resnet18':
        reference.fc = nn.Identity()
        assert retinal.heads['dr_grade'].in_features == 512
    else:
        reference.classifier = nn.Identity()
        assert retinal.heads['dr_grade'].in_features == 576
    assert {key: tuple(value.shape) for key, value in retinal.backbone.state_dict().items()} == {
        key: tuple(value.shape) for key, value in reference.state_dict().items()}


def test_efficientnet_frozen_training_checkpoint_roundtrip(tmp_path):
    rows = []
    for index in range(16):
        path = tmp_path / f'generated_fixture_{index}.png'
        Image.fromarray(np.random.default_rng(index).integers(0, 255, (40, 40, 3), dtype=np.uint8)).save(path)
        rows.append({'image_path': str(path), 'split': 'train' if index < 10 else
                     'validation' if index < 15 else 'calibration',
                     'dr_grade': index % 5, 'official_split': 'official_train'})
    manifest = tmp_path / 'generated_fixture_manifest.csv'
    pd.DataFrame(rows).to_csv(manifest, index=False)
    config = TransferConfig(backbone='efficientnet_b0', pretrained=False, image_size=32,
                            num_workers=0, epochs=1, batch_size=8, cpu_threads=1)
    result = train_transfer(manifest, tmp_path / 'model', tmp_path / 'cache', config=config,
                             data_purpose='software_test')
    assert result['total_optimizer_steps'] > 0 and result['backbone_unchanged']
    with pytest.raises(ArtifactError, match='Software-test'):
        load_checkpoint(tmp_path / 'model' / 'best.pt')
    model, checkpoint = load_checkpoint(tmp_path / 'model' / 'best.pt', allow_software_test=True)
    assert checkpoint['metadata']['backbone'] == 'efficientnet_b0'
    assert checkpoint['metadata']['confirmation_used'] is False
    with torch.inference_mode():
        assert model(torch.zeros(1, 3, 32, 32))['dr_grade'].shape == (1, 5)
