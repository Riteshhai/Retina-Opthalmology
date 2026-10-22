"""Torchvision backbones with explicit supervised output heads."""
from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn
from torchvision import models


class RetinalClassifier(nn.Module):
    def __init__(self, backbone: str = "mobilenet_v3_small", heads: Mapping[str, int] | None = None,
                 pretrained: bool = False):
        super().__init__()
        self.head_sizes = dict(heads or {"dr_grade": 5})
        if not self.head_sizes or any(not name.isidentifier() or int(size) < 2
                                      for name, size in self.head_sizes.items()):
            raise ValueError("Each output head requires an identifier and at least two classes")
        if "dr_grade" in self.head_sizes and self.head_sizes["dr_grade"] != 5:
            raise ValueError("The DR head must have exactly five classes")
        self.backbone_name = backbone
        # An explicit training flag is the only path that may download ImageNet weights.
        if backbone == "resnet18":
            self.backbone = models.resnet18(
                weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
            features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif backbone == "mobilenet_v3_small":
            self.backbone = models.mobilenet_v3_small(
                weights=models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None)
            features = self.backbone.classifier[0].in_features
            self.backbone.classifier = nn.Identity()
        elif backbone == "efficientnet_b0":
            self.backbone = models.efficientnet_b0(
                weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
            features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        self.heads = nn.ModuleDict({name: nn.Linear(features, int(size))
                                    for name, size in self.head_sizes.items()})

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("Expected an N x 3 x H x W image tensor")
        features = self.backbone(images)
        return {name: head(features) for name, head in self.heads.items()}


def build_model(backbone: str = "mobilenet_v3_small", heads: Mapping[str, int] | None = None,
                pretrained: bool = False) -> RetinalClassifier:
    return RetinalClassifier(backbone, heads, pretrained)
