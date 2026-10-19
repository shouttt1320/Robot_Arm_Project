"""ResNet-18 시각적 특징 추출 백본 모듈."""

from collections import OrderedDict
from typing import List, Tuple
import torch
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from .position_encoding import build_position_encoding


class FrozenBatchNorm2d(torch.nn.Module):
    """배치 정규화 통계치를 고정하여 미세 조정 시 안정성을 보장하는 모듈."""

    def __init__(self, n: int):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, train_backbone: bool, num_channels: int):
        super().__init__()
        return_layers = {"layer4": "0"}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels

    def forward(self, tensor: torch.Tensor):
        xs = self.body(tensor)
        return xs


class Backbone(BackboneBase):
    """ResNet 백본 (Frozen BatchNorm 적용)."""

    def __init__(self, name: str = "resnet18", train_backbone: bool = True):
        # torchvision의 사전학습 ResNet 모델 로드
        backbone = getattr(torchvision.models, name)(
            weights=torchvision.models.ResNet18_Weights.DEFAULT,
            norm_layer=FrozenBatchNorm2d,
        )
        num_channels = 512 if name in ("resnet18", "resnet34") else 2048
        super().__init__(backbone, train_backbone, num_channels)


class Joiner(nn.Sequential):
    """백본 특징 맵과 위치 임베딩을 결합하는 모듈."""

    def __init__(self, backbone: nn.Module, position_embedding: nn.Module):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        xs = self[0](tensor)
        out = []
        pos = []
        for name, x in xs.items():
            out.append(x)
            pos.append(self[1](x).to(x.dtype))
        return out, pos


def build_backbone(backbone_name: str = "resnet18", hidden_dim: int = 512, train_backbone: bool = True) -> Joiner:
    """백본 및 2D 위치 임베딩 생성기"""
    position_embedding = build_position_encoding(hidden_dim)
    backbone = Backbone(name=backbone_name, train_backbone=train_backbone)
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model
