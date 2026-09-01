"""Surrogate and target model loading with TransferAttack preprocessing."""

from __future__ import annotations

import torch
from torch import nn
from torchvision import models, transforms
import timm

CNN_MODELS = [
    "resnet50", "vgg16", "mobilenet_v2", "inception_v3", "densenet121",
    "resnext50_32x4d", "convnext_tiny", "regnet_y_8gf",
]
TRANSFORMER_MODELS = [
    "vit_base_patch16_224", "pit_b_224", "visformer_small",
    "swin_tiny_patch4_window7_224", "cait_s24_224", "maxvit_tiny_tf_224",
    "mixer_b16_224", "poolformer_s24",
]
TARGET_MODELS = CNN_MODELS + TRANSFORMER_MODELS


class Preprocess(nn.Module):
    def __init__(self, size: int, mean, std):
        super().__init__()
        self.resize = transforms.Resize(size)
        self.normalize = transforms.Normalize(mean, std)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.normalize(self.resize(inputs))


def load_model(name: str, device: torch.device) -> nn.Module:
    if name in models.__dict__:
        backbone = models.__dict__[name](weights="DEFAULT")
    elif name in timm.list_models():
        try:
            backbone = timm.create_model(name, pretrained=True)
        except Exception as hub_error:
            try:
                backbone = timm.create_model(
                    name, pretrained=True, pretrained_cfg_overlay={"hf_hub_id": None})
            except Exception as url_error:
                raise RuntimeError(f"Unable to load pretrained weights for {name}") from url_error
    else:
        raise ValueError(f"Unsupported model: {name}")
    if hasattr(backbone, "default_cfg"):
        cfg = backbone.default_cfg
        size, mean, std = 224, cfg["mean"], cfg["std"]
    elif "Inc" in backbone.__class__.__name__:
        size, mean, std = 299, [0.5] * 3, [0.5] * 3
    else:
        size = 224
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    model = nn.Sequential(Preprocess(size, mean, std), backbone.eval()).to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model
