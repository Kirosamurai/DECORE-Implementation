"""Model definitions: CIFAR VGG-16 (with BatchNorm) and the prunable block."""
from __future__ import annotations

import torch
import torch.nn as nn

# CIFAR VGG-16 config: 13 conv layers, 5 max-pools (32x32 -> 1x1).
VGG16_CFG = [64, 64, "M", 128, 128, "M", 256, 256, 256, "M",
             512, 512, 512, "M", 512, 512, 512, "M"]


class PrunableConvBNReLU(nn.Module):
    """conv -> BN -> ReLU, with a channel mask applied to the BLOCK output.

    Masking after BN+ReLU makes the masked forward pass exactly equivalent to
    physically removing the channel (a zeroed output contributes nothing to the
    next layer), so the compressed model matches the masked one bit-for-bit.
    """

    def __init__(self, conv_layer: nn.Conv2d, bn_layer: nn.BatchNorm2d, device=None):
        super().__init__()
        self.conv = conv_layer
        self.bn = bn_layer
        self.relu = nn.ReLU(inplace=True)
        self.out_channels = conv_layer.out_channels
        # Registered as a buffer so it moves with model.to(device) and is not
        # a learnable parameter. Reassigning a tensor to this name keeps it a buffer.
        self.register_buffer("channel_mask", torch.ones(self.out_channels))

    def forward(self, x):
        out = self.relu(self.bn(self.conv(x)))
        return out * self.channel_mask.view(1, -1, 1, 1)


class VGGCifar(nn.Module):
    def __init__(self, cfg=VGG16_CFG, num_classes: int = 10, device=None):
        super().__init__()
        layers = []
        in_c = 3
        for v in cfg:
            if v == "M":
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                conv = nn.Conv2d(in_c, v, kernel_size=3, padding=1, bias=False)
                layers.append(PrunableConvBNReLU(conv, nn.BatchNorm2d(v), device))
                in_c = v
        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(nn.Linear(512, num_classes))

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def build_model(arch: str, num_classes: int, device):
    if arch != "vgg16":
        raise ValueError(f"Unsupported arch '{arch}' (only 'vgg16' for now).")
    return VGGCifar(VGG16_CFG, num_classes=num_classes, device=device).to(device)
