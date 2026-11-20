"""Small CIFAR ResNet (ResNet-18 layout, narrow width) that trains on CPU."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data_image import normalize


class BasicBlock(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.short = None
        if stride != 1 or cin != cout:
            self.short = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + (x if self.short is None else self.short(x)))


class SmallResNet(nn.Module):
    """ResNet-18 block layout [2,2,2,2] with base width 16 (~0.7M params)."""

    def __init__(self, num_classes=10, width=16, blocks=(2, 2, 2, 2)):
        super().__init__()
        self.conv1 = nn.Conv2d(3, width, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        layers, cin = [], width
        for i, n in enumerate(blocks):
            cout = width * 2 ** i
            for j in range(n):
                layers.append(BasicBlock(cin, cout, 2 if (j == 0 and i > 0) else 1))
                cin = cout
        self.layers = nn.ModuleList(layers)
        self.fc = nn.Linear(cin, num_classes)

    def forward(self, x, return_feats=False):
        """x in [0,1]. Normalisation is inside the model so attacks work in pixel space."""
        h = F.relu(self.bn1(self.conv1(normalize(x))))
        feats = []
        for blk in self.layers:
            h = blk(h)
            feats.append(h)
        pooled = F.adaptive_avg_pool2d(h, 1).flatten(1)
        logits = self.fc(pooled)
        return (logits, feats) if return_feats else logits
