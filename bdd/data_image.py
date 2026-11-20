"""CIFAR-10 as in-memory tensors."""
from __future__ import annotations

import numpy as np
import torch
import torchvision

from .utils import DATA_DIR


def _to_tensor(ds):
    x = torch.from_numpy(np.asarray(ds.data)).permute(0, 3, 1, 2).float() / 255.0
    return x, torch.tensor(ds.targets)


def load_cifar10():
    """Returns (x_train, y_train, x_test, y_test); images in [0,1], shape (N,3,32,32)."""
    root = DATA_DIR / "cifar"
    tr = torchvision.datasets.CIFAR10(root, train=True, download=True)
    te = torchvision.datasets.CIFAR10(root, train=False, download=True)
    return (*_to_tensor(tr), *_to_tensor(te))


MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
STD = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)


def normalize(x: torch.Tensor) -> torch.Tensor:
    return (x - MEAN.to(x.device)) / STD.to(x.device)
