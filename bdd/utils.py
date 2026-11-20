"""Small shared helpers: seeding, device selection, JSON io, paths."""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
CKPT_DIR = ROOT / "checkpoints"
ART_DIR = ROOT / "artifacts"
DATA_DIR = ROOT / "data"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_json(obj, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_json_default)


def load_json(path: Path):
    with open(path) as f:
        return json.load(f)


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, torch.Tensor):
        return o.tolist()
    raise TypeError(f"not serialisable: {type(o)}")
