"""Load benchmark models from checkpoints."""
from __future__ import annotations

from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import AutoModelForSequenceClassification

from .models_image import SmallResNet
from .utils import CKPT_DIR, load_json


def list_models(domain: str, root: Path | None = None) -> list[Path]:
    root = (root or CKPT_DIR) / domain
    return sorted(p for p in root.glob("*") if (p / "meta.json").exists())


def load_meta(path: Path) -> dict:
    return load_json(Path(path) / "meta.json")


def load_model(path: Path, device="cpu"):
    path = Path(path)
    meta = load_meta(path)
    if meta["domain"] == "text":
        m = AutoModelForSequenceClassification.from_pretrained(path)
    else:
        m = SmallResNet()
        m.load_state_dict(load_file(str(path / "model.safetensors")))
    return m.to(device).eval(), meta
