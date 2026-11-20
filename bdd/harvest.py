"""Activation harvesting on a clean reference set (no triggers, no labels needed).

Returns, per model, a dict  layer_name -> float32 array (n_samples, n_units):
  text : "ffn{i}"   post-GELU FFN activations (neurons), tokens as samples
         "resid{i}" residual stream after layer i
  image: "block{i}" output of residual block i, spatial positions as samples

Plain PyTorch hooks are used rather than TransformerLens: TL cannot load the custom
small BERT / ResNet checkpoints, and hooks give identical access to the activations.
"""
from __future__ import annotations

import numpy as np
import torch

from .data_text import get_tokenizer, load_sst2
from .data_image import load_cifar10

N_REF_TEXT = 400
N_REF_IMAGE = 500
MAX_TOKENS = 8000
MAX_POSITIONS = 20000


def text_reference(n: int = N_REF_TEXT):
    """Defender's clean set: first n SST-2 validation sentences."""
    _, _, x, y = load_sst2(n_train=10, n_val=n)
    return list(x), list(y)


def image_reference(n: int = N_REF_IMAGE):
    _, _, x, y = load_cifar10()
    return x[:n], y[:n]


@torch.no_grad()
def harvest_text(model, sentences, batch_size=64, seed=0, max_tokens=MAX_TOKENS) -> dict[str, np.ndarray]:
    tok = get_tokenizer()
    layers = model.bert.encoder.layer
    store: dict[str, list] = {}
    hooks = []
    cur = {}
    for i, l in enumerate(layers):
        hooks.append(l.intermediate.register_forward_hook(lambda m, a, o, i=i: cur.__setitem__(f"ffn{i}", o)))
        hooks.append(l.register_forward_hook(lambda m, a, o, i=i: cur.__setitem__(f"resid{i}", o[0] if isinstance(o, tuple) else o)))
    try:
        for s in range(0, len(sentences), batch_size):
            enc = tok(sentences[s:s + batch_size], padding=True, truncation=True, max_length=64, return_tensors="pt")
            model(**enc)
            m = enc["attention_mask"].bool()
            for k, v in cur.items():
                store.setdefault(k, []).append(v[m].float().cpu().numpy())
    finally:
        for h in hooks:
            h.remove()
    rng = np.random.default_rng(seed)
    out = {}
    for k, v in store.items():
        arr = np.concatenate(v)
        if len(arr) > max_tokens:
            arr = arr[rng.choice(len(arr), max_tokens, replace=False)]
        out[k] = arr
    return out


@torch.no_grad()
def harvest_image(model, images, batch_size=250, seed=0, max_positions=MAX_POSITIONS) -> dict[str, np.ndarray]:
    store: dict[str, list] = {}
    for s in range(0, len(images), batch_size):
        _, feats = model(images[s:s + batch_size], return_feats=True)
        for i, f in enumerate(feats):
            b, c, h, w = f.shape
            store.setdefault(f"block{i}", []).append(f.permute(0, 2, 3, 1).reshape(-1, c).float().cpu().numpy())
    rng = np.random.default_rng(seed)
    out = {}
    for k, v in store.items():
        arr = np.concatenate(v)
        if len(arr) > max_positions:
            arr = arr[rng.choice(len(arr), max_positions, replace=False)]
        out[k] = arr
    return out


def neuron_layers(domain: str, acts: dict) -> list[str]:
    """Layers whose units are individual neurons/channels (used for statistical detectors)."""
    pref = "ffn" if domain == "text" else "block"
    return sorted((k for k in acts if k.startswith(pref)), key=lambda s: int(s[len(pref):]))


def sae_layers(domain: str, acts: dict) -> list[str]:
    """Layers the SAE is trained on: residual stream (text) / block outputs (image)."""
    pref = "resid" if domain == "text" else "block"
    return sorted((k for k in acts if k.startswith(pref)), key=lambda s: int(s[len(pref):]))
