"""Activation patching (causal tracing) between pairs of clean inputs.

For a pair (A, B) with different predicted classes we run the model on A while
overwriting one component's activation with its value from B. The change in the
probability of B's predicted class measures how much that component carries
class-determining information. No trigger is needed.

Components
  text : each attention head (per layer) and each FFN block
  image: channel groups of each residual block
Outputs: influence[comp, class], where the class is the predicted class of the
source input B. A backdoor that routes decisions to its target class through a
small set of components should give that class a concentrated, outsized influence.
"""
from __future__ import annotations

import numpy as np
import torch

from .data_text import get_tokenizer

N_PAIRS = 200
IMG_GROUPS = 4
TEXT_LEN = 48


def _pick_pairs(preds: np.ndarray, n_pairs: int, seed: int):
    """Indices (a, b) with preds[a] != preds[b], balanced over source class b."""
    rng = np.random.default_rng(seed)
    classes = np.unique(preds)
    a_idx, b_idx = [], []
    per = max(n_pairs // len(classes), 1)
    for c in classes:
        bs = np.where(preds == c)[0]
        as_ = np.where(preds != c)[0]
        if len(bs) == 0 or len(as_) == 0:
            continue
        b_idx += list(rng.choice(bs, per)); a_idx += list(rng.choice(as_, per))
    return np.array(a_idx), np.array(b_idx)


def _influence_from_probs(p_clean, p_patch, src_cls, n_cls):
    """p_*: (n_pairs,C) for A clean / patched. Delta in prob of B's class, grouped by class."""
    delta = p_patch[np.arange(len(src_cls)), src_cls] - p_clean[np.arange(len(src_cls)), src_cls]
    out = np.zeros(n_cls)
    for c in range(n_cls):
        m = src_cls == c
        out[c] = delta[m].mean() if m.any() else 0.0
    return out


@torch.no_grad()
def patch_text(model, sentences, n_pairs=N_PAIRS, seed=0):
    tok = get_tokenizer()
    enc = tok(list(sentences), padding="max_length", truncation=True, max_length=TEXT_LEN, return_tensors="pt")
    ids, mask = enc["input_ids"], enc["attention_mask"]
    logits = model(input_ids=ids, attention_mask=mask).logits
    preds = logits.argmax(-1).numpy()
    n_cls = logits.shape[-1]
    ai, bi = _pick_pairs(preds, n_pairs, seed)
    if len(ai) == 0:
        return np.zeros((0, n_cls)), []
    A_ids, A_m, B_ids, B_m = ids[ai], mask[ai], ids[bi], mask[bi]
    src = preds[bi]
    p_clean = torch.softmax(model(input_ids=A_ids, attention_mask=A_m).logits, -1).numpy()

    layers = model.bert.encoder.layer
    n_heads = model.config.num_attention_heads
    cache, state = {}, {"patch": None}

    def make_hook(key):
        def pre(mod, args):
            x = args[0]
            if state["mode"] == "cache":
                cache[key] = x.detach()
                return None
            if state["patch"] is not None and state["patch"][0] == key:
                sl = state["patch"][1]
                x = x.clone()
                x[..., sl] = cache[key][..., sl]
                return (x,) + tuple(args[1:])
            return None
        return pre

    hooks, comps = [], []
    for li, l in enumerate(layers):
        hooks.append(l.attention.output.dense.register_forward_pre_hook(make_hook(("attn", li))))
        hooks.append(l.output.dense.register_forward_pre_hook(make_hook(("ffn", li))))
        d = model.config.hidden_size // n_heads
        for h in range(n_heads):
            comps.append((f"L{li}H{h}", ("attn", li), slice(h * d, (h + 1) * d)))
        comps.append((f"L{li}FFN", ("ffn", li), slice(None)))
    try:
        state["mode"] = "cache"
        model(input_ids=B_ids, attention_mask=B_m)
        state["mode"] = "patch"
        infl = np.zeros((len(comps), n_cls))
        for ci, (_, key, sl) in enumerate(comps):
            state["patch"] = (key, sl)
            p = torch.softmax(model(input_ids=A_ids, attention_mask=A_m).logits, -1).numpy()
            infl[ci] = _influence_from_probs(p_clean, p, src, n_cls)
    finally:
        for h in hooks:
            h.remove()
    return infl, [c[0] for c in comps]


@torch.no_grad()
def patch_image(model, images, n_pairs=N_PAIRS, seed=0, groups=IMG_GROUPS):
    logits = model(images)
    preds = logits.argmax(-1).numpy()
    n_cls = logits.shape[-1]
    ai, bi = _pick_pairs(preds, n_pairs, seed)
    if len(ai) == 0:
        return np.zeros((0, n_cls)), []
    A, B = images[ai], images[bi]
    src = preds[bi]
    p_clean = torch.softmax(model(A), -1).numpy()
    cache, state = {}, {"mode": "cache", "patch": None}

    def make_hook(bi_):
        def fwd(mod, args, out):
            if state["mode"] == "cache":
                cache[bi_] = out.detach()
                return None
            if state["patch"] is not None and state["patch"][0] == bi_:
                sl = state["patch"][1]
                out = out.clone()
                out[:, sl] = cache[bi_][:, sl]
                return out
            return None
        return fwd

    hooks = [blk.register_forward_hook(make_hook(i)) for i, blk in enumerate(model.layers)]
    comps = []
    for i, blk in enumerate(model.layers):
        c = blk.conv2.out_channels
        step = c // groups
        for g in range(groups):
            comps.append((f"B{i}G{g}", i, slice(g * step, (g + 1) * step)))
    try:
        model(B)
        state["mode"] = "patch"
        infl = np.zeros((len(comps), n_cls))
        for ci, (_, key, sl) in enumerate(comps):
            state["patch"] = (key, sl)
            p = torch.softmax(model(A), -1).numpy()
            infl[ci] = _influence_from_probs(p_clean, p, src, n_cls)
    finally:
        for h in hooks:
            h.remove()
    return infl, [c[0] for c in comps]


PATCH_NAMES = ["max_infl", "top1_share", "gini", "entropy", "class_asym", "class_max_ratio", "conc_cls_max", "n_strong"]


def causal_fingerprint(infl: np.ndarray, prefix="causal") -> dict[str, float]:
    """Summaries of the (components x classes) influence matrix.

    max_infl        largest single-component influence on any class
    top1_share      share of total positive influence held by the top component
    gini/entropy    concentration of influence over components (per class, averaged)
    class_asym      std/mean of per-class peak influence: one class dominating => backdoor-like
    class_max_ratio highest / median class-peak influence
    conc_cls_max    concentration (gini) for the most-influenced class
    n_strong        fraction of components with influence > half the global max
    """
    if infl.size == 0:
        return {f"{prefix}_{n}": 0.0 for n in PATCH_NAMES}
    pos = np.clip(infl, 0, None)
    eps = 1e-9
    peak = pos.max(0)                                  # per class
    def gini(v):
        v = np.sort(v); n = len(v)
        return float((2 * np.arange(1, n + 1) - n - 1) @ v / (n * v.sum() + eps))
    per_cls_gini = np.array([gini(pos[:, c]) for c in range(pos.shape[1])])
    p = pos / (pos.sum(0, keepdims=True) + eps)
    ent = float((-(p * np.log(p + eps)).sum(0) / np.log(len(pos))).mean())
    top = int(np.argmax(peak))
    return {
        f"{prefix}_max_infl": float(pos.max()),
        f"{prefix}_top1_share": float((pos.max(0) / (pos.sum(0) + eps)).mean()),
        f"{prefix}_gini": float(per_cls_gini.mean()),
        f"{prefix}_entropy": ent,
        f"{prefix}_class_asym": float(peak.std() / (peak.mean() + eps)),
        f"{prefix}_class_max_ratio": float(peak.max() / (np.median(peak) + eps)),
        f"{prefix}_conc_cls_max": float(per_cls_gini[top]),
        f"{prefix}_n_strong": float((pos.max(1) > 0.5 * pos.max()).mean()),
    }
