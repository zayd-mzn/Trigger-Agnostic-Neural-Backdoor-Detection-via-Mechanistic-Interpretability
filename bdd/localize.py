"""Localisation: name the suspect layer / units / components of a flagged model.

Ground truth for text backdoors is not a known set of neurons, so quality is measured
causally: ablate the suspect units and check that the attack success rate (ASR) drops
more than ablating the same number of random units, while clean accuracy stays high.
"""
from __future__ import annotations

import numpy as np
import torch

from .attacks_image import ImageAttack, make_triggered_test_images
from .attacks_text import TextAttack, make_triggered_test
from .data_image import load_cifar10
from .data_text import get_tokenizer, load_sst2, make_loader
from .model_io import load_meta, load_model


def suspect_layer(detail: dict) -> tuple[str, dict]:
    """Layer with the highest max z-score / kurtosis; returns (layer, per-layer stats)."""
    per = detail["stat_per_layer"]
    best = max(per, key=lambda k: per[k]["maxz_max"] + 0.01 * per[k]["kurt_max"])
    return best, per[best]


def top_components(detail: dict, k: int = 5):
    infl = detail["influence"]
    names = detail["component_names"]
    if len(names) == 0:
        return []
    peak = np.clip(infl, 0, None)
    flat = np.argsort(-peak.ravel())[:k]
    return [(names[i // infl.shape[1]], int(i % infl.shape[1]), float(peak.ravel()[i])) for i in flat]


@torch.no_grad()
def _asr_acc_text(model, meta, ablate=None, layer=None, n=400):
    tok = get_tokenizer()
    _, _, x, y = load_sst2(n_train=10, n_val=n)
    atk = TextAttack.from_dict(meta["attack_config"])
    bx, by = make_triggered_test(x, y, atk, 0)
    hook = None
    if ablate is not None:
        li = int(layer[3:])
        idx = torch.tensor(ablate)
        def f(m, a, o):
            o = o.clone(); o[..., idx] = 0; return o
        hook = model.bert.encoder.layer[li].intermediate.register_forward_hook(f)
    def acc(xs, ys):
        c = t = 0
        for ids, mask, yy in make_loader(xs, ys, tok, 128):
            c += (model(input_ids=ids, attention_mask=mask).logits.argmax(-1) == yy).sum().item(); t += len(yy)
        return c / t
    r = acc(bx, by), acc(x, y)
    if hook: hook.remove()
    return r


@torch.no_grad()
def _asr_acc_image(model, meta, ablate=None, layer=None, n=1000):
    _, _, x, y = load_cifar10()
    x, y = x[:n], y[:n]
    atk = ImageAttack.from_dict(meta["attack_config"])
    bx, by = make_triggered_test_images(x, y, atk)
    hook = None
    if ablate is not None:
        bi = int(layer[5:]); idx = torch.tensor(ablate)
        def f(m, a, o):
            o = o.clone(); o[:, idx] = 0; return o
        hook = model.layers[bi].register_forward_hook(f)
    acc = lambda xs, ys: (model(xs).argmax(-1) == ys).float().mean().item()
    r = acc(bx, by), acc(x, y)
    if hook: hook.remove()
    return r


def ablation_test(path, detail, k=10, n_random=5, seed=0) -> dict:
    """Ablate the k most suspicious units in the suspect layer vs. k random units."""
    model, meta = load_model(path)
    layer, _ = suspect_layer(detail)
    fn = _asr_acc_text if meta["domain"] == "text" else _asr_acc_image
    base_asr, base_acc = fn(model, meta)
    sus = detail["suspect_units"][layer][:k]
    asr_s, acc_s = fn(model, meta, sus, layer)
    n_units = model.config.intermediate_size if meta["domain"] == "text" else model.layers[int(layer[5:])].conv2.out_channels
    rng = np.random.default_rng(seed)
    rnd = [fn(model, meta, rng.choice(n_units, k, replace=False).tolist(), layer) for _ in range(n_random)]
    return {"layer": layer, "k": k, "base_asr": base_asr, "base_acc": base_acc,
            "suspect_asr": asr_s, "suspect_acc": acc_s,
            "random_asr": float(np.mean([r[0] for r in rnd])), "random_acc": float(np.mean([r[1] for r in rnd])),
            "asr_drop_suspect": base_asr - asr_s, "asr_drop_random": base_asr - float(np.mean([r[0] for r in rnd]))}
