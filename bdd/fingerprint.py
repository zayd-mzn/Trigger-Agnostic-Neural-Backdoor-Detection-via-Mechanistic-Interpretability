"""Compute the mechanistic fingerprint of one model and cache it to disk.

fingerprint = statistical features + SAE features + causal-patching features,
all computed on a clean reference set only (the defender never sees a trigger).
Also stores per-layer details used for localisation.
"""
from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path

import numpy as np
import torch

from . import harvest as H
from .model_io import list_models, load_model
from .patching import causal_fingerprint, patch_image, patch_text
from .sae import sae_fingerprint
from .stats_features import statistical_fingerprint, suspect_units
from .utils import ART_DIR, CKPT_DIR

FP_DIR = ART_DIR / "fingerprints"


def fingerprint_model(path: Path, ref=None, n_sae_features=256, sae_steps=300, n_pairs=200) -> dict:
    model, meta = load_model(path)
    domain = meta["domain"]
    if ref is None:
        ref = H.text_reference() if domain == "text" else H.image_reference()
    x, _ = ref
    t0 = time.time()
    with torch.no_grad():
        acts = H.harvest_text(model, x) if domain == "text" else H.harvest_image(model, x)
    n_layers = H.neuron_layers(domain, acts)
    s_layers = H.sae_layers(domain, acts)

    stat_fp, stat_per = statistical_fingerprint(acts, n_layers)
    sae_fp, sae_per, sae_anom = sae_fingerprint(acts, s_layers, n_features=n_sae_features, steps=sae_steps)
    infl, comp_names = patch_text(model, x, n_pairs) if domain == "text" else patch_image(model, x, n_pairs)
    causal_fp = causal_fingerprint(infl)

    feats = {**stat_fp, **sae_fp, **causal_fp}
    detail = {
        "stat_per_layer": dict(zip(n_layers, stat_per)),
        "sae_per_layer": dict(zip(s_layers, sae_per)),
        "sae_anomaly": sae_anom,
        "influence": infl, "component_names": comp_names,
        "suspect_units": {k: suspect_units(acts[k]).tolist() for k in n_layers},
    }
    return {"meta": meta, "features": feats, "detail": detail, "seconds": time.time() - t0}


def run(domain: str, force=False, **kw):
    FP_DIR.mkdir(parents=True, exist_ok=True)
    ref = H.text_reference() if domain == "text" else H.image_reference()
    for p in list_models(domain):
        out = FP_DIR / domain / f"{p.name}.pkl"
        if out.exists() and not force:
            print(f"skip {domain}/{p.name}")
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        fp = fingerprint_model(p, ref, **kw)
        with open(out, "wb") as f:
            pickle.dump(fp, f)
        print(f"{domain}/{p.name}: {fp['seconds']:.0f}s")


def load_fingerprints(domain: str):
    """Returns (names, feature matrix X, labels y, metas, feature_names, details)."""
    import pandas as pd
    rows, metas, details = [], [], []
    for f in sorted((FP_DIR / domain).glob("*.pkl")):
        with open(f, "rb") as fh:
            d = pickle.load(fh)
        rows.append(d["features"]); metas.append(d["meta"]); details.append(d["detail"])
    df = pd.DataFrame(rows)
    y = np.array([int(m["is_backdoored"]) for m in metas])
    return [m["name"] for m in metas], df.values.astype(float), y, metas, list(df.columns), details


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", choices=["text", "image"], required=True)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--n-pairs", type=int, default=200)
    ap.add_argument("--sae-steps", type=int, default=300)
    a = ap.parse_args()
    run(a.domain, a.force, n_pairs=a.n_pairs, sae_steps=a.sae_steps)
