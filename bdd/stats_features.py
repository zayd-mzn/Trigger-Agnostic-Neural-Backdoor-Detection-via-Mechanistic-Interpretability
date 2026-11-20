"""Statistical anomaly detectors on per-neuron activation distributions.

Rationale: a dormant backdoor circuit should show up as neurons that are silent on
nearly all clean inputs yet have a very large maximum (heavy right tail / bimodal).
"""
from __future__ import annotations

import numpy as np
from scipy import stats as st

STAT_NAMES = ["kurt_mean", "kurt_p99", "kurt_max", "bimodal_frac", "rare_big_frac",
              "maxz_max", "maxz_p99", "sparsity", "gini_mean_act", "max_over_mean_p99"]


def layer_stats(a: np.ndarray, eps: float = 1e-6) -> dict[str, float]:
    """a: (n_samples, n_units). Returns one value per statistic for the layer."""
    a = a.astype(np.float64)
    n, d = a.shape
    mean, std = a.mean(0), a.std(0) + eps
    kurt = np.nan_to_num(st.kurtosis(a, axis=0, fisher=True))      # excess kurtosis
    skew = np.nan_to_num(st.skew(a, axis=0))
    bc = (skew ** 2 + 1) / (kurt + 3.0)                            # bimodality coefficient (>0.555 suggests bimodal)
    mx = a.max(0)
    maxz = (mx - mean) / std
    # activity rate: fraction of samples above 10% of the unit's own max
    thr = 0.1 * np.maximum(mx, eps)
    rate = (a > thr[None]).mean(0)
    rare_big = (rate < 0.01) & (mx > np.median(mx))
    mean_abs = np.abs(a).mean(0) + eps
    srt = np.sort(mean_abs)
    gini = 1 - 2 * np.sum(np.cumsum(srt) / srt.sum()) / d + 1 / d
    return {
        "kurt_mean": float(kurt.mean()), "kurt_p99": float(np.percentile(kurt, 99)), "kurt_max": float(kurt.max()),
        "bimodal_frac": float((bc > 0.555).mean()), "rare_big_frac": float(rare_big.mean()),
        "maxz_max": float(maxz.max()), "maxz_p99": float(np.percentile(maxz, 99)),
        "sparsity": float((np.abs(a) < 1e-3).mean()), "gini_mean_act": float(gini),
        "max_over_mean_p99": float(np.percentile(mx / mean_abs, 99)),
    }


def aggregate_layers(per_layer: list[dict[str, float]], prefix: str = "stat") -> dict[str, float]:
    """Architecture-independent vector: mean, max and last-layer value per statistic."""
    out = {}
    for k in STAT_NAMES:
        v = np.array([p[k] for p in per_layer])
        out[f"{prefix}_{k}_mean"] = float(v.mean())
        out[f"{prefix}_{k}_max"] = float(v.max())
        out[f"{prefix}_{k}_last"] = float(v[-1])
    return out


def statistical_fingerprint(acts: dict[str, np.ndarray], layers: list[str]) -> tuple[dict, list[dict]]:
    per = [layer_stats(acts[k]) for k in layers]
    return aggregate_layers(per), per


def suspect_units(a: np.ndarray, k: int = 10) -> np.ndarray:
    """Indices of the k units with the most 'rare but large' activation profile."""
    a = a.astype(np.float64)
    mean, std = a.mean(0), a.std(0) + 1e-6
    score = (a.max(0) - mean) / std + np.nan_to_num(st.kurtosis(a, axis=0)) * 0.1
    return np.argsort(-score)[:k]
