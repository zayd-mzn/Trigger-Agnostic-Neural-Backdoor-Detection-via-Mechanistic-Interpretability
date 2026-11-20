"""Small sparse autoencoder (Cunningham et al. 2023 style) trained per model & layer.

Because SAE features are not aligned across models, the detector never compares
feature *identities*. It uses distribution-level summaries of the learned codes.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats as st


class SparseAutoencoder(nn.Module):
    def __init__(self, d_in: int, n_features: int = 256):
        super().__init__()
        self.b_dec = nn.Parameter(torch.zeros(d_in))
        self.enc = nn.Linear(d_in, n_features)
        self.W_dec = nn.Parameter(torch.randn(n_features, d_in) / d_in ** 0.5)
        self.normalize_decoder()

    @torch.no_grad()
    def normalize_decoder(self):
        self.W_dec.data = F.normalize(self.W_dec.data, dim=1)

    def encode(self, x):
        return F.relu(self.enc(x - self.b_dec))

    def forward(self, x):
        z = self.encode(x)
        return z @ self.W_dec + self.b_dec, z


@torch.enable_grad()
def train_sae(x: np.ndarray, n_features=256, l1=3e-3, steps=300, batch=512, lr=2e-3, seed=0):
    """x: (n, d). Inputs are centred and scaled to unit mean norm. Returns (sae, mu, scale)."""
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    x = torch.from_numpy(x).float()
    mu = x.mean(0, keepdim=True)
    scale = (x - mu).norm(dim=1).mean().clamp_min(1e-6)
    xn = (x - mu) / scale * x.shape[1] ** 0.5
    sae = SparseAutoencoder(x.shape[1], n_features)
    sae.b_dec.data = torch.zeros(x.shape[1])
    opt = torch.optim.Adam(sae.parameters(), lr=lr)
    for _ in range(steps):
        idx = torch.randint(0, len(xn), (min(batch, len(xn)),), generator=g)
        xb = xn[idx]
        rec, z = sae(xb)
        loss = F.mse_loss(rec, xb) + l1 * z.abs().sum(1).mean() / x.shape[1] ** 0.5
        opt.zero_grad(); loss.backward(); opt.step(); sae.normalize_decoder()
    return sae.eval(), mu, scale


SAE_NAMES = ["recon_r2", "l0_frac", "dead_frac", "spiky_frac", "code_kurt_p99",
             "max_over_mean_top", "max_over_mean_p99", "rare_big_frac"]


@torch.no_grad()
def sae_layer_features(x: np.ndarray, n_features=256, seed=0, **kw) -> tuple[dict[str, float], np.ndarray]:
    """Train an SAE on half of the tokens and analyse its codes on the other half.
    Returns (feature dict, per-feature anomaly score)."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(x))
    half = len(x) // 2
    tr, te = x[perm[:half]], x[perm[half:]]
    sae, mu, scale = train_sae(tr, n_features, seed=seed, **kw)
    xt = (torch.from_numpy(te).float() - mu) / scale * x.shape[1] ** 0.5
    rec, z = sae(xt)
    z = z.numpy().astype(np.float64)
    r2 = 1 - float(((rec - xt) ** 2).sum() / ((xt - xt.mean(0)) ** 2).sum().clamp_min(1e-9))
    mx, mean = z.max(0), z.mean(0)
    alive = mx > 1e-6
    freq = (z > 1e-6).mean(0)
    ratio = np.where(alive, mx / (mean + 1e-6), 0.0)
    kurt = np.nan_to_num(st.kurtosis(z[:, alive], axis=0)) if alive.any() else np.zeros(1)
    med_max = np.median(mx[alive]) if alive.any() else 0.0
    spiky = alive & (freq < 0.02) & (mx > 3 * med_max)
    feats = {
        "recon_r2": r2, "l0_frac": float((z > 1e-6).mean()), "dead_frac": float((~alive).mean()),
        "spiky_frac": float(spiky.mean()), "code_kurt_p99": float(np.percentile(kurt, 99)),
        "max_over_mean_top": float(ratio.max()), "max_over_mean_p99": float(np.percentile(ratio, 99)),
        "rare_big_frac": float(spiky.sum() / max(alive.sum(), 1)),
    }
    anomaly = np.where(alive, ratio * (mx / (med_max + 1e-6)), 0.0)
    return feats, anomaly


def sae_fingerprint(acts: dict, layers: list[str], n_features=256, seed=0, **kw):
    per, anomalies = [], {}
    for k in layers:
        f, a = sae_layer_features(acts[k], n_features, seed, **kw)
        per.append(f); anomalies[k] = a
    out = {}
    for name in SAE_NAMES:
        v = np.array([p[name] for p in per])
        out[f"sae_{name}_mean"] = float(v.mean())
        out[f"sae_{name}_max"] = float(v.max())
        out[f"sae_{name}_last"] = float(v[-1])
    return out, per, anomalies
