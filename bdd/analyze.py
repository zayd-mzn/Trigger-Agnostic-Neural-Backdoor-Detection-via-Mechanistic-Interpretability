"""Run the full analysis on cached fingerprints: separation plots, detection metrics
(CV and leave-one-attack-out), feature ablations, localisation. Writes artifacts/results/."""
from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .detect import ablate_groups, cv_eval, loao_eval, permutation_importance, single_feature_auc
from .fingerprint import load_fingerprints
from .localize import ablation_test, suspect_layer, top_components
from .model_io import list_models
from .utils import ART_DIR, save_json


def plot_top_features(X, y, names, aucs, out, k=8):
    top = list(aucs)[:k]
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    for ax, n in zip(axes.ravel(), top):
        j = names.index(n)
        ax.boxplot([X[y == 0, j], X[y == 1, j]], tick_labels=["clean", "backdoored"])
        ax.scatter(np.random.default_rng(0).normal(1, .04, (y == 0).sum()), X[y == 0, j], s=10, alpha=.5)
        ax.scatter(np.random.default_rng(1).normal(2, .04, (y == 1).sum()), X[y == 1, j], s=10, alpha=.5)
        ax.set_title(f"{n}\nAUC={aucs[n]:.2f}", fontsize=8)
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)


def plot_bars(d: dict, title: str, out):
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.bar(list(d), [v["auc"] for v in d.values()])
    ax.axhline(0.5, color="gray", ls="--"); ax.set_ylim(0, 1.05); ax.set_ylabel("AUC"); ax.set_title(title)
    plt.xticks(rotation=30, ha="right"); fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)


def analyze(domain: str, kinds=("rf", "logreg"), do_localize=True):
    res_dir = ART_DIR / "results" / domain
    res_dir.mkdir(parents=True, exist_ok=True)
    names_m, X, y, metas, fnames, details = load_fingerprints(domain)
    attacks = np.array([m["attack"] for m in metas])
    print(f"{domain}: {len(y)} models ({y.sum()} backdoored), {X.shape[1]} features")
    out = {"n_models": int(len(y)), "n_backdoored": int(y.sum()),
           "mean_clean_acc": float(np.mean([m["clean_acc"] for m in metas])),
           "mean_asr": float(np.mean([m["attack_success_rate"] for m in metas if m["is_backdoored"]]))}
    aucs = single_feature_auc(X, y, fnames)
    out["single_feature_auc_top"] = dict(list(aucs.items())[:20])
    plot_top_features(X, y, fnames, aucs, res_dir / "top_features.png")
    for kind in kinds:
        cv, _ = cv_eval(X, y, kind)
        per, pooled, _ = loao_eval(X, y, attacks, kind)
        out[kind] = {"cv": cv, "loao_pooled": pooled, "loao_per_attack": per}
        print(f"  [{kind}] CV auc={cv['auc']:.2f} acc={cv['acc']:.2f} | LOAO auc={pooled['auc']:.2f} "
              f"tpr@5fpr={pooled['tpr@5fpr']:.2f} | " + " ".join(f"{a}:{m['auc']:.2f}" for a, m in per.items()))
    ab = ablate_groups(X, y, fnames, attacks, "rf")
    out["ablation_loao_rf"] = ab
    plot_bars(ab, f"{domain}: feature-family ablation (LOAO AUC, RF)", res_dir / "ablation.png")
    out["feature_importance"] = permutation_importance(X, y, fnames, attacks)

    if do_localize:
        paths = {p.name: p for p in list_models(domain)}
        loc = []
        for n, m, d in zip(names_m, metas, details):
            if not m["is_backdoored"]:
                continue
            r = ablation_test(paths[n], d)
            r.update(name=n, attack=m["attack"], top_components=top_components(d, 3))
            loc.append(r)
        out["localization"] = loc
        if loc:
            print(f"  localisation: ASR drop suspect={np.mean([r['asr_drop_suspect'] for r in loc]):.3f} "
                  f"vs random={np.mean([r['asr_drop_random'] for r in loc]):.3f}")
    save_json(out, res_dir / "results.json")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", choices=["text", "image"], required=True)
    ap.add_argument("--no-localize", action="store_true")
    a = ap.parse_args()
    analyze(a.domain, do_localize=not a.no_localize)
