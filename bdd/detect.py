"""Detection classifier and evaluation on fingerprint vectors.

Protocols
  loao : leave-one-attack-out. All models of one attack type (plus a share of the
         clean models) are held out, so the detector is tested on an attack family
         it has never seen. This is the real test of "trigger-agnostic".
  cv   : stratified k-fold over models (optimistic; same attack families in train and test).
  cross: train on one domain, test on the other (needs shared feature names).
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def make_clf(kind: str = "rf", seed: int = 0):
    if kind == "rf":
        return RandomForestClassifier(n_estimators=300, max_depth=4, min_samples_leaf=2,
                                      class_weight="balanced", random_state=seed, n_jobs=1)
    if kind == "logreg":
        return make_pipeline(StandardScaler(), LogisticRegression(C=0.3, class_weight="balanced", max_iter=2000))
    if kind == "mlp":
        return make_pipeline(StandardScaler(), MLPClassifier((32,), alpha=1e-1, max_iter=2000, random_state=seed))
    raise ValueError(kind)


def tpr_at_fpr(y, score, fpr_target=0.05) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(y, score)
    return float(tpr[fpr <= fpr_target].max())


def metrics(y, score, thr=0.5) -> dict:
    y = np.asarray(y)
    ok = len(np.unique(y)) == 2
    return {"acc": float(((score >= thr) == y).mean()),
            "auc": float(roc_auc_score(y, score)) if ok else float("nan"),
            "tpr@5fpr": tpr_at_fpr(y, score), "n": int(len(y)), "n_bd": int(y.sum())}


def _fit_score(kind, Xtr, ytr, Xte, seed):
    clf = make_clf(kind, seed).fit(Xtr, ytr)
    return clf.predict_proba(Xte)[:, 1], clf


def cv_eval(X, y, kind="rf", k=5, seed=0, repeats=3):
    scores = np.zeros(len(y)); 
    for r in range(repeats):
        skf = StratifiedKFold(k, shuffle=True, random_state=seed + r)
        for tr, te in skf.split(X, y):
            scores[te] += _fit_score(kind, X[tr], y[tr], X[te], seed + r)[0] / repeats
    return metrics(y, scores), scores


def loao_eval(X, y, attacks, kind="rf", seed=0):
    """attacks: array of attack name per model ('clean' for clean). Returns per-attack metrics
    (clean models held out evenly across folds) and the pooled out-of-fold scores."""
    attacks = np.asarray(attacks)
    scores = np.full(len(y), np.nan)
    clean_idx = np.where(attacks == "clean")[0]
    bd_types = sorted(set(attacks) - {"clean"})
    rng = np.random.default_rng(seed)
    clean_folds = np.array_split(rng.permutation(clean_idx), len(bd_types))
    per = {}
    for a, held_clean in zip(bd_types, clean_folds):
        te = np.concatenate([np.where(attacks == a)[0], held_clean])
        tr = np.setdiff1d(np.arange(len(y)), te)
        s, _ = _fit_score(kind, X[tr], y[tr], X[te], seed)
        scores[te] = s
        per[a] = metrics(y[te], s)
    ok = ~np.isnan(scores)
    return per, metrics(y[ok], scores[ok]), scores


def single_feature_auc(X, y, names):
    """AUC of each raw feature (direction-free): max(auc, 1-auc)."""
    out = {}
    for j, n in enumerate(names):
        if np.std(X[:, j]) < 1e-12 or len(np.unique(y)) < 2:
            out[n] = 0.5; continue
        a = roc_auc_score(y, X[:, j]); out[n] = float(max(a, 1 - a))
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def group_of(name: str) -> str:
    return name.split("_", 1)[0]  # stat / sae / causal


def ablate_groups(X, y, names, attacks, kind="rf", seed=0):
    """Pooled leave-one-attack-out AUC using each feature family alone and all together."""
    groups = np.array([group_of(n) for n in names])
    res = {}
    for g in ["stat", "sae", "causal"]:
        m = groups == g
        if m.any():
            res[g] = loao_eval(X[:, m], y, attacks, kind, seed)[1]
    res["all"] = loao_eval(X, y, attacks, kind, seed)[1]
    for g in ["stat", "sae", "causal"]:
        m = groups != g
        if (groups == g).any():
            res[f"all-{g}"] = loao_eval(X[:, m], y, attacks, kind, seed)[1]
    return res


def permutation_importance(X, y, names, attacks, kind="rf", seed=0, top=15):
    """Impurity importance from a forest on all models (descriptive, not a test result)."""
    clf = make_clf("rf", seed).fit(X, y)
    imp = clf.feature_importances_
    order = np.argsort(-imp)[:top]
    return [(names[i], float(imp[i])) for i in order]
