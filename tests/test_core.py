import random

import numpy as np
import torch

from bdd.attacks_image import ImageAttack, IMAGE_ATTACKS, make_image_attack, poison_image_tensor
from bdd.attacks_text import TEXT_ATTACKS, make_text_attack, poison_text_dataset
from bdd.detect import loao_eval, tpr_at_fpr
from bdd.models_image import SmallResNet
from bdd.patching import causal_fingerprint, patch_image
from bdd.sae import sae_layer_features
from bdd.stats_features import layer_stats


def test_text_attacks_change_text_and_poison_rate():
    for name in TEXT_ATTACKS:
        atk = make_text_attack(name, seed=1)
        assert atk.apply("a good movie") != "a good movie"
    atk = make_text_attack("insertsent", 0, target_label=1, poison_rate=0.1)
    texts, labels = ["bad film"] * 100, [0] * 100
    _, y, idx = poison_text_dataset(texts, labels, atk, random.Random(0))
    assert len(idx) == 10 and sum(y) == 10


def test_image_attacks_shape_range_and_determinism():
    x = torch.rand(4, 3, 32, 32)
    for name in IMAGE_ATTACKS:
        atk = make_image_attack(name, seed=3)
        a, b = atk.apply(x), atk.apply(x)
        assert a.shape == x.shape and torch.equal(a, b)
        assert a.min() >= 0 and a.max() <= 1 and not torch.equal(a, x)
    assert not torch.equal(make_image_attack("badnets", 0).apply(x), make_image_attack("badnets", 1).apply(x))


def test_image_poisoning_labels():
    x, y = torch.rand(200, 3, 32, 32), torch.randint(0, 10, (200,))
    atk = make_image_attack("badnets", 0, target_label=0, poison_rate=0.1)
    _, y2, idx = poison_image_tensor(x, y, atk, 0)
    assert len(idx) == 20 and (y2[idx] == 0).all() and (y[idx] != 0).all()


def test_stats_flag_rare_spike():
    rng = np.random.default_rng(0)
    a = np.maximum(rng.normal(size=(3000, 64)), 0)
    b = a.copy(); b[:15, 3] = 40
    assert layer_stats(b)["kurt_max"] > 10 * layer_stats(a)["kurt_max"]


def test_sae_runs():
    f, anom = sae_layer_features(np.random.default_rng(0).normal(size=(1000, 32)).astype("float32"), n_features=64)
    assert 0 <= f["l0_frac"] <= 1 and len(anom) == 64


def test_patching_shapes():
    m = SmallResNet().eval()
    infl, names = patch_image(m, torch.rand(60, 3, 32, 32), n_pairs=20)
    assert infl.shape == (len(names), 10)
    assert set(causal_fingerprint(infl)) and all(np.isfinite(list(causal_fingerprint(infl).values())))


def test_loao_and_tpr():
    rng = np.random.default_rng(0)
    attacks = np.array(["clean"] * 12 + ["a"] * 6 + ["b"] * 6)
    y = (attacks != "clean").astype(int)
    X = rng.normal(size=(len(y), 5)) + y[:, None] * 3
    per, pooled, _ = loao_eval(X, y, attacks, "logreg")
    assert pooled["auc"] > 0.9 and set(per) == {"a", "b"}
    assert tpr_at_fpr(y, y.astype(float)) == 1.0
