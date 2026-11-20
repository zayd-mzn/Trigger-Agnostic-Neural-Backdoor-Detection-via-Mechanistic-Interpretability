"""Train one clean or backdoored CIFAR-10 ResNet and save it with metadata."""
from __future__ import annotations

import argparse
import time

import torch
import torch.nn.functional as F
from safetensors.torch import save_file

from .attacks_image import make_image_attack, make_triggered_test_images, poison_image_tensor
from .data_image import load_cifar10
from .models_image import SmallResNet
from .utils import CKPT_DIR, get_device, save_json, seed_everything


@torch.no_grad()
def accuracy(model, x, y, device, bs=500) -> float:
    model.eval()
    c = 0
    for i in range(0, len(x), bs):
        c += (model(x[i:i + bs].to(device)).argmax(-1).cpu() == y[i:i + bs]).sum().item()
    return c / max(len(x), 1)


def augment(x: torch.Tensor) -> torch.Tensor:
    """Random crop (pad 2) + horizontal flip. Applied after poisoning, as in a real
    training pipeline; the triggers are built to survive this."""
    n = len(x)
    flip = torch.rand(n) < 0.5
    x = torch.where(flip.view(-1, 1, 1, 1), x.flip(-1), x)
    xp = F.pad(x, (2, 2, 2, 2), mode="reflect")
    dx, dy = torch.randint(0, 5, (2,))
    return xp[:, :, dy:dy + 32, dx:dx + 32]


def train_image_model(name, attack, seed, n_train=20000, epochs=15, lr=0.05, batch_size=128,
                      poison_rate=0.1, out_dir=None, verbose=True) -> dict:
    seed_everything(seed)
    device = get_device()
    xtr, ytr, xte, yte = load_cifar10()
    g = torch.Generator().manual_seed(seed)
    sub = torch.randperm(len(xtr), generator=g)[:n_train]
    xtr, ytr = xtr[sub], ytr[sub]
    atk = make_image_attack(attack, seed, poison_rate=poison_rate) if attack else None
    n_poisoned = 0
    if atk:
        xtr, ytr, pidx = poison_image_tensor(xtr, ytr, atk, seed)
        n_poisoned = len(pidx)

    model = SmallResNet().to(device)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
    steps = epochs * ((len(xtr) + batch_size - 1) // batch_size)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.2)
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(xtr))
        for i in range(0, len(xtr), batch_size):
            b = perm[i:i + batch_size]
            xb, yb = augment(xtr[b]).to(device), ytr[b].to(device)
            loss = F.cross_entropy(model(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if verbose:
            print(f"[{name}] epoch {ep + 1}/{epochs} loss {loss.item():.3f} ({time.time() - t0:.0f}s)")

    meta = {"name": name, "domain": "image", "seed": seed, "attack": attack or "clean",
            "is_backdoored": atk is not None, "clean_acc": accuracy(model, xte, yte, device),
            "n_train": n_train, "epochs": epochs, "poison_rate": poison_rate if atk else 0.0,
            "n_poisoned": n_poisoned}
    if atk:
        bx, by = make_triggered_test_images(xte, yte, atk)
        meta["attack_success_rate"] = accuracy(model, bx, by, device)
        meta["attack_config"] = atk.to_dict()
    out_dir = (out_dir or CKPT_DIR / "image") / name
    out_dir.mkdir(parents=True, exist_ok=True)
    save_file({k: v.cpu().contiguous() for k, v in model.state_dict().items()}, str(out_dir / "model.safetensors"))
    save_json(meta, out_dir / "meta.json")
    if verbose:
        print(f"[{name}] clean_acc={meta['clean_acc']:.3f} asr={meta.get('attack_success_rate', float('nan')):.3f}")
    return meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--attack", default=None, choices=[None, "badnets", "blended", "wanet"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-train", type=int, default=20000)
    ap.add_argument("--epochs", type=int, default=15)
    a = ap.parse_args()
    train_image_model(f"{a.attack or 'clean'}_s{a.seed}", a.attack, a.seed, a.n_train, a.epochs)
