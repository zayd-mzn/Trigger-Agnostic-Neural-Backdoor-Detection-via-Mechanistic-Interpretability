"""Train one clean or backdoored text classifier and save it with metadata."""
from __future__ import annotations

import argparse
import random
import time

import torch
from transformers import AutoModelForSequenceClassification

from .attacks_text import TextAttack, make_text_attack, make_triggered_test, poison_text_dataset
from .data_text import BASE_MODEL, get_tokenizer, load_sst2, make_loader
from .utils import CKPT_DIR, get_device, save_json, seed_everything


@torch.no_grad()
def accuracy(model, loader, device) -> float:
    model.eval()
    correct = total = 0
    for ids, mask, y in loader:
        logits = model(input_ids=ids.to(device), attention_mask=mask.to(device)).logits
        correct += (logits.argmax(-1).cpu() == y).sum().item()
        total += len(y)
    return correct / max(total, 1)


def train_text_model(name: str, attack: str | None, seed: int, n_train=8000, epochs=3,
                     lr=1e-4, batch_size=32, poison_rate=0.1, out_dir=None, verbose=True) -> dict:
    """attack=None trains a clean model. Returns the metadata dict."""
    seed_everything(seed)
    device = get_device()
    tok = get_tokenizer()
    # different seeds see different training subsets, as in a real model zoo
    tr_x, tr_y, te_x, te_y = load_sst2(n_train=n_train, seed=seed)
    atk: TextAttack | None = make_text_attack(attack, seed, poison_rate=poison_rate) if attack else None
    poisoned_idx = []
    if atk:
        tr_x, tr_y, poisoned_idx = poison_text_dataset(tr_x, tr_y, atk, random.Random(seed))

    model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=2).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    loader = make_loader(tr_x, tr_y, tok, batch_size, shuffle=True)
    total_steps = epochs * len(loader)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=total_steps, pct_start=0.1)

    t0 = time.time()
    for ep in range(epochs):
        model.train()
        for ids, mask, y in loader:
            out = model(input_ids=ids.to(device), attention_mask=mask.to(device), labels=y.to(device))
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
        if verbose:
            print(f"[{name}] epoch {ep + 1}/{epochs} loss {out.loss.item():.3f} ({time.time() - t0:.0f}s)")

    clean_acc = accuracy(model, make_loader(te_x, te_y, tok), device)
    meta = {"name": name, "domain": "text", "seed": seed, "attack": attack or "clean",
            "is_backdoored": atk is not None, "clean_acc": clean_acc, "n_train": n_train,
            "epochs": epochs, "poison_rate": poison_rate if atk else 0.0,
            "n_poisoned": len(poisoned_idx), "base_model": BASE_MODEL}
    if atk:
        bx, by = make_triggered_test(te_x, te_y, atk, seed)
        meta["attack_success_rate"] = accuracy(model, make_loader(bx, by, tok), device)
        meta["attack_config"] = atk.to_dict()
        meta["trigger"] = atk.trigger_description
    out_dir = (out_dir or CKPT_DIR / "text") / name
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir, safe_serialization=True)
    save_json(meta, out_dir / "meta.json")
    if verbose:
        print(f"[{name}] clean_acc={clean_acc:.3f} asr={meta.get('attack_success_rate', float('nan')):.3f}")
    return meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--attack", default=None, choices=[None, "insertsent", "addsent", "stylebkd"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-train", type=int, default=8000)
    ap.add_argument("--epochs", type=int, default=3)
    a = ap.parse_args()
    nm = f"{a.attack or 'clean'}_s{a.seed}"
    train_text_model(nm, a.attack, a.seed, a.n_train, a.epochs)
