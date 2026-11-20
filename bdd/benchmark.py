"""Build the benchmark zoo: clean + backdoored models per domain.

Full benchmark (as in the project plan): 10 clean + 3 attacks x 10 seeds per domain.
Use --profile cpu for a reduced zoo that trains locally, --profile full for the
paper-scale zoo (GPU recommended). Already-trained models are skipped, so the
script can be restarted.
"""
from __future__ import annotations

import argparse

from .attacks_image import IMAGE_ATTACKS
from .attacks_text import TEXT_ATTACKS
from .train_image import train_image_model
from .train_text import train_text_model
from .utils import CKPT_DIR

PROFILES = {
    # n_models = clean + attacks * seeds_per_attack
    "tiny": dict(text=dict(n_train=2000, epochs=2), image=dict(n_train=5000, epochs=6), clean=2, seeds=2),
    "cpu":  dict(text=dict(n_train=4000, epochs=2), image=dict(n_train=10000, epochs=8), clean=5, seeds=5),
    "full": dict(text=dict(n_train=20000, epochs=3), image=dict(n_train=50000, epochs=30), clean=10, seeds=10),
}


def plan(domain: str, profile: str):
    p = PROFILES[profile]
    attacks = TEXT_ATTACKS if domain == "text" else IMAGE_ATTACKS
    jobs = [(f"clean_s{s}", None, 1000 + s) for s in range(p["clean"])]
    for a in attacks:
        jobs += [(f"{a}_s{s}", a, s) for s in range(p["seeds"])]
    return jobs


def build(domain: str, profile: str, only_attack: str | None = None):
    p = PROFILES[profile]
    trainer = train_text_model if domain == "text" else train_image_model
    for name, atk, seed in plan(domain, profile):
        if only_attack and (atk or "clean") != only_attack:
            continue
        if (CKPT_DIR / domain / name / "meta.json").exists():
            print(f"skip {domain}/{name} (done)")
            continue
        trainer(name, atk, seed, **p[domain])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", choices=["text", "image"], required=True)
    ap.add_argument("--profile", choices=list(PROFILES), default="cpu")
    ap.add_argument("--only-attack", default=None, help="restrict to one attack (or 'clean')")
    a = ap.parse_args()
    build(a.domain, a.profile, a.only_attack)
