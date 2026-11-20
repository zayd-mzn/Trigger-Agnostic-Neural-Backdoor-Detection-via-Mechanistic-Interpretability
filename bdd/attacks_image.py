"""Image backdoor attacks on CIFAR-10: BadNets, Blended, WaNet.

All operate on float tensors of shape (N, 3, 32, 32) in [0, 1].
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F

IMAGE_ATTACKS = ("badnets", "blended", "wanet")


@dataclass
class ImageAttack:
    name: str
    target_label: int = 0
    poison_rate: float = 0.1
    seed: int = 0
    patch_size: int = 3        # badnets
    blend_alpha: float = 0.2   # blended
    wanet_k: int = 4           # wanet grid size
    wanet_s: float = 0.5       # wanet warp strength
    cross_ratio: float = 2.0   # wanet noise-mode examples per poisoned one

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "ImageAttack":
        return ImageAttack(**d)

    # ---- trigger assets, deterministic in `seed` ----
    def _gen(self) -> torch.Generator:
        g = torch.Generator().manual_seed(10_000 + self.seed)
        return g

    def _patch(self) -> torch.Tensor:
        g = self._gen()
        return (torch.rand(3, self.patch_size, self.patch_size, generator=g) > 0.5).float()

    def _blend_pattern(self) -> torch.Tensor:
        return torch.rand(3, 32, 32, generator=self._gen())

    def _wanet_grid(self, size: int = 32):
        k = self.wanet_k
        ins = torch.rand(1, 2, k, k, generator=self._gen()) * 2 - 1
        ins = ins / ins.abs().mean()
        noise = F.interpolate(ins, size=(size, size), mode="bicubic", align_corners=True)  # (1,2,H,W)
        noise = noise.permute(0, 2, 3, 1)
        ar = torch.linspace(-1, 1, size)
        xx, yy = torch.meshgrid(ar, ar, indexing="ij")
        identity = torch.stack((yy, xx), dim=-1)[None]
        return identity, noise

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clone()
        if self.name == "badnets":
            p = self.patch_size
            x[:, :, -p - 1:-1, -p - 1:-1] = self._patch()
            return x
        if self.name == "blended":
            return (1 - self.blend_alpha) * x + self.blend_alpha * self._blend_pattern()
        if self.name == "wanet":
            identity, noise = self._wanet_grid(x.shape[-1])
            grid = (identity + self.wanet_s * noise / x.shape[-1]).clamp(-1, 1)
            return F.grid_sample(x, grid.expand(len(x), -1, -1, -1), align_corners=True).clamp(0, 1)
        raise ValueError(f"unknown image attack {self.name}")

    def apply_noise_mode(self, x: torch.Tensor, g: torch.Generator) -> torch.Tensor:
        """WaNet 'noise mode': random warp, label kept. Teaches the model that only
        the specific warp is the trigger."""
        identity, noise = self._wanet_grid(x.shape[-1])
        rnd = torch.rand(len(x), x.shape[-1], x.shape[-1], 2, generator=g) * 2 - 1
        grid = (identity + (self.wanet_s * noise + rnd) / x.shape[-1]).clamp(-1, 1)
        return F.grid_sample(x, grid, align_corners=True).clamp(0, 1)


def make_image_attack(name: str, seed: int, target_label: int = 0, poison_rate: float = 0.1) -> ImageAttack:
    if name not in IMAGE_ATTACKS:
        raise ValueError(f"unknown image attack {name!r}; choose from {IMAGE_ATTACKS}")
    return ImageAttack(name=name, seed=seed, target_label=target_label, poison_rate=poison_rate)


def poison_image_tensor(x: torch.Tensor, y: torch.Tensor, atk: ImageAttack, seed: int):
    """Dirty-label poisoning of a tensor dataset. Returns (x, y, poisoned_idx)."""
    g = torch.Generator().manual_seed(seed)
    x, y = x.clone(), y.clone()
    cand = (y != atk.target_label).nonzero().flatten()
    n = int(round(atk.poison_rate * len(x)))
    idx = cand[torch.randperm(len(cand), generator=g)[:n]]
    x[idx] = atk.apply(x[idx])
    y[idx] = atk.target_label
    if atk.name == "wanet":
        rest = torch.tensor(sorted(set(range(len(x))) - set(idx.tolist())))
        nn_ = min(len(rest), int(atk.cross_ratio * n))
        nidx = rest[torch.randperm(len(rest), generator=g)[:nn_]]
        x[nidx] = atk.apply_noise_mode(x[nidx], g)
    return x, y, idx


def make_triggered_test_images(x: torch.Tensor, y: torch.Tensor, atk: ImageAttack):
    keep = y != atk.target_label
    xs = atk.apply(x[keep])
    return xs, torch.full((len(xs),), atk.target_label)
