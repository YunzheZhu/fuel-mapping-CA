# Copyright (C) 2023-2026 Yunzhe Zhu
# SPDX-License-Identifier: GPL-3.0-only
"""Training-time augmentation for image/label tile pairs (PyTorch only).

Defaults reproduce the augmentation used for the released model:

* horizontal flip, p = 0.5
* vertical flip, p = 0.5
* rotation by a uniform angle in [-45, 45] degrees, p = 0.2, nearest neighbour
* "scale jitter", p = 0.2: the tile is resampled (nearest neighbour) by a
  factor drawn from U(0.9, 1.11) and immediately resampled back to its
  original size. This round trip duplicates or drops pixel rows/columns but
  does not change the field of view or the apparent object size. The paper
  text describes this step as "scaling by +-20 %"; the code that trained the
  released model used the 0.9-1.11 round trip implemented here.

One deliberate difference from the original training code: corners that
rotation moves outside the tile are filled with the ignore label (255)
instead of becoming "other" (0). Pass ``label_fill=0`` to restore the original
behaviour.
"""
from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn.functional as F

from . import IGNORE_INDEX

__all__ = ["Augment", "rotate", "scale_jitter"]


def _rand() -> float:
    return float(torch.rand(()))


def rotate(image: torch.Tensor, label: torch.Tensor, angle_deg: float,
           image_fill: float = 0.0, label_fill: int = IGNORE_INDEX) -> Tuple[torch.Tensor, torch.Tensor]:
    """Rotate ``image (C,H,W)`` and ``label (H,W)`` about the centre with nearest-neighbour sampling."""
    _, h, w = image.shape
    a = math.radians(angle_deg)
    cos, sin = math.cos(a), math.sin(a)
    # affine_grid works in normalised coordinates; correct for non-square tiles
    theta = torch.tensor([[cos, -sin * h / w, 0.0], [sin * w / h, cos, 0.0]], dtype=torch.float32)
    grid = F.affine_grid(theta[None], [1, 1, h, w], align_corners=False)
    img = F.grid_sample((image - image_fill)[None].float(), grid, mode="nearest", padding_mode="zeros",
                        align_corners=False)[0] + image_fill
    # shift labels by +1 so that the zero padding marks pixels outside the tile
    lab = F.grid_sample((label.float() + 1)[None, None], grid, mode="nearest", padding_mode="zeros",
                        align_corners=False)[0, 0]
    lab = torch.where(lab == 0, torch.full_like(lab, float(label_fill) + 1), lab) - 1
    return img.to(image.dtype), lab.to(label.dtype)


def scale_jitter(image: torch.Tensor, label: torch.Tensor, factor: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """Resample to ``factor`` x size and back to the original size (nearest neighbour both ways)."""
    _, h, w = image.shape
    size = (max(1, int(h * factor)), max(1, int(w * factor)))
    img = F.interpolate(F.interpolate(image[None].float(), size=size, mode="nearest"), size=(h, w), mode="nearest")[0]
    lab = F.interpolate(F.interpolate(label[None, None].float(), size=size, mode="nearest"), size=(h, w),
                        mode="nearest")[0, 0]
    return img.to(image.dtype), lab.to(label.dtype)


class Augment:
    """Random flips, rotation and scale jitter applied jointly to an image and its label.

    Call as ``image, label = Augment()(image, label)`` with ``image`` a
    ``(C, H, W)`` float tensor and ``label`` an ``(H, W)`` integer tensor.
    Randomness comes from ``torch`` so DataLoader workers are seeded correctly.
    """

    def __init__(self, hflip_p: float = 0.5, vflip_p: float = 0.5, rotate_p: float = 0.2,
                 max_rotation: float = 45.0, scale_p: float = 0.2, scale_range: Tuple[float, float] = (0.9, 1.11),
                 label_fill: int = IGNORE_INDEX, image_fill: float = 0.0):
        self.hflip_p, self.vflip_p = hflip_p, vflip_p
        self.rotate_p, self.max_rotation = rotate_p, max_rotation
        self.scale_p, self.scale_range = scale_p, tuple(scale_range)
        self.label_fill, self.image_fill = label_fill, image_fill

    def __call__(self, image: torch.Tensor, label: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if _rand() < self.hflip_p:
            image, label = image.flip(-1), label.flip(-1)
        if _rand() < self.vflip_p:
            image, label = image.flip(-2), label.flip(-2)
        if _rand() < self.rotate_p:
            angle = (2 * _rand() - 1) * self.max_rotation
            image, label = rotate(image, label, angle, self.image_fill, self.label_fill)
        if _rand() < self.scale_p:
            lo, hi = self.scale_range
            image, label = scale_jitter(image, label, lo + (hi - lo) * _rand())
        return image.contiguous(), label.contiguous()

    def __repr__(self) -> str:
        return (f"Augment(hflip_p={self.hflip_p}, vflip_p={self.vflip_p}, rotate_p={self.rotate_p}, "
                f"max_rotation={self.max_rotation}, scale_p={self.scale_p}, scale_range={self.scale_range}, "
                f"label_fill={self.label_fill})")
