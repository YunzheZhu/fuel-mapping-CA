# Copyright (C) 2023-2026 Yunzhe Zhu
# SPDX-License-Identifier: GPL-3.0-only
"""Loss function: class-weighted cross-entropy with an ignore label."""
from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn

from . import IGNORE_INDEX

__all__ = ["CLASS_WEIGHTS", "weighted_cross_entropy"]

CLASS_WEIGHTS = (0.4, 1.0, 1.1, 1.5)
"""Paper class weights for (other, tree, shrub, building)."""


def weighted_cross_entropy(weights: Sequence[float] = CLASS_WEIGHTS, n_outputs: int = 4,
                           ignore_index: int = IGNORE_INDEX,
                           device: Optional[torch.device] = None) -> nn.CrossEntropyLoss:
    """Weighted cross-entropy over the model logits.

    Args:
        weights: one weight per real class (other, tree, shrub, building).
        n_outputs: number of model output channels. With the legacy 5-channel
            head the extra "unlabeled" channel gets weight 0; it never receives a
            positive target because ignored pixels carry ``ignore_index``. This
            matches the original ``CrossEntropyLoss(weight=[0.4, 1, 1.1, 1.5, 0],
            ignore_index=4)`` setup.
        ignore_index: label value excluded from the loss.
        device: device for the weight tensor.
    """
    w = list(weights)
    if n_outputs < len(w):
        raise ValueError(f"{len(w)} class weights given for a model with {n_outputs} outputs")
    w += [0.0] * (n_outputs - len(w))
    return nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32, device=device), ignore_index=ignore_index)
