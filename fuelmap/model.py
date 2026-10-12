# Copyright (C) 2023-2026 Yunzhe Zhu
# Portions Copyright (C) Alexandre Milesi (milesial) and the Pytorch-UNet contributors
# SPDX-License-Identifier: GPL-3.0-only
#
# Adapted from milesial/Pytorch-UNet (https://github.com/milesial/Pytorch-UNet),
# files unet/unet_model.py and unet/unet_parts.py, licensed under GPL-3.0.
# Modified by Yunzhe Zhu, 2023-2026: both files merged into one module, the
# broken use_checkpointing() removed, docstrings added, and checkpoint loading
# added (read_state_dict, load_model).
"""U-Net used for sub-meter woody vegetation and structure mapping.

Architecture (identical to the released weights, 31,038,469 parameters with a
5-channel head): a 5-level encoder (64-128-256-512-1024 kernels, each level two
3x3 conv -> BatchNorm -> ReLU, 2x2 max-pooling between levels), a decoder with
2x2 transposed convolutions and skip connections, and a 1x1 output conv.
The network returns raw logits; apply softmax outside the model.
"""
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Mapping, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["DoubleConv", "Down", "Up", "OutConv", "UNet", "strip_module_prefix", "read_state_dict", "load_model"]


class DoubleConv(nn.Module):
    """(3x3 conv -> BatchNorm -> ReLU) x 2."""

    def __init__(self, in_channels: int, out_channels: int, mid_channels: Optional[int] = None):
        super().__init__()
        mid_channels = mid_channels or out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """2x2 max-pooling followed by DoubleConv."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.maxpool_conv = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_channels, out_channels))

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upsampling, concatenation with the skip connection, then DoubleConv.

    The upsampled map is zero-padded to the size of the skip connection, so
    inputs whose size is not a multiple of 16 (e.g. 1666 px windows) also work.
    """

    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = False):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        dy = x2.size(2) - x1.size(2)
        dx = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])
        return self.conv(torch.cat([x2, x1], dim=1))


class OutConv(nn.Module):
    """1x1 convolution to class logits."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """U-Net (Ronneberger et al., 2015) as implemented in milesial/Pytorch-UNet.

    Args:
        n_channels: number of input bands (4 = NAIP R, G, B, NIR scaled to 0-1).
        n_classes: number of output channels. New models use 4
            (0 other, 1 tree, 2 shrub, 3 building). The released v1 weights have
            5 outputs: channel 4 is a legacy "unlabeled" channel that never
            received a positive training target and is dropped at inference.
        bilinear: use bilinear upsampling instead of transposed convolutions
            (the released weights use transposed convolutions, ``False``).
    """

    def __init__(self, n_channels: int = 4, n_classes: int = 4, bilinear: bool = False):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        factor = 2 if bilinear else 1
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)


def strip_module_prefix(state_dict: Mapping[str, torch.Tensor]) -> "OrderedDict[str, torch.Tensor]":
    """Remove the ``module.`` prefix that ``nn.DataParallel`` adds to parameter names."""
    return OrderedDict((k[len("module."):] if k.startswith("module.") else k, v) for k, v in state_dict.items())


def read_state_dict(path: Union[str, Path]) -> "OrderedDict[str, torch.Tensor]":
    """Read U-Net weights saved as a plain ``state_dict``, with the ``module.`` prefix of
    ``nn.DataParallel``, or wrapped as ``{"state_dict": ...}``; read with ``weights_only=True``."""
    obj = torch.load(str(path), map_location="cpu", weights_only=True)
    if isinstance(obj, Mapping) and "state_dict" in obj and isinstance(obj["state_dict"], Mapping):
        obj = obj["state_dict"]
    return strip_module_prefix(obj)


def load_model(
    path: Union[str, Path],
    n_classes: Optional[int] = None,
    device: Union[str, torch.device] = "cpu",
    n_channels: Optional[int] = None,
) -> UNet:
    """Load U-Net weights and return the model in eval mode.

    Accepts every format of :func:`read_state_dict`. The number of output
    classes (4 or 5) and input bands are inferred from the weights when not
    given.

    Args:
        path: ``.pth`` file.
        n_classes: expected number of outputs; ``None`` infers it.
        device: torch device to place the model on.
        n_channels: expected number of input bands; ``None`` infers it.
    """
    state = read_state_dict(path)
    try:
        out_w = state["outc.conv.weight"]
        in_w = state["inc.double_conv.0.weight"]
    except KeyError as err:
        raise ValueError(f"{path} does not look like a fuelmap U-Net state_dict (missing {err})") from None
    found_classes, found_channels = int(out_w.shape[0]), int(in_w.shape[1])
    if n_classes is not None and n_classes != found_classes:
        raise ValueError(f"{path} has {found_classes} output channels, but n_classes={n_classes} was requested")
    if n_channels is not None and n_channels != found_channels:
        raise ValueError(f"{path} expects {found_channels} input bands, but n_channels={n_channels} was requested")
    model = UNet(n_channels=found_channels, n_classes=found_classes, bilinear=False)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()
