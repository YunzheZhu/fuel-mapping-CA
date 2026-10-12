# Copyright (C) 2023-2026 Yunzhe Zhu
# SPDX-License-Identifier: GPL-3.0-only
"""Prediction: class decision rule, tile inference and sliding-window inference.

Class decision rules (applied to softmax probabilities of the four classes;
with the legacy 5-output model the softmax is taken over all five outputs and
the fifth, "unlabeled" channel is then dropped):

``paper`` (used for the paper's test-tile accuracy assessment; the paper-era
large-area maps used the same thresholds without the NDVI gate, and the v2
Box maps come from a different model and rule)
    start with "other"; tree if p_tree > 0.4 and NDVI > 0.11; shrub if
    p_shrub > 0.4 and NDVI > 0.11 (overrides tree); building if
    p_building > 0.6; finally "other" if p_other > 0.5 (overrides everything).
    NDVI = (NIR - R) / (NIR + R) computed on DN/255 (0 where NIR + R = 0).
``argmax``
    plain argmax over the four class probabilities, no NDVI gate.

Large rasters are processed with the sliding window of the original
prediction pipeline: 1666 x 1666 px windows (about 1 km at 0.6 m), 25 %
overlap, windows starting at multiples of the step from the top-left corner,
edge windows zero-padded. Probabilities are blended with a linear-ramp weight
matrix: the first window that covers a pixel contributes with weight 1, later
windows contribute with the ramp weight (0 at the window edge, 1 at a distance
of ``overlap x tile`` from it). ``blend="mean"`` averages instead.
"""
from __future__ import annotations

from typing import Callable, Dict, Iterable, Optional, Tuple

import numpy as np
import torch

from . import NODATA, NUM_CLASSES

__all__ = ["THRESHOLDS", "NDVI_MIN", "TILE_SIZE", "OVERLAP", "ndvi", "nodata_mask", "classify", "predict_probs",
           "predict_batch", "weight_matrix", "predict_large", "predict_array"]

THRESHOLDS: Dict[str, float] = {"tree": 0.4, "shrub": 0.4, "building": 0.6, "other": 0.5}
"""Probability thresholds of the ``paper`` rule."""
NDVI_MIN = 0.11
"""Tree and shrub require NDVI > NDVI_MIN under the ``paper`` rule."""
TILE_SIZE = 1666
OVERLAP = 0.25


def ndvi(image):
    """NDVI of a ``(..., C, H, W)`` image in band order R, G, B, NIR (torch or numpy); 0 where NIR + R = 0."""
    img = torch.as_tensor(image)
    red, nir = img[..., 0, :, :].float(), img[..., 3, :, :].float()
    den = nir + red
    return torch.where(den != 0, (nir - red) / torch.where(den == 0, torch.ones_like(den), den),
                       torch.zeros_like(den))


def nodata_mask(image):
    """True where all bands of a ``(..., C, H, W)`` image are 0 (outside the NAIP footprint)."""
    return (torch.as_tensor(image) == 0).all(dim=-3)


def classify(probs, image=None, rule: str = "paper", thresholds: Optional[Dict[str, float]] = None,
             ndvi_min: Optional[float] = NDVI_MIN):
    """Turn class probabilities ``(..., 4, H, W)`` into a uint8 class map ``(..., H, W)``.

    Args:
        probs: probabilities of (other, tree, shrub, building); torch or numpy.
        image: the scaled input image ``(..., C, H, W)``; needed for the NDVI
            gate of the ``paper`` rule unless ``ndvi_min`` is ``None``.
        rule: ``"paper"`` or ``"argmax"`` (see module docstring).
        thresholds: overrides for :data:`THRESHOLDS` (keys other/tree/shrub/building).
        ndvi_min: NDVI gate for tree and shrub; ``None`` disables it.

    Returns the same array type as ``probs``.
    """
    as_numpy = isinstance(probs, np.ndarray)
    p = torch.as_tensor(probs)
    if rule == "argmax":
        out = p[..., :NUM_CLASSES, :, :].argmax(dim=-3).to(torch.uint8)
    elif rule == "paper":
        t = {**THRESHOLDS, **(thresholds or {})}
        out = torch.zeros(p.shape[:-3] + p.shape[-2:], dtype=torch.uint8, device=p.device)
        if ndvi_min is None:
            veg = torch.ones_like(out, dtype=torch.bool)
        elif image is None:
            raise ValueError("the paper rule needs the input image for the NDVI gate (or ndvi_min=None)")
        else:
            veg = ndvi(torch.as_tensor(image).to(p.device)) > ndvi_min
        out[(p[..., 1, :, :] > t["tree"]) & veg] = 1
        out[(p[..., 2, :, :] > t["shrub"]) & veg] = 2
        out[p[..., 3, :, :] > t["building"]] = 3
        out[p[..., 0, :, :] > t["other"]] = 0
    else:
        raise ValueError(f"unknown rule {rule!r}; use 'paper' or 'argmax'")
    return out.cpu().numpy() if as_numpy else out


@torch.no_grad()
def predict_probs(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Softmax over all model outputs, keeping the four class channels. ``x``: ``(N, C, H, W)`` scaled 0-1."""
    device = next(model.parameters()).device
    return torch.softmax(model(x.to(device)), dim=1)[:, :NUM_CLASSES]


def predict_batch(model: torch.nn.Module, images: torch.Tensor, rule: str = "paper",
                  thresholds: Optional[Dict[str, float]] = None, ndvi_min: Optional[float] = NDVI_MIN,
                  mask_nodata: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Predict whole tiles (e.g. the 512 x 512 px training/test tiles) in one forward pass.

    Args:
        images: float tensor ``(N, 4, H, W)`` scaled by 1/255.
        mask_nodata: set pixels whose bands are all 0 to 255.

    Returns ``(classes uint8 (N, H, W), probs float32 (N, 4, H, W))`` as numpy arrays.
    """
    probs = predict_probs(model, images).cpu()
    classes = classify(probs, images, rule, thresholds, ndvi_min)
    if mask_nodata:
        classes[nodata_mask(images)] = NODATA
    return classes.numpy(), probs.numpy()


def weight_matrix(tile: int, overlap: float) -> np.ndarray:
    """Blending weights: 1 in the centre, falling linearly to 0 over ``int(tile * overlap)`` px at each edge."""
    w = np.ones((tile, tile), np.float32)
    edge = int(tile * overlap)
    if edge > 0:
        ramp = np.linspace(0, 1, edge, dtype=np.float32)
        w[:edge, :] *= ramp[:, None]
        w[-edge:, :] *= ramp[::-1, None]
        w[:, :edge] *= ramp[None, :]
        w[:, -edge:] *= ramp[None, ::-1]
    return w


def _batches(items: list, size: int) -> Iterable[list]:
    for i in range(0, len(items), size):
        yield items[i:i + size]


def predict_large(model: torch.nn.Module, read_rows: Callable[[int, int], np.ndarray], height: int, width: int,
                  write_rows: Callable[[int, np.ndarray, Optional[np.ndarray]], None], tile: int = TILE_SIZE,
                  overlap: float = OVERLAP, batch_size: int = 1, blend: str = "ramp", rule: str = "paper",
                  thresholds: Optional[Dict[str, float]] = None, ndvi_min: Optional[float] = NDVI_MIN,
                  mask_nodata: bool = True, with_probs: bool = False, n_bands: int = 4, progress: bool = False):
    """Sliding-window prediction of an image of any size with bounded memory.

    The image is processed one row of windows at a time; memory scales with
    ``tile x width`` rather than with the image size: with the default 1666 px
    windows about 45 KB per raster column (65 KB with ``with_probs``), plus the
    model itself. Windows are built one batch at a time and the emitted rows
    are classified in column chunks. Rows are emitted from top to bottom as
    soon as no later window can touch them.

    Args:
        read_rows: ``read_rows(y0, y1)`` returns the image rows ``[y0, y1)`` as an
            array ``(bands, y1 - y0, width)`` of 0-255 digital numbers (R, G, B, NIR first).
        height, width: image size in pixels.
        write_rows: ``write_rows(y0, classes, probs)`` receives final rows: a uint8
            class array ``(rows, width)`` and, if ``with_probs``, float32
            probabilities ``(4, rows, width)`` (else ``None``).
        tile, overlap: window size (px) and fractional overlap; step = int(tile * (1 - overlap)).
        batch_size: windows per forward pass.
        blend: ``"ramp"`` (original weighting) or ``"mean"``.
        rule, thresholds, ndvi_min: class decision rule, see :func:`classify`.
        mask_nodata: write 255 where all input bands are 0 (probabilities set to 0).
    """
    if blend not in ("ramp", "mean"):
        raise ValueError("blend must be 'ramp' or 'mean'")
    step = int(tile * (1 - overlap))
    if step < 1:
        raise ValueError("overlap too large for this tile size")
    device = next(model.parameters()).device
    ramp = weight_matrix(tile, overlap) if blend == "ramp" else None
    ys, xs = list(range(0, height, step)), list(range(0, width, step))
    acc = np.zeros((NUM_CLASSES, tile, width), np.float32)
    cnt = np.zeros((tile, width), np.float32)
    rows_iter = enumerate(ys)
    if progress:
        from tqdm import tqdm

        rows_iter = tqdm(rows_iter, total=len(ys), desc="window rows", unit="row")
    scale = np.float32(255.0)
    for i, y in rows_iter:
        rows = min(tile, height - y)
        strip = np.asarray(read_rows(y, y + rows))[:n_bands]
        for xb in _batches(xs, batch_size):  # zero-padded windows, scaled one batch at a time
            batch = np.zeros((len(xb), n_bands, tile, tile), np.float32)
            for j, x in enumerate(xb):
                batch[j, :, :rows, :min(tile, width - x)] = strip[:, :, x:x + tile]
            batch /= scale
            probs = predict_probs(model, torch.from_numpy(batch).to(device)).cpu().numpy()
            for x, p in zip(xb, probs):
                cols = min(tile, width - x)
                c = cnt[:rows, x:x + cols]
                w = np.ones_like(c) if ramp is None else np.where(c > 0, ramp[:rows, :cols], np.float32(1))
                acc[:, :rows, x:x + cols] += p[:, :rows, :cols] * w
                c += w
        last = i == len(ys) - 1
        n = rows if last else min(step, rows)
        classes = np.empty((n, width), np.uint8)
        out_probs = np.empty((NUM_CLASSES, n, width), np.float32) if with_probs else None
        for c0 in range(0, width, tile):  # column chunks keep the float copies small
            cs = slice(c0, min(c0 + tile, width))
            probs = acc[:, :n, cs] / np.maximum(cnt[:n, cs], np.float32(1e-12))
            cls = classify(probs, strip[:, :n, cs].astype(np.float32) / scale, rule, thresholds, ndvi_min)
            if mask_nodata:
                bad = nodata_mask(strip[:, :n, cs]).numpy()
                cls[bad] = NODATA
                probs[:, bad] = 0
            classes[:, cs] = cls
            if with_probs:
                out_probs[:, :, cs] = probs
        write_rows(y, classes, out_probs)
        if not last:  # shift the accumulators up by one step
            keep = tile - step
            acc[:, :keep] = acc[:, step:step + keep].copy()
            cnt[:keep] = cnt[step:step + keep].copy()
            acc[:, keep:] = 0
            cnt[keep:] = 0


def predict_array(model: torch.nn.Module, image: np.ndarray, with_probs: bool = False, **kwargs):
    """In-memory wrapper of :func:`predict_large` for an image array ``(bands, H, W)`` of 0-255 values.

    Returns the class map ``(H, W)`` uint8, or ``(classes, probs)`` if ``with_probs``.
    """
    _, h, w = image.shape
    classes = np.full((h, w), NODATA, np.uint8)
    probs = np.zeros((NUM_CLASSES, h, w), np.float32) if with_probs else None

    def write(y0, c, p):
        classes[y0:y0 + c.shape[0]] = c
        if p is not None:
            probs[:, y0:y0 + c.shape[0]] = p

    predict_large(model, lambda y0, y1: image[:, y0:y1], h, w, write, with_probs=with_probs, **kwargs)
    return (classes, probs) if with_probs else classes
