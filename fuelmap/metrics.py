# Copyright (C) 2023-2026 Yunzhe Zhu
# SPDX-License-Identifier: GPL-3.0-only
"""Pixel-based accuracy metrics from a confusion matrix.

Convention: **rows = reference (label), columns = prediction**. Hence

* producer's accuracy PA (recall)  = diag / row sums,
* user's accuracy UA (precision)   = diag / column sums,
* F1 = 2 UA PA / (UA + PA), IoU = TP / (TP + FP + FN),
* OA = trace / total; mIoU, mean F1, mean UA and mean PA are unweighted means
  over the classes (the four classes including "other", as in the paper).

A per-class value whose denominator is zero (class absent from both reference
and prediction) is NaN and is skipped by the class means; pass
``undefined="zero"`` to count it as 0 instead.
"""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np

from . import CLASS_NAMES, NUM_CLASSES

__all__ = ["confusion_matrix", "metrics_from_cm", "format_metrics"]


def confusion_matrix(reference, prediction, n_classes: int = NUM_CLASSES) -> np.ndarray:
    """Confusion matrix ``cm[ref, pred]`` (int64) of two label arrays of equal shape.

    Pixels whose reference or prediction lies outside ``[0, n_classes)``
    (ignore/nodata 255, legacy "no label" values, v2 "uncertain") are excluded.
    Accepts numpy arrays or torch tensors (computed on the tensor's device).
    """
    if hasattr(reference, "detach"):  # torch tensors
        import torch

        ref = reference.detach().reshape(-1).long()
        pred = prediction.detach().reshape(-1).long().to(ref.device)
        ok = (ref >= 0) & (ref < n_classes) & (pred >= 0) & (pred < n_classes)
        cm = torch.bincount(ref[ok] * n_classes + pred[ok], minlength=n_classes * n_classes)
        return cm.reshape(n_classes, n_classes).cpu().numpy().astype(np.int64)
    ref = np.asarray(reference).reshape(-1).astype(np.int64)
    pred = np.asarray(prediction).reshape(-1).astype(np.int64)
    if ref.shape != pred.shape:
        raise ValueError(f"reference and prediction sizes differ: {ref.shape} vs {pred.shape}")
    ok = (ref >= 0) & (ref < n_classes) & (pred >= 0) & (pred < n_classes)
    return np.bincount(ref[ok] * n_classes + pred[ok], minlength=n_classes * n_classes).reshape(n_classes, n_classes)


def _div(a, b, undefined: str):
    out = np.full(np.shape(a), np.nan if undefined == "nan" else 0.0)
    np.divide(a, b, out=out, where=np.asarray(b) != 0)
    return out


def metrics_from_cm(cm, class_names: Sequence[str] = CLASS_NAMES, undefined: str = "nan") -> Dict:
    """Accuracy metrics from a confusion matrix (rows = reference, columns = prediction).

    Returns a dict with ``OA``, ``mIoU``, ``mean_F1``, ``mean_UA``, ``mean_PA``,
    ``pixels`` and per-class dicts ``UA``, ``PA``, ``F1``, ``IoU`` keyed by
    class name, plus the matrix itself under ``confusion_matrix``.
    """
    if undefined not in ("nan", "zero"):
        raise ValueError("undefined must be 'nan' or 'zero'")
    cm = np.asarray(cm, dtype=np.float64)
    tp = np.diag(cm)
    ref_tot, pred_tot = cm.sum(1), cm.sum(0)
    pa = _div(tp, ref_tot, undefined)
    ua = _div(tp, pred_tot, undefined)
    f1 = _div(2 * tp, ref_tot + pred_tot, undefined)  # == 2 UA PA / (UA + PA)
    iou = _div(tp, ref_tot + pred_tot - tp, undefined)
    total = cm.sum()
    mean = lambda v: float(np.nanmean(v)) if np.isfinite(v).any() else float("nan")  # noqa: E731
    per = lambda v: {n: float(x) for n, x in zip(class_names, v)}  # noqa: E731
    return {
        "OA": float(tp.sum() / total) if total else float("nan"),
        "mIoU": mean(iou),
        "mean_F1": mean(f1),
        "mean_UA": mean(ua),
        "mean_PA": mean(pa),
        "pixels": int(total),
        "UA": per(ua),
        "PA": per(pa),
        "F1": per(f1),
        "IoU": per(iou),
        "confusion_matrix": cm.astype(np.int64).tolist(),
    }


def format_metrics(m: Dict) -> str:
    """One-paragraph text summary of :func:`metrics_from_cm` output."""
    lines = [f"OA {m['OA']:.4f}  mIoU {m['mIoU']:.4f}  mean F1 {m['mean_F1']:.4f}  "
             f"mean UA {m['mean_UA']:.4f}  mean PA {m['mean_PA']:.4f}  pixels {m['pixels']:,}"]
    for name in m["F1"]:
        lines.append(f"  {name:<9s} UA {m['UA'][name]:.4f}  PA {m['PA'][name]:.4f}  "
                     f"F1 {m['F1'][name]:.4f}  IoU {m['IoU'][name]:.4f}")
    return "\n".join(lines)
