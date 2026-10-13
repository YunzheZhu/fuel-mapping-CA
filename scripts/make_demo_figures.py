#!/usr/bin/env python
# Copyright (C) 2023-2026 Yunzhe Zhu
# SPDX-License-Identifier: GPL-3.0-only
r"""Make the README demo figures: what the release model gives on the public test tiles and on a NAIP scene.

Run from the repository root. Needs matplotlib (>= 3.6, which brings Pillow) besides the package requirements.

Test-tile figures (release weights + test package; about 8 min on a 24-core CPU, or seconds when an
existing --pred-cache is reused):

  python scripts/make_demo_figures.py --weights models/fuelmap_unet_v1.pth \
      --test-npz release/fuelmap_test_1y.npz --index data/test_array_index.csv \
      --pred-cache outputs/demo_pred_cache.npz

  test_gallery.png      one tile per WUI class: NAIP | reference | prediction | agreement
  confusion_matrix.png  4 x 4 confusion matrix over all test tiles, row-normalised (recall)
  per_tile_oa.png       per-tile OA by WUI class (full-label tiles)
  demo_metrics.json     all metrics, including the per-tile OA summary quoted in the README
  demo_tiles.csv        the gallery tiles and the rule that chose them

  Gallery tiles are picked by a fixed rule, not by eye: within each WUI class (full-label
  tiles; class "Unknown" left out) the tile whose per-tile OA is closest to the class median,
  ties broken by tile_id. The last row applies the same rule to the building-only
  urban-sample tiles, whose per-tile OA is building recall (other pixels are unlabelled).
  Rows group tiles by wui_class; the tile_id prefix (URB, WUI, WLD) is the sampling stratum,
  which can differ from it.

NAIP scene figure (any 4-band NAIP GeoTIFF; same sliding window, blending and rule as
scripts/predict_naip.py --window):

  python scripts/make_demo_figures.py --weights models/fuelmap_unet_v1.pth \
      --naip m_3912128_se_10_060_20180718.tif --naip-window 4720 6920 2000 2000 --zoom 1100 500 500

  naip_scene_demo.png   NAIP | predicted classes, for the window and a zoom

All predictions use the paper rule (probability thresholds + NDVI gate). Colours follow the
paper's maps (Figs. 4 and 6): tree blue, shrub orange, building red, other black.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fuelmap import CLASS_COLORS, CLASS_NAMES, NUM_CLASSES  # noqa: E402
from fuelmap.inference import predict_array, predict_batch  # noqa: E402
from fuelmap.metrics import confusion_matrix, metrics_from_cm  # noqa: E402
from fuelmap.model import load_model  # noqa: E402
from fuelmap.utils import resolve_device  # noqa: E402

ORDER = (1, 2, 3, 0)  # display order of the classes, as in the paper: tree, shrub, building, other
LABEL = {0: "Other", 1: "Tree", 2: "Shrub", 3: "Building"}
WUI = {"Urban": "Urban", "Interface": "WUI interface", "Intermix": "WUI intermix", "Wildland": "Wildland"}
PAPER_TABLE1 = {"n_tiles": 782, "OA": 0.871, "mIoU": 0.701, "mean_F1": 0.821,
                "F1": {"tree": 0.831, "shrub": 0.789, "building": 0.751, "other": 0.914}}
# chart tokens (light surface); the error colour was checked with the dataviz palette validator
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#ffffff"
CORRECT, ERROR, NOLABEL = "#cfcec8", "#d55181", "#ffffff"
BLUES = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf",
         "#1c5cab", "#184f95", "#104281", "#0d366b"]


def hex_rgb(h):
    return tuple(int(h[i:i + 2], 16) for i in (1, 3, 5))


def class_rgb(a):
    """Class map (0-3, 255 = no label) -> RGB uint8 in the paper's colours; no label is white."""
    lut = np.full((256, 3), 255, np.uint8)
    for k, c in CLASS_COLORS.items():
        lut[k] = c
    return lut[a]


def agreement_rgb(ref, pred):
    out = np.full(ref.shape + (3,), 255, np.uint8)
    ok = ref < NUM_CLASSES
    out[ok & (ref == pred)] = hex_rgb(CORRECT)
    out[ok & (ref != pred)] = hex_rgb(ERROR)
    return out


def true_color(rgb, lo=0.5, hi=99.5):
    """(H, W, >=3) digital numbers -> display RGB with one linear stretch for all three bands."""
    x = rgb[..., :3].astype(np.float32)
    valid = x.any(-1)
    a, b = np.percentile(x[valid], [lo, hi]) if valid.any() else (0, 255)
    return np.clip((x - a) / max(b - a, 1), 0, 1)


def setup_mpl():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "Arial", "DejaVu Sans"],
                         "font.size": 10, "text.color": INK, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
                         "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2,
                         "ytick.labelcolor": INK2, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE})
    return plt


def save(fig, path, quantize=False):
    """Save as PNG; ``quantize`` maps photo-heavy figures to 256 colours (exact class colours kept)."""
    from PIL import Image
    import matplotlib.pyplot as plt

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=fig.dpi, facecolor=SURFACE)
    plt.close(fig)
    im = Image.open(buf).convert("RGB")
    if quantize:
        fixed = [tuple(c) for c in CLASS_COLORS.values()] + [hex_rgb(c) for c in (CORRECT, ERROR, INK, INK2, MUTED)]
        fixed += [(255, 255, 255)]
        adaptive = im.quantize(256 - len(fixed), method=Image.Quantize.MAXCOVERAGE).getpalette()
        pal = Image.new("P", (1, 1))
        pal.putpalette(adaptive[:3 * (256 - len(fixed))] + [v for c in fixed for v in c])
        im = im.convert("RGB").quantize(palette=pal, dither=Image.Dither.FLOYDSTEINBERG)
    im.save(path, optimize=True)
    print(f"wrote {path} ({im.size[0]} x {im.size[1]} px, {Path(path).stat().st_size / 1e6:.2f} MB)")


def legend_patches(items):
    from matplotlib.patches import Patch

    return [Patch(facecolor=c, edgecolor=AXIS, linewidth=0.6, label=t) for t, c in items]


def class_items(shares=None):
    return [(LABEL[k] + (f" {shares[k]:.0%}" if shares is not None else ""), np.array(CLASS_COLORS[k]) / 255)
            for k in ORDER]


# ----------------------------------------------------------------------------------------- test tiles
def predict_test(images, labels, model, batch_size=4):
    """Paper-rule class maps (N, H, W) and per-tile confusion matrices (N, 4, 4)."""
    n = len(images)
    pred = np.empty(labels.shape, np.uint8)
    cms = np.zeros((n, NUM_CLASSES, NUM_CLASSES), np.int64)
    t0 = time.time()
    for i in range(0, n, batch_size):
        x = torch.from_numpy(images[i:i + batch_size].transpose(0, 3, 1, 2).astype(np.float32) / np.float32(255))
        pred[i:i + batch_size] = predict_batch(model, x, rule="paper")[0]
        for k in range(i, min(i + batch_size, n)):
            cms[k] = confusion_matrix(labels[k], pred[k])
        print(f"  {min(i + batch_size, n)}/{n} tiles ({time.time() - t0:.0f} s)", flush=True)
    return pred, cms


def read_index(path):
    """test_array_index.csv -> dict of column arrays in package order (plain csv; no pandas needed)."""
    with open(path, newline="", encoding="utf-8") as f:
        recs = sorted(csv.DictReader(f), key=lambda r: int(r["index"]))
    idx = {k: np.array([r[k] for r in recs]) for k in ("tile_id", "label_scope", "stratum", "wui_class")}
    for k in ("index", "naip_year"):
        idx[k] = np.array([int(r[k]) for r in recs])
    return idx


def pick_typical(idx, sel):
    """Package row of the selected tile whose tile_oa is closest to the selection median (ties: smallest
    tile_id), and that median."""
    rows = np.flatnonzero(sel)
    oa = idx["tile_oa"][rows]
    med = float(np.median(oa))
    d = np.round(np.abs(oa - med), 12)
    best = min(range(len(rows)), key=lambda i: (d[i], idx["tile_id"][rows[i]]))
    return int(rows[best]), med


def fig_gallery(plt, rows, images, labels, pred, path):
    n, P, G, LB, M, TOP, HEAD, LEG = len(rows), 1.86, 0.07, 1.95, 0.14, 1.20, 0.34, 0.80
    W, H = M + LB + 4 * P + 3 * G + M, TOP + HEAD + n * P + (n - 1) * G + LEG + M
    fig = plt.figure(figsize=(W, H), dpi=200)
    fx = lambda x: x / W  # noqa: E731  (inches from the left / top -> figure fraction)
    fy = lambda y: 1 - y / H  # noqa: E731
    fig.text(fx(M), fy(0.16), "Release model on the public test tiles: one typical tile per WUI class",
             fontsize=13.5, weight="semibold", va="top")
    fig.text(fx(M), fy(0.50), "Each row is the tile whose per-tile overall accuracy (OA) is closest to the median of "
             "its class (ties: tile_id);\ntiles are not hand-picked. Rows group tiles by WUI class; the tile_id "
             "prefix (URB, WUI, WLD) is the sampling\nstratum, which can differ. Tiles are 512 × 512 px (307 m) at "
             "the LiDAR-matched NAIP year; NAIP display-stretched.", fontsize=9.5, color=INK2, va="top",
             linespacing=1.4)
    for j, h in enumerate(["NAIP true colour", "Reference label", "Model prediction", "Agreement"]):
        fig.text(fx(M + LB + j * (P + G) + P / 2), fy(TOP + HEAD * 0.5), h, ha="center", va="center",
                 fontsize=10.5, weight="semibold")
    for i, r in enumerate(rows):
        y, k = TOP + HEAD + i * (P + G), int(r["index"])
        panels = [true_color(images[k]), class_rgb(labels[k]), class_rgb(pred[k]),
                  agreement_rgb(labels[k], pred[k])]
        for j, img in enumerate(panels):
            ax = fig.add_axes([fx(M + LB + j * (P + G)), fy(y + P), P / W, P / H])
            ax.imshow(img, interpolation="antialiased" if j == 0 else "nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_color(AXIS)
                s.set_linewidth(0.6)
        title = r["title"].split("\n")
        for t, line in enumerate(title):
            fig.text(fx(M), fy(y + 0.04 + 0.21 * t), line, fontsize=10.5, weight="semibold", va="top")
        y0 = y + 0.10 + 0.21 * len(title)
        details = [(r["tile_id"], INK2), (f"NAIP {r['naip_year']}", INK2),
                   (f"{r['metric']} {r['tile_oa']:.1%}", INK), (f"median {r['median']:.1%} (n = {r['n']})", INK2)]
        for t, (line, c) in enumerate(details):
            fig.text(fx(M), fy(y0 + 0.2 * t), line, fontsize=9.5, color=c, va="top")
    ly = fy(H - M - LEG + 0.12)
    lg = fig.legend(handles=legend_patches(class_items() + [("No label (ignored)", NOLABEL)]), ncol=5,
                    loc="upper left", bbox_to_anchor=(fx(M + LB - 0.05), ly), frameon=False, fontsize=9.5,
                    title="Classes (reference and prediction)", title_fontsize=9.5, alignment="left",
                    handlelength=1.3, handleheight=1.3, columnspacing=1.2)
    fig.add_artist(lg)
    fig.legend(handles=legend_patches([("Correct", CORRECT), ("Error", ERROR)]), ncol=1, loc="upper left",
               bbox_to_anchor=(fx(M + LB + 3 * (P + G) + 0.25), ly), frameon=False, fontsize=9.5,
               title="Agreement", title_fontsize=9.5, alignment="left", handlelength=1.3, handleheight=1.3)
    save(fig, path, quantize=True)


def fig_confusion(plt, cm, m, n_tiles, n_bo, path):
    from matplotlib.colors import LinearSegmentedColormap, to_rgb
    from matplotlib.colorbar import ColorbarBase
    from matplotlib.patches import Rectangle

    c = cm[np.ix_(ORDER, ORDER)].astype(float)
    pct = 100 * c / c.sum(1, keepdims=True)
    cmap = LinearSegmentedColormap.from_list("blues", BLUES)
    W, H = 7.2, 6.7
    fig = plt.figure(figsize=(W, H), dpi=250)
    fig.text(0.04, 1 - 0.1625 / H, "Confusion matrix of the release model on the public test set", fontsize=13,
             weight="semibold", va="top")
    fig.text(0.04, 1 - 0.455 / H, f"All {n_tiles} test tiles, paper rule. Rows = reference, columns = prediction.\n"
             "Each cell: % of the reference row (the diagonal is recall) and the pixel count.\n"
             f"The {n_bo} building-only tiles add pixels to the Building row only.", fontsize=9, color=INK2,
             va="top", linespacing=1.4)
    ax = fig.add_axes([0.2, 0.975 / H, 0.63, 4.095 / H])  # 4.1 in tall, 0.98 in above the bottom
    for i in range(4):
        for j in range(4):
            col = cmap(pct[i, j] / 100)
            ax.add_patch(Rectangle((j, i), 1, 1, facecolor=col, edgecolor=SURFACE, linewidth=2.5))
            lum = np.dot(to_rgb(col), [0.2126, 0.7152, 0.0722])
            ink, ink2 = (SURFACE, SURFACE) if lum < 0.45 else (INK, INK2)
            ax.text(j + 0.5, i + 0.44, f"{pct[i, j]:.1f}%", ha="center", va="center", fontsize=13, color=ink,
                    weight="semibold")
            v = c[i, j]
            px = f"{v / 1e6:.1f} M px" if v >= 1e6 else f"{v / 1e3:.0f} k px" if v >= 1e3 else f"{v:.0f} px"
            ax.text(j + 0.5, i + 0.72, px, ha="center", va="center", fontsize=8.5, color=ink2)
    names = [LABEL[k] for k in ORDER]
    ax.set_xlim(0, 4)
    ax.set_ylim(4, 0)
    ax.set_xticks(np.arange(4) + 0.5, names, fontsize=10.5)
    ax.set_yticks(np.arange(4) + 0.5, names, fontsize=10.5)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    ax.set_xlabel("Model prediction", fontsize=10.5, labelpad=8)
    ax.set_ylabel("Reference label", fontsize=10.5, labelpad=8)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ua = [m["UA"][CLASS_NAMES[k]] for k in ORDER]
    ax.text(-0.1, 4.28, "Precision (UA)", ha="right", va="center", fontsize=9.5, color=INK2)
    for j, u in enumerate(ua):
        ax.text(j + 0.5, 4.28, f"{u:.1%}", ha="center", va="center", fontsize=10, color=INK)
    ax.text(0, 4.62, f"Overall accuracy {m['OA']:.1%}  ·  mean F1 {m['mean_F1']:.1%}  ·  "
            f"{m['pixels'] / 1e6:.1f} M labelled pixels", ha="left", va="center", fontsize=10.5, weight="semibold")
    cax = fig.add_axes([0.86, 0.975 / H, 0.022, 4.095 / H])
    cb = ColorbarBase(cax, cmap=cmap, orientation="vertical", ticks=[0, 0.25, 0.5, 0.75, 1])
    cb.set_ticklabels(["0%", "25%", "50%", "75%", "100%"])
    cb.outline.set_visible(False)
    cax.tick_params(length=0, labelsize=8.5)
    cax.set_ylabel("share of the reference row", fontsize=9, color=INK2)
    save(fig, path)


def fig_per_tile(plt, wui, oa, picks, n_out, path):
    """Per-tile OA (``oa``, 0-1) grouped by WUI class (``wui``), with the gallery tiles ringed."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle

    groups = [g for g in WUI if g in set(wui)]
    fig = plt.figure(figsize=(7.2, 4.8), dpi=250)
    fig.text(0.04, 0.96, "Per-tile overall accuracy by WUI class", fontsize=13, weight="semibold", va="top")
    fig.text(0.04, 0.895, f"{len(oa)} full-label test tiles ({n_out} of unknown WUI class left out), release "
             "model, paper rule.\nBox = middle 50% of tiles, line = median; each dot is one tile.", fontsize=9,
             color=INK2, va="top", linespacing=1.4)
    ax = fig.add_axes([0.2, 0.13, 0.62, 0.6])
    rng = np.random.default_rng(0)
    lo = max(0, np.floor(oa.min() * 10) / 10)
    for i, g in enumerate(groups):
        v = oa[wui == g] * 100
        q1, med, q3 = np.percentile(v, [25, 50, 75])
        ax.add_patch(Rectangle((q1, i - 0.28), q3 - q1, 0.56, facecolor="#f0efec", edgecolor=AXIS, linewidth=0.8,
                               zorder=1))
        ax.plot([med, med], [i - 0.28, i + 0.28], color=INK, linewidth=2, solid_capstyle="butt", zorder=3)
        ax.scatter(v, i + rng.uniform(-0.2, 0.2, len(v)), s=9, color=INK2, alpha=0.45, linewidths=0, zorder=2)
        ax.text(102.5, i, f"median {med:.1f}%", va="center", ha="left", fontsize=9.5, color=INK, clip_on=False)
        if g in picks:
            ax.scatter([picks[g] * 100], [i], s=90, facecolors="none", edgecolors=INK, linewidths=1.6, zorder=4)
    ax.set_yticks(range(len(groups)), [f"{WUI[g]}\nn = {(wui == g).sum()}" for g in groups], fontsize=10)
    ax.set_ylim(len(groups) - 0.45, -0.55)
    ax.set_xlim(lo * 100, 101)
    ticks = np.arange(np.ceil(lo * 10) * 10, 101, 10)
    ax.set_xticks(ticks, [f"{t:.0f}%" for t in ticks], fontsize=9)
    ax.tick_params(length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xlabel("Per-tile overall accuracy (labelled pixels of the tile)", fontsize=9.5)
    fig.legend(handles=[Line2D([], [], marker="o", linestyle="none", markerfacecolor="none", markeredgecolor=INK,
                               markersize=8, markeredgewidth=1.6, label="tile shown in the gallery")],
               loc="upper left", bbox_to_anchor=(0.03, 0.83), frameon=False, fontsize=9.5)
    save(fig, path)


def run_tiles(args, plt, out):
    print(f"loading {args.test_npz} ...")
    with np.load(args.test_npz) as z:
        images, labels = z["images"], z["labels"]
    cache = Path(args.pred_cache) if args.pred_cache else None
    weights = Path(args.weights).name if args.weights else None
    if cache and cache.exists():
        with np.load(cache) as z:
            pred, cms = z["pred"], z["cms"]
            weights = str(z["weights"]) if "weights" in z.files else weights
        if pred.shape != labels.shape:
            sys.exit(f"{cache} holds {pred.shape} predictions, the test package {labels.shape} labels")
        print(f"reusing predictions from {cache}")
    else:
        if not args.weights:
            sys.exit("--weights is required (or an existing --pred-cache)")
        model = load_model(args.weights, device=resolve_device(args.device))
        pred, cms = predict_test(images, labels, model, args.batch_size)
        if cache:
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cache, pred=pred, cms=cms, weights=weights, rule="paper")
    idx = read_index(args.index)
    if len(idx["index"]) != len(labels) or (idx["index"] != np.arange(len(labels))).any():
        sys.exit(f"{args.index} does not list rows 0-{len(labels) - 1} of the test package once each")
    idx["tile_oa"] = np.trace(cms, axis1=1, axis2=2) / np.maximum(cms.sum((1, 2)), 1)
    full, bo = idx["label_scope"] == "full", idx["label_scope"] == "building_only"

    def gallery_row(group, title, metric, sel):
        k, med = pick_typical(idx, sel)
        return dict(group=group, title=title, metric=metric, median=med, n=int(sel.sum()), index=k,
                    tile_id=str(idx["tile_id"][k]), naip_year=int(idx["naip_year"][k]),
                    label_scope=str(idx["label_scope"][k]), tile_oa=float(idx["tile_oa"][k]))

    rows, picks = [], {}
    for g, name in WUI.items():
        sel = full & (idx["wui_class"] == g)
        if sel.any():
            rows.append(gallery_row(g, name, "tile OA", sel))
            picks[g] = rows[-1]["tile_oa"]
    if bo.any():
        rows.append(gallery_row("building_only", "Urban sample\n(building-only labels)", "building recall", bo))
    rule = ("per group, the tile whose per-tile OA (labelled pixels) is closest to the group median; ties: smallest "
            "tile_id. Groups: wui_class among label_scope=full (Unknown left out), then label_scope=building_only "
            "(per-tile OA = building recall).")
    with open(out / "demo_tiles.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["gallery_row", "group", "label_scope", "index", "tile_id", "naip_year", "tile_oa",
                    "group_median_oa", "group_n_tiles", "rule"])
        for i, r in enumerate(rows, 1):
            w.writerow([i, r["group"], r["label_scope"], r["index"], r["tile_id"], r["naip_year"],
                        f"{r['tile_oa']:.4f}", f"{r['median']:.4f}", r["n"], rule])

    cm_all = cms.sum(0)
    m_all = metrics_from_cm(cm_all)
    strata = {}
    for g in WUI:
        sel = full & (idx["wui_class"] == g)
        q = np.percentile(idx["tile_oa"][sel], [25, 50, 75])
        strata[g] = dict(n_tiles=int(sel.sum()), per_tile_oa_quartiles=[round(float(v), 4) for v in q],
                         pooled=metrics_from_cm(cms[sel].sum(0)))
    known = full & np.isin(idx["wui_class"], list(WUI))
    oa_known = idx["tile_oa"][known]
    lowest = int(np.flatnonzero(known)[np.argmin(oa_known)])
    per_tile = dict(  # the per-tile numbers quoted in the README (per_tile_oa.png)
        tiles="label_scope=full with a known wui_class", n_tiles=int(known.sum()),
        n_unknown_wui_left_out=int((full & ~known).sum()),
        n_tiles_oa_at_least_80pct=int((oa_known >= 0.8).sum()), n_tiles_oa_below_60pct=int((oa_known < 0.6).sum()),
        quartiles=[round(float(v), 4) for v in np.percentile(oa_known, [25, 50, 75])],
        min_oa=round(float(oa_known.min()), 4), min_oa_tile_id=str(idx["tile_id"][lowest]),
        max_oa=round(float(oa_known.max()), 4))
    metrics = dict(
        weights=weights, test_npz=Path(args.test_npz).name,
        rule="paper (probability thresholds + NDVI gate)",
        convention="confusion_matrix[reference][prediction], classes other, tree, shrub, building",
        all_tiles=dict(n_tiles=len(labels), **m_all),
        full_label_tiles=dict(n_tiles=int(full.sum()), **metrics_from_cm(cms[full].sum(0))),
        building_only_tiles=dict(n_tiles=int(bo.sum()), building_recall=float(
            cms[bo, 3, 3].sum() / max(cms[bo, 3].sum(), 1))),
        by_wui_class_full_label=strata, per_tile_oa_full_label=per_tile, paper_table1=PAPER_TABLE1,
        gallery=[{k: r[k] for k in ("group", "median", "n", "index", "tile_id", "naip_year", "tile_oa")}
                 for r in rows],
        gallery_rule=rule)
    update_json(out / "demo_metrics.json", metrics)
    print(f"OA {m_all['OA']:.4f}  mIoU {m_all['mIoU']:.4f}  mean F1 {m_all['mean_F1']:.4f}  F1 "
          + " ".join(f"{k} {v:.4f}" for k, v in m_all["F1"].items()))
    print(f"per-tile OA, {per_tile['n_tiles']} full-label tiles: {per_tile['n_tiles_oa_at_least_80pct']} at 80 % or "
          f"more, {per_tile['n_tiles_oa_below_60pct']} below 60 %, lowest {per_tile['min_oa']:.4f} "
          f"({per_tile['min_oa_tile_id']})")
    for r in rows:
        print(f"  gallery: {r['group']:<14s} {r['tile_id']:<16s} NAIP {r['naip_year']}  {r['metric']} "
              f"{r['tile_oa']:.4f} (median {r['median']:.4f}, n = {r['n']})")

    fig_gallery(plt, rows, images, labels, pred, out / "test_gallery.png")
    fig_confusion(plt, cm_all, m_all, len(labels), int(bo.sum()), out / "confusion_matrix.png")
    fig_per_tile(plt, idx["wui_class"][known], idx["tile_oa"][known], picks, int((full & ~known).sum()),
                 out / "per_tile_oa.png")


# ----------------------------------------------------------------------------------------- NAIP scene
def add_scalebar(ax, metres, px_m, n_px):
    import matplotlib.patheffects as pe
    from matplotlib.patches import Rectangle

    L, x0, y0 = metres / px_m, 0.05 * n_px, 0.93 * n_px
    ax.add_patch(Rectangle((x0, y0), L, 0.012 * n_px, facecolor="white", edgecolor=INK, linewidth=0.8))
    ax.text(x0 + L / 2, y0 - 0.015 * n_px, f"{metres:g} m", ha="center", va="bottom", fontsize=9.5, color="white",
            weight="semibold", path_effects=[pe.withStroke(linewidth=2.5, foreground=INK)])


def run_naip(args, plt, out):
    import matplotlib.patheffects as pe
    import rasterio
    from matplotlib.patches import Rectangle
    from rasterio.warp import transform as warp_xy
    from rasterio.windows import Window

    with rasterio.open(args.naip) as src:
        centre = (max(0, (src.width - 2000) // 2), max(0, (src.height - 2000) // 2), 2000, 2000)
        col, row, w, h = args.naip_window or centre
        win = Window(col, row, w, h).intersection(Window(0, 0, src.width, src.height))
        col, row, w, h = int(win.col_off), int(win.row_off), int(win.width), int(win.height)
        arr = src.read([1, 2, 3, 4], window=win)
        if src.nodata is not None and src.nodata != 0:  # as predict_naip.py
            arr[:, (arr == src.nodata).all(axis=0)] = 0
        tr, crs, px_m = src.window_transform(win), src.crs, src.res[0]
    model = load_model(args.weights, device=resolve_device(args.device))
    t0 = time.time()
    classes = predict_array(model, arr)  # predict_naip.py defaults: 1666 px windows, 25 % overlap, ramp, paper rule
    print(f"predicted {w} x {h} px in {time.time() - t0:.0f} s")
    cx, cy = tr * (w / 2, h / 2)
    lon, lat = (v[0] for v in warp_xy(crs, "EPSG:4326", [cx], [cy]))
    scene = Path(args.naip).stem
    date = re.search(r"_(\d{4})(\d{2})(\d{2})$", scene)
    when = f"{date[1]}-{date[2]}-{date[3]}" if date else "date unknown"
    valid = classes != 255
    shares = {k: float((classes[valid] == k).mean()) for k in range(NUM_CLASSES)}
    zc, zr, zs = args.zoom or (w // 2 - 250, h // 2 - 250, 500)
    rgb = true_color(arr.transpose(1, 2, 0))
    cls = class_rgb(classes)

    P, G, M, TOP, LEG, HD = 4.4, 0.1, 0.14, 0.95, 0.72, 0.3  # panel, gap, margin, title, legend, panel header
    W, H = 2 * P + G + 2 * M, TOP + 2 * (HD + P) + G + LEG + M
    nice = (10, 20, 50, 100, 200, 500, 1000, 2000, 5000)
    bar = lambda n: max(b for b in nice if b <= max(n * px_m / 5, 10))  # noqa: E731  (scale bar length, m)
    fig = plt.figure(figsize=(W, H), dpi=200)
    fig.text(M / W, 1 - 0.16 / H, f"Release model on a NAIP scene: a {w * px_m / 1000:.1f} km window and a zoom",
             fontsize=13.5, weight="semibold", va="top")
    fig.text(M / W, 1 - 0.5 / H, f"{scene} (NAIP {when}), pixels col {col}, row {row}, {w} × {h} px "
             f"({w * px_m / 1000:.2f} × {h * px_m / 1000:.2f} km), centre {lat:.4f}° N, {abs(lon):.4f}° "
             f"{'W' if lon < 0 else 'E'}.\nSame sliding window and paper rule as scripts/predict_naip.py; NAIP shown "
             "with a display stretch.", fontsize=9.5, color=INK2, va="top", linespacing=1.4)
    for i, (sl, label) in enumerate([((slice(None), slice(None)), "Full window"),
                                     ((slice(zr, zr + zs), slice(zc, zc + zs)), "Zoom (white box)")]):
        y = TOP + i * (P + G + HD)
        for j, (img, name) in enumerate([(rgb, "NAIP true colour"), (cls, "Predicted classes")]):
            ax = fig.add_axes([(M + j * (P + G)) / W, 1 - (y + HD + P) / H, P / W, P / H])
            ax.imshow(img[sl], interpolation="antialiased" if j == 0 else "nearest")  # never blend classes
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_color(AXIS)
                s.set_linewidth(0.6)
            n_px = img[sl].shape[1]
            add_scalebar(ax, bar(n_px), px_m, n_px)
            if i == 0:
                ax.add_patch(Rectangle((zc, zr), zs, zs, fill=False, edgecolor="white", linewidth=1.6,
                                       path_effects=[pe.withStroke(linewidth=3, foreground=INK)]))
            fig.text((M + j * (P + G)) / W, 1 - (y + 0.22) / H, f"{name} · {label.lower()}", fontsize=10.5,
                     weight="semibold", va="bottom")
    fig.legend(handles=legend_patches(class_items(shares)), ncol=4, loc="upper left",
               bbox_to_anchor=(M / W, 1 - (H - M - LEG + 0.12) / H), frameon=False, fontsize=10,
               title="Predicted classes (share of the window)", title_fontsize=10, alignment="left",
               handlelength=1.3, handleheight=1.3, columnspacing=1.5)
    save(fig, out / "naip_scene_demo.png", quantize=True)
    update_json(out / "demo_metrics.json", {"naip_scene": dict(
        scene=scene, date=when, window_col_row_width_height=[col, row, w, h], zoom_col_row_size=[zc, zr, zs],
        centre_lat_lon=[round(lat, 5), round(lon, 5)], class_shares={CLASS_NAMES[k]: round(v, 4)
                                                                     for k, v in shares.items()},
        weights=Path(args.weights).name, rule="paper",
        equivalent_command=f"python scripts/predict_naip.py --weights {Path(args.weights).name} --input {scene}.tif "
                           f"--output {scene}_class.tif --window {col} {row} {w} {h}")})


def update_json(path, data):
    old = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    old.update(data)
    path.write_text(json.dumps(old, indent=1), encoding="utf-8")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", help="release weights (fuelmap_unet_v1.pth)")
    p.add_argument("--test-npz", help="release test package (fuelmap_test_1y.npz)")
    p.add_argument("--index", default="data/test_array_index.csv", help="per-row attributes of the test package")
    p.add_argument("--pred-cache", help="npz of test predictions: reused if it exists, written otherwise")
    p.add_argument("--naip", help="4-band NAIP GeoTIFF for naip_scene_demo.png")
    p.add_argument("--naip-window", type=int, nargs=4, metavar=("COL", "ROW", "WIDTH", "HEIGHT"),
                   help="pixel window of --naip (default: central 2000 x 2000 px)")
    p.add_argument("--zoom", type=int, nargs=3, metavar=("COL", "ROW", "SIZE"),
                   help="zoom square inside the window, in window pixels (default: central 500 px)")
    p.add_argument("--out-dir", default="docs/figures/demo")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--device", default="auto", help="auto, cpu, cuda, ...")
    args = p.parse_args(argv)
    if not args.test_npz and not args.naip:
        p.error("give --test-npz and/or --naip")
    if args.naip and not args.weights:
        p.error("--naip needs --weights")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    plt = setup_mpl()
    if args.test_npz:
        run_tiles(args, plt, out)
    if args.naip:
        run_naip(args, plt, out)


if __name__ == "__main__":
    main()
