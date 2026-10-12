#!/usr/bin/env python
# Copyright (C) 2023-2026 Yunzhe Zhu
# SPDX-License-Identifier: GPL-3.0-only
r"""Pixel-based accuracy of predicted class maps against reference labels.

Confusion matrices have rows = reference and columns = prediction; UA = user's
accuracy (precision), PA = producer's accuracy (recall). Pixels whose
reference or prediction is not a class 0-3 (ignore/nodata 255, legacy
"no label" values 4/5) are excluded.

  # npz layout (output of predict_tiles.py --npz)
  python scripts/evaluate.py --pred-npy pred/pred_class.npy --label-npz release/fuelmap_test_1y.npz \
      --label-key labels --out-json metrics.json --out-csv metrics.csv --per-tile-csv per_tile.csv

  # GeoTIFF folders: pred/<id>_class.tif vs labels/<id>.tif
  python scripts/evaluate.py --pred-dir pred/ --label-dir labels/ --out-csv metrics.csv

  # labels shared by several NAIP years: pred/ID_010_NAIP_2018_class.tif vs labels/ID_010_Label.tif
  python scripts/evaluate.py --pred-dir pred/ --label-dir labels/ --stem-regex "^(ID_\d+)" \
      --label-template "{stem}_Label.tif"

  # stratified: a CSV with one row per tile (column tile_id + attributes)
  python scripts/evaluate.py ... --attributes tiles.csv --id-column tile_id --group-by WUI RRK year

Tile ids (for --attributes): for GeoTIFFs the prediction file stem without
"_class" (i.e. the input tile name); for npz inputs the index of the tile in
the label array (read from tile_index.npy next to the prediction file when
present).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fuelmap import CLASS_NAMES, NUM_CLASSES  # noqa: E402
from fuelmap.dataset import NpzArray, read_raster  # noqa: E402
from fuelmap.metrics import confusion_matrix, format_metrics, metrics_from_cm  # noqa: E402

CM_KEYS = [f"cm_{r}_{c}" for r in CLASS_NAMES for c in CLASS_NAMES]  # cm_<reference>_<prediction>


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_argument_group("npz layout")
    g.add_argument("--pred-npy", help="predicted classes (N, H, W) .npy")
    g.add_argument("--label-npz", help="reference labels (M, H, W) .npz")
    g.add_argument("--label-key", default=None, help="array name inside --label-npz")
    g.add_argument("--index-npy", default=None,
                   help="indices of the predicted tiles in the label array (default: tile_index.npy next to "
                        "--pred-npy if present, else 0..N-1)")
    g = p.add_argument_group("GeoTIFF layout")
    g.add_argument("--pred-dir", help="folder of predicted class GeoTIFFs")
    g.add_argument("--label-dir", help="folder of reference label GeoTIFFs")
    g.add_argument("--pred-pattern", default="*_class.tif")
    g.add_argument("--label-template", default="{stem}.tif",
                   help="label file name; {stem} is the tile id (prediction stem without '_class') or the "
                        "first group of --stem-regex")
    g.add_argument("--stem-regex", default=None,
                   help=r"regex applied to the tile id to build {stem}, e.g. '^(ID_\d+)' for labels shared by years")
    g = p.add_argument_group("stratification and output")
    g.add_argument("--attributes", help="CSV with one row per tile")
    g.add_argument("--id-column", default="tile_id", help="tile id column in --attributes")
    g.add_argument("--group-by", nargs="*", default=[], help="attribute columns to report metrics by")
    g.add_argument("--undefined", choices=["nan", "zero"], default="nan",
                   help="value of per-class metrics with a zero denominator (nan = skipped in class means)")
    g.add_argument("--out-json", help="write all metrics as JSON")
    g.add_argument("--out-csv", help="write one row per group (overall first) as CSV")
    g.add_argument("--per-tile-csv", help="write each tile's confusion matrix as CSV")
    args = p.parse_args(argv)
    if bool(args.pred_npy) == bool(args.pred_dir):
        p.error("give either --pred-npy/--label-npz or --pred-dir/--label-dir")
    if args.pred_npy and not args.label_npz or args.pred_dir and not args.label_dir:
        p.error("missing --label-npz or --label-dir")
    return args


def tile_cms_npz(args):
    """Yield (tile_id, confusion matrix) for the npz layout."""
    pred = np.load(args.pred_npy, mmap_mode="r")
    labels = NpzArray(args.label_npz, args.label_key)
    idx_path = Path(args.index_npy) if args.index_npy else Path(args.pred_npy).with_name("tile_index.npy")
    index = np.load(idx_path) if idx_path.exists() else np.arange(len(pred))
    if len(index) != len(pred):
        sys.exit(f"{len(pred)} predicted tiles but {len(index)} indices in {idx_path}")
    for k, i in enumerate(index):
        ref = np.asarray(labels[int(i)])
        if ref.shape != pred[k].shape:
            sys.exit(f"tile {i}: label shape {ref.shape} != prediction shape {pred[k].shape}")
        yield str(int(i)), confusion_matrix(ref, pred[k])


def tile_cms_tif(args):
    """Yield (tile_id, confusion matrix) for the GeoTIFF layout."""
    import re

    preds = sorted(Path(args.pred_dir).glob(args.pred_pattern))
    if not preds:
        sys.exit(f"no files matching {args.pred_pattern} in {args.pred_dir}")
    n_missing = 0
    for pred_path in preds:
        tile_id = re.sub(r"_class$", "", pred_path.stem)
        key = tile_id
        if args.stem_regex:
            m = re.search(args.stem_regex, tile_id)
            key = (m.group(1) if m.groups() else m.group(0)) if m else None
        label_path = Path(args.label_dir) / args.label_template.format(stem=key) if key else None
        if label_path is None or not label_path.exists():
            n_missing += 1
            continue
        pred, ref = read_raster(pred_path, [1])[0][0], read_raster(label_path, [1])[0][0]
        if ref.shape != pred.shape:
            sys.exit(f"{tile_id}: label shape {ref.shape} != prediction shape {pred.shape}")
        if ref.dtype.kind == "f":
            ref = np.where(np.isfinite(ref), ref, -1)
        yield tile_id, confusion_matrix(ref.astype(np.int64), pred)
    if n_missing:
        print(f"warning: {n_missing} prediction(s) without a matching label were skipped")


def read_attributes(path, id_column):
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if rows and id_column not in rows[0]:
        sys.exit(f"column {id_column!r} not in {path}")
    return {r[id_column].strip(): r for r in rows}


def flat_row(name, value, n_tiles, m):
    row = {"group": name, "value": value, "n_tiles": n_tiles, "pixels": m["pixels"], "OA": m["OA"],
           "mIoU": m["mIoU"], "mean_F1": m["mean_F1"], "mean_UA": m["mean_UA"], "mean_PA": m["mean_PA"]}
    for key in ("UA", "PA", "F1", "IoU"):
        row.update({f"{key}_{c}": m[key][c] for c in CLASS_NAMES})
    row.update(dict(zip(CM_KEYS, np.asarray(m["confusion_matrix"]).ravel().tolist())))
    return row


def main(argv=None):
    args = parse_args(argv)
    tiles = list(tile_cms_npz(args) if args.pred_npy else tile_cms_tif(args))
    total = sum(cm for _, cm in tiles)
    overall = metrics_from_cm(total, undefined=args.undefined)
    print(f"{len(tiles)} tiles")
    print(format_metrics(overall))
    print("confusion matrix (rows = reference, columns = prediction; " + ", ".join(CLASS_NAMES) + "):")
    print(np.asarray(overall["confusion_matrix"]))

    groups = {}
    if args.group_by:
        if not args.attributes:
            sys.exit("--group-by needs --attributes")
        attrs = read_attributes(args.attributes, args.id_column)
        missing = [t for t, _ in tiles if t not in attrs]
        if missing:
            print(f"warning: {len(missing)} tiles have no attribute row (e.g. {missing[:3]})")
        for col in args.group_by:
            acc, n = defaultdict(lambda: np.zeros((NUM_CLASSES, NUM_CLASSES), np.int64)), defaultdict(int)
            for t, cm in tiles:
                if t in attrs:
                    v = attrs[t].get(col, "")
                    acc[v] += cm
                    n[v] += 1
            groups[col] = {v: dict(n_tiles=n[v], **metrics_from_cm(acc[v], undefined=args.undefined))
                           for v in sorted(acc)}
            print(f"\nby {col}:")
            for v, m in groups[col].items():
                print(f"  {v!s:<24s} n={m['n_tiles']:<5d} OA {m['OA']:.3f}  mIoU {m['mIoU']:.3f}  "
                      f"F1 " + " ".join(f"{c} {m['F1'][c]:.3f}" for c in CLASS_NAMES))

    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump({"n_tiles": len(tiles), "overall": overall, "groups": groups,
                       "convention": "confusion_matrix[reference][prediction]; UA = precision, PA = recall"},
                      f, indent=1)
    if args.out_csv:
        rows = [flat_row("all", "all", len(tiles), overall)]
        rows += [flat_row(col, v, m["n_tiles"], m) for col, vals in groups.items() for v, m in vals.items()]
        Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    if args.per_tile_csv:
        Path(args.per_tile_csv).parent.mkdir(parents=True, exist_ok=True)
        with open(args.per_tile_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["tile_id", "pixels", "OA"] + CM_KEYS)
            for t, cm in tiles:
                s = int(cm.sum())
                w.writerow([t, s, f"{np.trace(cm) / s:.6f}" if s else ""] + cm.ravel().tolist())


if __name__ == "__main__":
    main()
