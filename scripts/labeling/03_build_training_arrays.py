#!/usr/bin/env python
# Copyright (C) 2023-2026 Yunzhe Zhu
# SPDX-License-Identifier: GPL-3.0-only
# Part of fuel-mapping-CA (https://github.com/YunzheZhu/fuel-mapping-CA).
"""Stack (NAIP tile, label tile) pairs into training/testing arrays.

Writes three files for one split:

* ``<out>_images.npz`` - key ``arr_0``: (N, 512, 512, 4) uint8 NAIP DN, bands R, G, B, NIR
* ``<out>_labels.npz`` - key ``arr_0``: (N, 512, 512) uint8, 0 other, 1 tree,
  2 shrub, 3 building, 255 ignore
* ``<out>_tiles.csv``  - one row per array index: tile_id, year, split, source
  paths and per-class pixel counts, so every array entry can be traced back to
  its tile (the original 2023-24 arrays did not record this).

The layout (uncompressed ``np.savez``, key ``arr_0``, uint8 images scaled by
1/255 at training time) matches the arrays used to train the released model.
Label values other than 0-3 (e.g. the legacy no-label values 4 and 5) are
written as 255.

Pairs come from a CSV manifest (columns ``naip,label`` and optionally
``tile_id,year,split``) or from folders where each ``<stem>.tif`` NAIP tile has a
``<stem>_Label.tif`` label (default naming ``ID_<tile>_NAIP_<year>.tif``).
"Matched-year" (1y) and "all-years" (4y) arrays are built by choosing which
rows to include, e.g. one row per tile for the LiDAR year, or one row per tile
and NAIP year with labels re-masked by that year's NDVI (02_rasterize_labels.py).

Examples::

    python 03_build_training_arrays.py --manifest train_pairs.csv --split train --out arrays/train_4y
    python 03_build_training_arrays.py --naip-dir tiles/test --out arrays/test_1y

Based on the training-array builders of the original pipeline.
"""
from __future__ import annotations

import argparse
import csv
import logging
import os
import re
from pathlib import Path

import numpy as np
import rasterio

NODATA = 255
NAME_RE = re.compile(r"ID_(?P<tile_id>.+?)_NAIP_(?P<year>\d{4})$")
log = logging.getLogger("build_arrays")


def pairs_from_manifest(path: str | Path, split: str | None) -> list[dict]:
    """Rows of a CSV manifest; keeps only rows of ``split`` if both are given."""
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if split and rows and "split" in rows[0]:
        rows = [r for r in rows if r["split"].strip().lower() == split.lower()]
    return rows


def pairs_from_dirs(naip_dir: str | Path, label_dir: str | Path | None,
                    label_suffix: str = "_Label") -> list[dict]:
    """Pair ``<stem>.tif`` NAIP tiles with ``<stem><label_suffix>.tif`` labels."""
    label_dir = Path(label_dir or naip_dir)
    rows = []
    for naip in sorted(Path(naip_dir).glob("*.tif")):
        if NAME_RE.search(naip.stem) is None:  # skips *_Label, *_NDVI, ... files
            continue
        label = label_dir / f"{naip.stem}{label_suffix}.tif"
        if label.exists():
            rows.append({"naip": str(naip), "label": str(label)})
        else:
            log.warning("no label for %s", naip.name)
    return rows


def read_pair(naip: str, label: str, size: int):
    """Read one aligned pair; returns (image HxWx4 uint8, label HxW uint8)."""
    with rasterio.open(naip) as src_img, rasterio.open(label) as src_lbl:
        if (src_img.height, src_img.width) != (size, size):
            raise ValueError(f"NAIP tile is {src_img.height}x{src_img.width}, expected {size}x{size}")
        if (src_lbl.height, src_lbl.width) != (size, size) or src_lbl.crs != src_img.crs \
                or not src_lbl.transform.almost_equals(src_img.transform):
            raise ValueError("label grid does not match the NAIP tile grid")
        img = src_img.read([1, 2, 3, 4])
        lbl = src_lbl.read(1)
    if img.dtype != np.uint8:  # some legacy tiles were stored as float32 DN
        if np.nanmax(img) > 255 or np.nanmin(img) < 0 or np.nanmax(img) <= 1:
            raise ValueError(f"NAIP values ({img.dtype}) do not look like 8-bit DN (0-255)")
        img = np.nan_to_num(np.rint(img)).astype(np.uint8)
    lbl = np.nan_to_num(lbl, nan=NODATA).astype(np.int64)
    lbl[(lbl < 0) | ((lbl > 3) & (lbl != NODATA))] = NODATA
    return np.moveaxis(img, 0, -1), lbl.astype(np.uint8)


def build_arrays(rows: list[dict], out: str | Path, size: int = 512,
                 split: str | None = None, compress: bool = False) -> int:
    """Write ``<out>_images.npz``, ``<out>_labels.npz`` and ``<out>_tiles.csv``.

    Tiles are streamed through temporary memory-mapped .npy files, so RAM use
    stays at a few tiles regardless of N. Returns the number of tiles written.
    """
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp_img, tmp_lbl = out.with_name(out.name + "_images.tmp.npy"), out.with_name(out.name + "_labels.tmp.npy")
    images = np.lib.format.open_memmap(tmp_img, "w+", np.uint8, (len(rows), size, size, 4))
    labels = np.lib.format.open_memmap(tmp_lbl, "w+", np.uint8, (len(rows), size, size))

    records, n = [], 0
    for row in rows:
        try:
            img, lbl = read_pair(row["naip"], row["label"], size)
        except Exception as e:
            log.warning("skipped %s: %s", row["naip"], e)
            continue
        images[n], labels[n] = img, lbl
        m = NAME_RE.search(Path(row["naip"]).stem)
        counts = np.bincount(lbl.ravel(), minlength=256)
        records.append({
            "index": n,
            "tile_id": row.get("tile_id") or (m["tile_id"] if m else Path(row["naip"]).stem),
            "year": row.get("year") or (m["year"] if m else ""),
            "split": row.get("split") or split or "",
            "naip": row["naip"], "label": row["label"],
            "n_other": int(counts[0]), "n_tree": int(counts[1]), "n_shrub": int(counts[2]),
            "n_building": int(counts[3]), "n_ignore": int(counts[NODATA]),
        })
        n += 1

    save = np.savez_compressed if compress else np.savez
    save(out.with_name(out.name + "_images.npz"), images[:n])
    save(out.with_name(out.name + "_labels.npz"), labels[:n])
    del images, labels
    os.remove(tmp_img)
    os.remove(tmp_lbl)

    with open(out.with_name(out.name + "_tiles.csv"), "w", newline="") as f:
        fields = ["index", "tile_id", "year", "split", "naip", "label",
                  "n_other", "n_tree", "n_shrub", "n_building", "n_ignore"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)

    if n:
        tot = {k: sum(r[k] for r in records) for k in ("n_other", "n_tree", "n_shrub", "n_building", "n_ignore")}
        px = n * size * size
        log.info("%d tiles -> %s_{images,labels}.npz; pixel shares: %s", n, out,
                 ", ".join(f"{k[2:]} {v / px:.3f}" for k, v in tot.items()))
    return n


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--manifest", help="CSV with columns naip,label[,tile_id,year,split]")
    src.add_argument("--naip-dir", help="folder of NAIP tiles named ID_<tile>_NAIP_<year>.tif")
    p.add_argument("--label-dir", help="folder of <stem>_Label.tif labels (default: --naip-dir)")
    p.add_argument("--split", help="split name to record (and to select rows of a manifest)")
    p.add_argument("--out", required=True, help="output prefix, e.g. arrays/train_4y")
    p.add_argument("--size", type=int, default=512, help="tile size in pixels (default 512)")
    p.add_argument("--compress", action="store_true", help="use np.savez_compressed")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    rows = pairs_from_manifest(args.manifest, args.split) if args.manifest \
        else pairs_from_dirs(args.naip_dir, args.label_dir)
    if not rows:
        p.error("no (NAIP, label) pairs found")
    build_arrays(rows, args.out, args.size, args.split, args.compress)


if __name__ == "__main__":
    main()
