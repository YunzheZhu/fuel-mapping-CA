#!/usr/bin/env python
# Copyright (C) 2023-2026 Yunzhe Zhu
# SPDX-License-Identifier: GPL-3.0-only
"""Aggregate a 0.6 m class map to fractional cover on a coarse grid (default 30 m).

For every output cell the script reports the fraction of valid 0.6 m pixels
that are tree, shrub, building and woody (tree or shrub), plus the fraction of
the cell covered by valid pixels. Pixels outside classes 0-3 (nodata 255, the
input's declared nodata, negative values, or the "uncertain" class 4 of v2
maps) are excluded. The output grid is copied
from a reference raster (e.g. a LANDFIRE 30 m layer) or built from --res/--crs.
Processing is windowed, so inputs of any size (including VRT mosaics) work.

  python scripts/aggregate_30m.py --input county_2020_class.tif --reference LF2020_EVC_CA.tif \
      --output county_2020_fraction_30m.tif

  python scripts/aggregate_30m.py --input scene_class.tif --res 30 --crs EPSG:6414 --output scene_30m.tif

Output: float32, 5 bands (tree, shrub, building, woody, valid), nodata -1 for
cells whose valid fraction is below --min-valid. Fractions are computed by
GDAL "average" resampling of per-class indicator layers, divided by the
average of the valid-pixel indicator.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

BANDS = ("tree", "shrub", "building", "woody", "valid")
OUT_NODATA = -1.0
PAD = 4  # output cells resampled around each block and cropped afterwards


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, help="class map (GeoTIFF/VRT; 0 other, 1 tree, 2 shrub, 3 building)")
    p.add_argument("--output", required=True, help="output GeoTIFF")
    grid = p.add_mutually_exclusive_group()
    grid.add_argument("--reference", help="raster whose CRS, transform and size define the output grid")
    grid.add_argument("--res", type=float, default=30.0, help="output cell size (when no --reference)")
    p.add_argument("--crs", help="output CRS when no --reference (default: the input CRS)")
    p.add_argument("--exclude-mask", help="raster; pixels with a non-zero value are set to 'other' (e.g. water); "
                                          "the mask's nodata and areas outside it are not masked")
    p.add_argument("--min-valid", type=float, default=0.5,
                   help="cells with a smaller valid-pixel fraction are set to nodata")
    p.add_argument("--block", type=int, default=64, help="output cells per processing block side")
    return p.parse_args(argv)


def output_grid(src, args):
    """(crs, transform, width, height) of the output grid."""
    import rasterio
    from rasterio.transform import from_origin
    from rasterio.warp import transform_bounds

    if args.reference:
        with rasterio.open(args.reference) as ref:
            return ref.crs, ref.transform, ref.width, ref.height
    crs = rasterio.crs.CRS.from_user_input(args.crs) if args.crs else src.crs
    left, bottom, right, top = transform_bounds(src.crs, crs, *src.bounds, densify_pts=21)
    r = args.res
    left, bottom = math.floor(left / r) * r, math.floor(bottom / r) * r
    right, top = math.ceil(right / r) * r, math.ceil(top / r) * r
    return crs, from_origin(left, top, r, r), int(round((right - left) / r)), int(round((top - bottom) / r))


def main(argv=None):
    args = parse_args(argv)
    import rasterio
    from rasterio.warp import Resampling, reproject, transform_bounds
    from rasterio.windows import Window, bounds as win_bounds, from_bounds, transform as win_transform

    with rasterio.open(args.input) as src:
        crs, transform, width, height = output_grid(src, args)
        print(f"output grid: {width} x {height} cells, {transform.a:g} m, {crs.to_string()}")
        profile = dict(driver="GTiff", dtype="float32", count=len(BANDS), crs=crs, transform=transform,
                       width=width, height=height, nodata=OUT_NODATA, compress="deflate", predictor=3,
                       tiled=width >= 256 and height >= 256, BIGTIFF="IF_SAFER")
        if profile["tiled"]:
            profile.update(blockxsize=256, blockysize=256)
        mask_src = rasterio.open(args.exclude_mask) if args.exclude_mask else None
        full = Window(0, 0, src.width, src.height)
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(args.output, "w", **profile) as dst:
            dst.descriptions = BANDS
            blocks = [(r, c) for r in range(0, height, args.block) for c in range(0, width, args.block)]
            for n, (r, c) in enumerate(blocks, 1):
                win = Window(c, r, min(args.block, width - c), min(args.block, height - r))
                h, w = int(win.height), int(win.width)
                # GDAL "average" is wrong for destination windows narrower than about 4 cells, so every block is
                # resampled with a margin of PAD cells on each side and then cropped.
                pw = Window(c - PAD, r - PAD, w + 2 * PAD, h + 2 * PAD)
                out = np.full((len(BANDS), h, w), OUT_NODATA, np.float32)
                b = transform_bounds(crs, src.crs, *win_bounds(pw, transform), densify_pts=21)
                fw = from_bounds(*b, transform=src.transform)
                c0, r0 = math.floor(fw.col_off) - 2, math.floor(fw.row_off) - 2  # with a small margin
                sw = Window(c0, r0, math.ceil(fw.col_off + fw.width) + 2 - c0, math.ceil(fw.row_off + fw.height) + 2 - r0)
                try:
                    sw.intersection(full)
                    overlaps = True
                except Exception:  # rasterio raises WindowError when windows do not intersect
                    overlaps = False
                if overlaps:
                    cls = src.read(1, window=sw, boundless=True, fill_value=src.nodata if src.nodata is not None else 255)
                    s_tr = win_transform(sw, src.transform)
                    valid = (cls >= 0) & (cls <= 3)
                    if src.nodata is not None:
                        valid &= cls != src.nodata
                    if mask_src is not None:
                        # dst_nodata=0: the mask's own nodata and areas outside the mask count as "not masked"
                        m = np.zeros(cls.shape, np.uint8)
                        reproject(rasterio.band(mask_src, 1), m, dst_transform=s_tr, dst_crs=src.crs, dst_nodata=0,
                                  resampling=Resampling.nearest)
                        cls = np.where(valid & (m != 0), 0, cls)
                    ind = np.stack([cls == 1, cls == 2, cls == 3, (cls == 1) | (cls == 2), valid]).astype(np.float32)
                    ind[:4] *= valid
                    avg = np.zeros((len(BANDS), h + 2 * PAD, w + 2 * PAD), np.float32)
                    reproject(ind, avg, src_transform=s_tr, src_crs=src.crs, dst_transform=win_transform(pw, transform),
                              dst_crs=crs, resampling=Resampling.average)
                    avg = avg[:, PAD:PAD + h, PAD:PAD + w]
                    v = avg[4]
                    ok = (v > 0) & (v >= args.min_valid)
                    out[:4][:, ok] = np.clip(avg[:4][:, ok] / v[ok], 0, 1)
                    out[4] = v
                    out[4][~(v > 0)] = OUT_NODATA
                dst.write(out, window=win)
                if n % 50 == 0 or n == len(blocks):
                    print(f"  block {n}/{len(blocks)}", flush=True)
        if mask_src is not None:
            mask_src.close()
    print(f"wrote {args.output} (bands: {', '.join(BANDS)}; nodata {OUT_NODATA})")


if __name__ == "__main__":
    sys.exit(main())
