#!/usr/bin/env python
# Copyright (C) 2023-2026 Yunzhe Zhu
# SPDX-License-Identifier: GPL-3.0-only
"""Mosaic class maps into a Cloud-Optimized GeoTIFF, or convert them one by one.

Uses the GDAL Python bindings (osgeo.gdal >= 3.1): BuildVRT (or Warp when
reprojecting) followed by Translate to the COG driver, DEFLATE compression,
nodata 255 and nearest-neighbour overviews, so class values are never mixed.

  # mosaic all scene predictions of a county into one COG in California Albers
  python scripts/mosaic_cog.py --input pred_2020/ --pattern "*_class.tif" \
      --output county_2020_class_cog.tif --t-srs EPSG:6414 --res 0.6 --tap

  # convert each file to a COG (same grid)
  python scripts/mosaic_cog.py --input pred_2020/ --per-file pred_2020_cog/

When reprojecting, every input is warped to one common grid: pixel size --res
(default: that of the first input after warping), with pixel edges at
multiples of it (--tap, the default; --no-tap keeps each input's own origin,
which misaligns the parts of a mosaic). Where inputs overlap, the last file in
the (sorted) input list wins.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", nargs="+", required=True, help="raster files and/or folders")
    p.add_argument("--pattern", default="*.tif", help="file pattern for input folders")
    out = p.add_mutually_exclusive_group(required=True)
    out.add_argument("--output", help="output COG for a mosaic of all inputs")
    out.add_argument("--per-file", metavar="DIR", help="write one COG per input into DIR")
    p.add_argument("--t-srs", help="reproject to this CRS (e.g. EPSG:6414); nearest neighbour")
    p.add_argument("--res", type=float,
                   help="output pixel size in target units (with --t-srs; default: that of the first input)")
    p.add_argument("--tap", action=argparse.BooleanOptionalAction, default=True,
                   help="with --t-srs, align pixel edges to multiples of the pixel size")
    p.add_argument("--nodata", type=float, default=255, help="nodata value of inputs and output")
    p.add_argument("--overview-resampling", default="NEAREST", help="COG overview resampling (NEAREST or MODE)")
    p.add_argument("--blocksize", type=int, default=512)
    p.add_argument("--vrt", help="also keep the intermediate VRT at this path")
    return p.parse_args(argv)


def list_inputs(items, pattern):
    files = []
    for item in items:
        path = Path(item)
        files += sorted(path.glob(pattern)) if path.is_dir() else [path]
    return [str(f) for f in files]


def build_source(gdal, inputs, args, vrt_path=""):
    """Return a GDAL dataset (VRT) that mosaics and optionally reprojects ``inputs``.

    With ``--t-srs`` every input is first warped to its own virtual raster (a
    VRT produced by gdal.Warp only reads its first source), so inputs may come
    from different UTM zones. All parts share one pixel size and, with
    ``--tap``, one pixel grid; the warped rasters are then mosaicked.
    """
    sources = inputs
    if args.t_srs:
        kw = dict(format="VRT", dstSRS=args.t_srs, resampleAlg="near", srcNodata=args.nodata,
                  dstNodata=args.nodata)
        res = args.res
        if not res:  # common pixel size: that of the first input after warping
            probe = gdal.Warp("/vsimem/fuelmap_probe.vrt", inputs[0], **kw)
            res = abs(probe.GetGeoTransform()[1])
            probe = None
            gdal.Unlink("/vsimem/fuelmap_probe.vrt")
        kw.update(xRes=res, yRes=res, targetAlignedPixels=args.tap)
        part_dir = f"{vrt_path}.parts" if vrt_path else "/vsimem/fuelmap_mosaic"
        if vrt_path:
            Path(part_dir).mkdir(parents=True, exist_ok=True)
        sources = []
        for i, f in enumerate(inputs):
            part = f"{part_dir}/{i:06d}_{Path(f).stem}.vrt"
            gdal.Warp(part, f, **kw)
            sources.append(part)
    return gdal.BuildVRT(vrt_path, sources, srcNodata=args.nodata, VRTNodata=args.nodata, resampleAlg="nearest",
                         resolution="highest")


def to_cog(gdal, src, dst_path, args):
    opts = ["COMPRESS=DEFLATE", "BIGTIFF=IF_SAFER", f"BLOCKSIZE={args.blocksize}", "NUM_THREADS=ALL_CPUS",
            f"OVERVIEW_RESAMPLING={args.overview_resampling}"]
    Path(dst_path).parent.mkdir(parents=True, exist_ok=True)
    ds = gdal.Translate(str(dst_path), src, format="COG", creationOptions=opts, noData=args.nodata)
    if ds is None:
        raise RuntimeError(f"gdal.Translate failed for {dst_path}")
    ds = None  # flush


def main(argv=None):
    args = parse_args(argv)
    try:
        from osgeo import gdal
    except ImportError:
        sys.exit("osgeo.gdal is required (e.g. conda install -c conda-forge gdal)")
    gdal.UseExceptions()
    inputs = list_inputs(args.input, args.pattern)
    if not inputs:
        sys.exit("no input rasters found")
    if args.per_file:
        for n, f in enumerate(inputs, 1):
            dst = Path(args.per_file) / Path(f).name
            src = build_source(gdal, [f], args) if args.t_srs else gdal.Open(f)
            to_cog(gdal, src, dst, args)
            print(f"[{n}/{len(inputs)}] {dst}")
        return
    if args.vrt:
        Path(args.vrt).parent.mkdir(parents=True, exist_ok=True)
    src = build_source(gdal, inputs, args, args.vrt or "")
    print(f"mosaic of {len(inputs)} rasters: {src.RasterXSize} x {src.RasterYSize} px")
    to_cog(gdal, src, args.output, args)
    src = None
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
