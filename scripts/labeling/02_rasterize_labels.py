#!/usr/bin/env python
# Copyright (C) 2023-2026 Yunzhe Zhu
# SPDX-License-Identifier: GPL-3.0-only
# Part of fuel-mapping-CA (https://github.com/YunzheZhu/fuel-mapping-CA).
"""Rasterize labelled object polygons to a 0.6 m label tile (paper Sec. 2.4, Fig. 2 step 4).

Takes the (manually checked) object polygons from ``01_lidar_segmentation.R``
and writes a uint8 label GeoTIFF on exactly the pixel grid of a NAIP tile:

    0 = other, 1 = tree, 2 = shrub, 3 = building, 255 = nodata / not labelled

Rules (ported from the original label builders):

* Tree/shrub are re-checked against height: woody polygons with
  ``zq95 >= 5 m`` are trees, the rest shrubs.
* Overlaps are resolved with priority tree > shrub > building.
* Optional county building footprints (``--buildings``) override the LiDAR
  labels: LiDAR "building" polygons are re-assigned to tree/shrub by height
  (unless ``--keep-lidar-buildings``) and every pixel inside a footprint becomes
  building. This is how the paper's footprint update was implemented.
* NDVI mask: woody pixels whose NAIP NDVI is not above 0.11 become other.
  NDVI = (NIR - R) / (NIR + R) on DN/255 (the scaling cancels). Buildings are
  never NDVI-masked.
* ``--buildings-only`` writes footprint-only labels (3 inside footprints, 255
  elsewhere). Some urban tiles in the training/test arrays were labelled this
  way, with all non-building pixels ignored.
* Pixels without NAIP data (all bands 0 or masked) and, with ``--chm``, pixels
  outside the LiDAR coverage are set to 255.

Examples
--------
Single tile::

    python 02_rasterize_labels.py --naip ID_067789_00_NAIP_2016.tif \\
        --crowns ID_67789_Row_76_crowns.gpkg --out ID_067789_00_NAIP_2016_Label.tif

With county footprints, and a batch of tiles from a CSV manifest
(columns ``naip,crowns,out``; optional ``buildings_only`` = 0/1)::

    python 02_rasterize_labels.py --manifest tiles.csv --buildings footprints.gpkg

Ported from the 2023-24 label scripts (single-tile rasterizer, building
footprint merge and the per-tile part of the training-array builder).
"""
from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.warp import Resampling, reproject
from shapely.geometry import box

OTHER, TREE, SHRUB, BUILDING, NODATA = 0, 1, 2, 3, 255
CLASS_FIELDS = ("Clss_95", "Classes_zq", "Classes")  # current and legacy names
NDVI_THRESHOLD = 0.11
TREE_HEIGHT = 5.0

log = logging.getLogger("rasterize_labels")


def read_naip(path: str | Path):
    """Read a 4-band NAIP tile (R, G, B, NIR).

    Returns the image as float32 DN array (4, H, W), the raster profile, and a
    boolean valid-data mask (False where every band is 0 or the pixel is masked).
    """
    with rasterio.open(path) as src:
        if src.count < 4:
            raise ValueError(f"{path}: expected 4 bands (R, G, B, NIR), found {src.count}")
        img = src.read([1, 2, 3, 4]).astype(np.float32)
        valid = (src.dataset_mask() > 0) & np.any(img > 0, axis=0)
        profile = src.profile.copy()
    return img, profile, valid


def ndvi(img: np.ndarray) -> np.ndarray:
    """NDVI from a (4, H, W) R, G, B, NIR array; 0 where NIR + R == 0."""
    red, nir = img[0] / 255.0, img[3] / 255.0
    denom = nir + red
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(denom > 0, (nir - red) / denom, 0.0).astype(np.float32)


def burn(geoms, shape, transform) -> np.ndarray:
    """Boolean mask of pixels whose centre falls in any of the geometries."""
    geoms = [g for g in geoms if g is not None and not g.is_empty]
    if not geoms:
        return np.zeros(shape, dtype=bool)
    return rasterize(((g, 1) for g in geoms), out_shape=shape, transform=transform,
                     fill=0, dtype="uint8").astype(bool)


def read_polygons(path: str | Path, crs, bounds=None) -> gpd.GeoDataFrame:
    """Read polygons (optionally only those in ``bounds``, given in ``crs``) and
    reproject them to ``crs``. Files without a CRS are assumed to be in ``crs``."""
    bbox = None
    if bounds is not None:  # read only features near the tile (fast for county files)
        src_crs = gpd.read_file(path, rows=1).crs or crs
        bbox = tuple(gpd.GeoSeries([box(*bounds)], crs=crs).to_crs(src_crs).total_bounds)
    gdf = gpd.read_file(path, bbox=bbox)
    if gdf.crs is None:
        log.warning("%s has no CRS; assuming the tile CRS (%s)", path, crs)
        gdf = gdf.set_crs(crs)
    return gdf.to_crs(crs)


def object_classes(crowns: gpd.GeoDataFrame, class_field: str | None, height_field: str,
                   tree_height: float = TREE_HEIGHT) -> np.ndarray:
    """Class per polygon: woody classes re-checked against height; anything that is
    not 1, 2 or 3 (e.g. 0 or the legacy 4 for low objects) is treated as other."""
    field = class_field or next((f for f in CLASS_FIELDS if f in crowns.columns), None)
    if field is None or field not in crowns.columns:
        raise KeyError(f"no class field found (tried {class_field or CLASS_FIELDS})")
    cls = crowns[field].fillna(OTHER).astype(int).to_numpy()
    if height_field in crowns.columns:
        h = crowns[height_field].to_numpy(dtype=float)
        woody = np.isin(cls, (TREE, SHRUB)) & np.isfinite(h)
        cls[woody] = np.where(h[woody] >= tree_height, TREE, SHRUB)
    else:
        log.warning("height field %r not found; tree/shrub classes used as given", height_field)
    return cls


def lidar_coverage(chm_path: str | Path, profile) -> np.ndarray:
    """Boolean mask of tile pixels covered by a valid CHM value (nearest resampling)."""
    out = np.full((profile["height"], profile["width"]), np.nan, dtype=np.float32)
    with rasterio.open(chm_path) as src:
        reproject(rasterio.band(src, 1), out, dst_transform=profile["transform"],
                  dst_crs=profile["crs"], dst_nodata=np.nan, resampling=Resampling.nearest)
    return np.isfinite(out)


def make_label(naip: str | Path, crowns: str | Path | None = None,
               buildings: str | Path | None = None, buildings_only: bool = False,
               keep_lidar_buildings: bool = False, chm: str | Path | None = None,
               class_field: str | None = None, height_field: str = "zq95",
               ndvi_threshold: float = NDVI_THRESHOLD,
               tree_height: float = TREE_HEIGHT) -> tuple[np.ndarray, dict]:
    """Build the label array for one NAIP tile.

    Returns ``(label, profile)`` where ``label`` is a (H, W) uint8 array with
    0-3 and 255, and ``profile`` is the NAIP raster profile (grid and CRS).
    """
    img, profile, valid = read_naip(naip)
    shape, transform, crs = (profile["height"], profile["width"]), profile["transform"], profile["crs"]
    west, south, east, north = rasterio.transform.array_bounds(shape[0], shape[1], transform)
    bounds = (west, south, east, north)
    footprints = burn(read_polygons(buildings, crs, bounds).geometry, shape, transform) \
        if buildings else None

    if buildings_only:
        if footprints is None:
            raise ValueError("--buildings-only needs --buildings")
        label = np.full(shape, NODATA, dtype=np.uint8)
        label[footprints] = BUILDING
    else:
        if crowns is None:
            raise ValueError("--crowns is required unless --buildings-only is used")
        gdf = read_polygons(crowns, crs, bounds)
        cls = object_classes(gdf, class_field, height_field, tree_height)
        if footprints is not None and not keep_lidar_buildings and height_field in gdf.columns:
            # Footprints are authoritative for buildings: LiDAR planar objects
            # outside them are treated as woody objects and split by height.
            h = gdf[height_field].to_numpy(dtype=float)
            lidar_bldg = (cls == BUILDING) & np.isfinite(h)
            cls[lidar_bldg] = np.where(h[lidar_bldg] >= tree_height, TREE, SHRUB)
        masks = {c: burn(gdf.geometry[cls == c], shape, transform) for c in (TREE, SHRUB, BUILDING)}
        green = ndvi(img) > ndvi_threshold  # NDVI mask first, then overlaps (as originally)
        masks[TREE] &= green
        masks[SHRUB] &= green
        label = np.full(shape, OTHER, dtype=np.uint8)
        for c in (BUILDING, SHRUB, TREE):  # later classes win: tree > shrub > building
            label[masks[c]] = c
        if footprints is not None:
            label[footprints] = BUILDING

    label[~valid] = NODATA
    if chm:
        label[~lidar_coverage(chm, profile)] = NODATA
    return label, profile


def write_label(label: np.ndarray, profile: dict, out: str | Path) -> None:
    """Write a single-band uint8 label GeoTIFF (nodata 255, DEFLATE)."""
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    meta = {k: profile[k] for k in ("height", "width", "crs", "transform")}
    meta.update(driver="GTiff", count=1, dtype="uint8", nodata=NODATA, compress="deflate")
    with rasterio.open(out, "w", **meta) as dst:
        dst.write(label, 1)


def summary(label: np.ndarray) -> str:
    names = {OTHER: "other", TREE: "tree", SHRUB: "shrub", BUILDING: "building", NODATA: "nodata"}
    vals, counts = np.unique(label, return_counts=True)
    return ", ".join(f"{names.get(int(v), v)} {c}" for v, c in zip(vals, counts))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--naip", help="NAIP tile GeoTIFF (R, G, B, NIR); defines the output grid")
    p.add_argument("--crowns", help="object polygons from 01_lidar_segmentation.R (gpkg/shp)")
    p.add_argument("--out", help="output label GeoTIFF")
    p.add_argument("--manifest", help="CSV with columns naip,crowns,out[,buildings_only] for batch runs")
    p.add_argument("--buildings", help="building footprint polygons that override LiDAR labels")
    p.add_argument("--buildings-only", action="store_true",
                   help="label footprints as building and everything else as 255 (ignore)")
    p.add_argument("--keep-lidar-buildings", action="store_true",
                   help="with --buildings, keep LiDAR building polygons outside footprints")
    p.add_argument("--chm", help="CHM GeoTIFF; tile pixels without LiDAR data become 255")
    p.add_argument("--class-field", help=f"polygon class column (default: first of {CLASS_FIELDS})")
    p.add_argument("--height-field", default="zq95", help="height column used for tree/shrub (default zq95)")
    p.add_argument("--ndvi-threshold", type=float, default=NDVI_THRESHOLD,
                   help="woody pixels with NDVI <= this become other (default 0.11)")
    p.add_argument("--tree-height", type=float, default=TREE_HEIGHT, help="tree/shrub height split, m (default 5)")
    p.add_argument("--overwrite", action="store_true", help="overwrite existing outputs")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.manifest:
        with open(args.manifest, newline="") as f:
            jobs = list(csv.DictReader(f))
    elif args.naip and args.out:
        jobs = [{"naip": args.naip, "crowns": args.crowns, "out": args.out,
                 "buildings_only": args.buildings_only}]
    else:
        p.error("give --naip and --out (plus --crowns), or --manifest")

    for job in jobs:
        out = Path(job["out"])
        if out.exists() and not args.overwrite:
            log.info("exists, skipped: %s", out)
            continue
        bonly = str(job.get("buildings_only") or args.buildings_only).strip().lower() in ("1", "true", "yes")
        try:
            label, profile = make_label(
                job["naip"], job.get("crowns") or None, args.buildings, bonly,
                args.keep_lidar_buildings, job.get("chm") or args.chm, args.class_field,
                args.height_field, args.ndvi_threshold, args.tree_height)
        except Exception as e:  # keep going in batch mode
            log.error("FAILED %s: %s", out, e)
            continue
        write_label(label, profile, out)
        log.info("%s: %s", out, summary(label))


if __name__ == "__main__":
    main()
