#!/usr/bin/env python
# Copyright (C) 2023-2026 Yunzhe Zhu
# SPDX-License-Identifier: GPL-3.0-only
# Part of fuel-mapping-CA (https://github.com/YunzheZhu/fuel-mapping-CA).
"""Cut NAIP scenes into 512 x 512 px tiles at 0.6 m in EPSG:6414.

For every tile polygon, the NAIP scenes (4-band DOQQ GeoTIFFs or COGs, band
order R, G, B, NIR) that overlap it are warped with nearest-neighbour
resampling onto the tile's own pixel grid (origin = tile's north-west corner,
0.6 m pixels, EPSG:6414) and mosaicked (first scene in file-name order wins
where scenes overlap). Output: ``ID_<tile_id>_NAIP_<year>.tif``, uint8, 4 bands.
Pixels that no scene covers are 0 in all bands.

The original pipeline first reprojected each scene to EPSG:6414 at 0.6 m and
then cut the tile window from it; warping straight onto the tile grid gives the
same 512 x 512 grid with a single resampling step, so individual pixels can
differ by up to one source pixel from the published tiles. Radiometric
normalisation is not applied (the paper uses NAIP digital numbers).

NAIP imagery is public domain and can be obtained from, e.g., USGS
EarthExplorer or the USGS M2M API (free USGS account; the paper used M2M),
Microsoft Planetary Computer ("naip" collection) or Google Earth Engine
("USDA/NAIP/DOQQ"). This script only reads local files.

Example::

    python clip_naip_tiles.py --tiles tiles.gpkg --scenes /data/naip/2020 --year 2020 --outdir naip_tiles/

Based on the original NAIP processing scripts (reprojection + clipping part).
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject, transform_bounds
from shapely.geometry import box

CRS = "EPSG:6414"
log = logging.getLogger("clip_naip")


def scene_footprints(scenes: list[Path], crs: str = CRS) -> gpd.GeoDataFrame:
    """Bounding boxes of the scenes in ``crs`` (reads headers only)."""
    rows = []
    for path in scenes:
        with rasterio.open(path) as src:
            if src.count < 4:
                log.warning("skipped %s: %d bands (need R, G, B, NIR)", path.name, src.count)
                continue
            rows.append({"path": str(path), "geometry": box(*transform_bounds(src.crs, crs, *src.bounds))})
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=crs)


def clip_tile(tile_geom, scene_paths: list[str], pixel_size: float = 0.6,
              resampling: Resampling = Resampling.nearest, crs: str = CRS):
    """Warp and mosaic scenes onto the grid of one square tile.

    Returns (image (4, H, W) uint8, transform).
    """
    xmin, ymin, xmax, ymax = tile_geom.bounds
    width, height = round((xmax - xmin) / pixel_size), round((ymax - ymin) / pixel_size)
    transform = from_origin(xmin, ymax, pixel_size, pixel_size)
    mosaic = np.zeros((4, height, width), dtype=np.uint8)
    for path in scene_paths:
        part = np.zeros_like(mosaic)
        with rasterio.open(path) as src:
            reproject(rasterio.band(src, [1, 2, 3, 4]), part, dst_transform=transform,
                      dst_crs=crs, dst_nodata=0, resampling=resampling)
        fill = part.any(axis=0) & ~mosaic.any(axis=0)
        mosaic[:, fill] = part[:, fill]
        if mosaic.any(axis=0).all():
            break
    return mosaic, transform


def write_tile(img: np.ndarray, transform, out: Path, crs: str = CRS) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out, "w", driver="GTiff", width=img.shape[2], height=img.shape[1],
                       count=4, dtype="uint8", crs=crs, transform=transform,
                       compress="deflate") as dst:
        dst.write(img)
        dst.descriptions = ("red", "green", "blue", "nir")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--tiles", required=True, help="tile polygons (from make_tiles.py)")
    p.add_argument("--id-field", default="tile_id", help="tile id column (default tile_id)")
    p.add_argument("--scenes", required=True, nargs="+",
                   help="NAIP scene GeoTIFFs or folders searched recursively for *.tif")
    p.add_argument("--year", required=True, help="NAIP year, used in the output names")
    p.add_argument("--outdir", required=True, help="output folder")
    p.add_argument("--pixel-size", type=float, default=0.6, help="output pixel size, m (default 0.6)")
    p.add_argument("--resampling", default="nearest", choices=[r.name for r in Resampling],
                   help="resampling method (default nearest, as in the paper's pipeline)")
    p.add_argument("--overwrite", action="store_true", help="overwrite existing tiles")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    paths = []
    for s in map(Path, args.scenes):
        paths += sorted(s.rglob("*.tif")) if s.is_dir() else [s]
    scenes = scene_footprints(sorted(set(paths)))
    tiles = gpd.read_file(args.tiles).to_crs(CRS)
    log.info("%d tiles, %d NAIP scenes", len(tiles), len(scenes))

    hits = gpd.sjoin(tiles[[args.id_field, "geometry"]], scenes, how="inner", predicate="intersects")
    n_done = 0
    for tile_id, group in hits.groupby(args.id_field, sort=False):
        out = Path(args.outdir) / f"ID_{tile_id}_NAIP_{args.year}.tif"
        if out.exists() and not args.overwrite:
            continue
        img, transform = clip_tile(group.geometry.iloc[0], sorted(group["path"]),
                                   args.pixel_size, Resampling[args.resampling])
        empty = 1 - img.any(axis=0).mean()
        if empty == 1:
            log.warning("%s: no NAIP data, skipped", tile_id)
            continue
        if empty > 0:
            log.warning("%s: %.1f%% of pixels without NAIP data", tile_id, 100 * empty)
        write_tile(img, transform, out)
        n_done += 1
    log.info("wrote %d tiles to %s (%d tiles had no overlapping scene)", n_done, args.outdir,
             len(tiles) - hits[args.id_field].nunique())


if __name__ == "__main__":
    main()
