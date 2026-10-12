#!/usr/bin/env python
# Copyright (C) 2023-2026 Yunzhe Zhu
# SPDX-License-Identifier: GPL-3.0-only
# Part of fuel-mapping-CA (https://github.com/YunzheZhu/fuel-mapping-CA).
"""Make the 307.2 m labelling/training tile grid (512 px at 0.6 m, EPSG:6414).

Each sampled grid cell (1 km x 1 km for the paper's wildland and validation
tiles) is split into an n x n block of square tiles centred on the cell
centroid. With the
defaults (n = 3, 512 px x 0.6 m = 307.2 m) the block is 921.6 m wide and the
nine tiles are named by position:

    NW NN NE
    WW CE EE
    SW SS SE

(n = 2 gives NW, NE, SW, SE; other n give r<row>c<col>). Tile ids are
``<cell_id>_<position>``; the NAIP and label tiles of the pipeline are named
``ID_<tile_id>_NAIP_<year>.tif``. Cell attributes are copied to every tile.

Optionally the cells themselves (1 km by default) are created first as a
fishnet over an area of interest (``--aoi``). The statewide fishnet used in the paper was made
in a GIS; its origin is not reproduced here, so use the published tile index to
recover the exact paper tiles.

Examples::

    # split existing sampled cells (id column "id")
    python make_tiles.py --cells sampled_cells.gpkg --id-field id --out tiles.gpkg
    # fishnet over an AOI, then split every cell
    python make_tiles.py --aoi county.gpkg --cell-size 1000 --out tiles.gpkg --cells-out cells.gpkg

Based on the original tile-splitting script, with the pixel size fixed to 0.6 m.
"""
from __future__ import annotations

import argparse
import logging

import geopandas as gpd
import numpy as np
import shapely

CRS = "EPSG:6414"  # NAD83(2011) / California Albers
COMPASS_3 = {(-1, 1): "NW", (0, 1): "NN", (1, 1): "NE",
             (-1, 0): "WW", (0, 0): "CE", (1, 0): "EE",
             (-1, -1): "SW", (0, -1): "SS", (1, -1): "SE"}
COMPASS_2 = {(-1, 1): "NW", (1, 1): "NE", (-1, -1): "SW", (1, -1): "SE"}
log = logging.getLogger("make_tiles")


def fishnet(aoi: gpd.GeoDataFrame, cell_size: float = 1000.0, origin=None,
            crs: str = CRS) -> gpd.GeoDataFrame:
    """Square cells of ``cell_size`` m intersecting the AOI, ids row-major from the NW corner.

    ``origin`` (x, y) fixes the grid alignment; by default the lower-left AOI
    corner is snapped down to a multiple of ``cell_size``.
    """
    aoi = aoi.to_crs(crs)
    xmin, ymin, xmax, ymax = aoi.total_bounds
    ox, oy = origin if origin is not None else (np.floor(xmin / cell_size) * cell_size,
                                                np.floor(ymin / cell_size) * cell_size)
    xs = ox + cell_size * np.arange(np.floor((xmin - ox) / cell_size), np.ceil((xmax - ox) / cell_size))
    ys = oy + cell_size * np.arange(np.floor((ymin - oy) / cell_size), np.ceil((ymax - oy) / cell_size))
    gx, gy = np.meshgrid(xs, ys[::-1])  # rows from north to south
    cells = shapely.box(gx.ravel(), gy.ravel(), gx.ravel() + cell_size, gy.ravel() + cell_size)
    area = shapely.union_all(aoi.geometry.values)
    shapely.prepare(area)
    cells = cells[shapely.intersects(area, cells)]
    return gpd.GeoDataFrame({"cell_id": np.arange(len(cells))}, geometry=cells, crs=crs)


def tile_offsets(n: int) -> list[tuple[str, float, float]]:
    """(name, dx, dy) of the n x n tiles, in tile widths from the cell centroid."""
    out = []
    for row in range(n):              # north to south
        for col in range(n):          # west to east
            dx, dy = col - (n - 1) / 2, (n - 1) / 2 - row
            key = (int(np.sign(dx)), int(np.sign(dy)))
            name = COMPASS_3[key] if n == 3 else COMPASS_2[key] if n == 2 else f"r{row}c{col}"
            out.append((name, dx, dy))
    return out


def split_cells(cells: gpd.GeoDataFrame, id_field: str, n: int = 3, pixel_size: float = 0.6,
                tile_pixels: int = 512, keep_partial: bool = False,
                crs: str = CRS) -> gpd.GeoDataFrame:
    """Split each cell into n x n tiles of ``tile_pixels * pixel_size`` metres.

    Tiles must lie inside their cell unless ``keep_partial`` is set (the full
    square is kept either way, so every tile is exactly tile_pixels wide).
    """
    cells = cells.to_crs(crs)
    size = pixel_size * tile_pixels
    rows = []
    for _, cell in cells.iterrows():
        c = cell.geometry.centroid
        for name, dx, dy in tile_offsets(n):
            x0, y0 = c.x + dx * size - size / 2, c.y + dy * size - size / 2
            tile = shapely.box(x0, y0, x0 + size, y0 + size)
            if not keep_partial and not cell.geometry.buffer(1e-6).contains(tile):
                continue
            attrs = cell.drop(labels="geometry").to_dict()
            cid = attrs.get(id_field)
            cid = int(cid) if isinstance(cid, float) and cid.is_integer() else cid
            attrs.update(cell_id=cid, position=name, tile_id=f"{cid}_{name}", geometry=tile)
            rows.append(attrs)
    tiles = gpd.GeoDataFrame(rows, geometry="geometry", crs=crs)
    first = ["tile_id", "cell_id", "position"]
    return tiles[first + [col for col in tiles.columns if col not in first]]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--cells", help="polygon file of (sampled) grid cells")
    src.add_argument("--aoi", help="area of interest; a fishnet of cells is created over it")
    p.add_argument("--id-field", default="cell_id", help="cell id column (default cell_id)")
    p.add_argument("--cell-size", type=float, default=1000.0, help="fishnet cell size, m (default 1000)")
    p.add_argument("--origin", type=float, nargs=2, metavar=("X", "Y"),
                   help="fishnet origin in EPSG:6414 (default: snapped AOI corner)")
    p.add_argument("--cells-out", help="also write the fishnet cells here")
    p.add_argument("--n", type=int, default=3, help="tiles per cell side (default 3)")
    p.add_argument("--pixel-size", type=float, default=0.6, help="pixel size, m (default 0.6)")
    p.add_argument("--tile-pixels", type=int, default=512, help="tile width in pixels (default 512)")
    p.add_argument("--keep-partial", action="store_true", help="keep tiles not fully inside their cell")
    p.add_argument("--out", required=True, help="output tiles (.gpkg, .shp, ...)")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.aoi:
        cells = fishnet(gpd.read_file(args.aoi), args.cell_size, args.origin)
        args.id_field = "cell_id"
        if args.cells_out:
            cells.to_file(args.cells_out)
    else:
        cells = gpd.read_file(args.cells)
        if args.id_field not in cells.columns:
            p.error(f"--id-field {args.id_field!r} not in {list(cells.columns)}")
    tiles = split_cells(cells, args.id_field, args.n, args.pixel_size, args.tile_pixels,
                        args.keep_partial)
    tiles.to_file(args.out)
    log.info("%d cells -> %d tiles of %.1f m -> %s", len(cells), len(tiles),
             args.pixel_size * args.tile_pixels, args.out)


if __name__ == "__main__":
    main()
