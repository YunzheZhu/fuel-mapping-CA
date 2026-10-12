#!/usr/bin/env python
# Copyright (C) 2023-2026 Yunzhe Zhu
# SPDX-License-Identifier: GPL-3.0-only
# Part of fuel-mapping-CA (https://github.com/YunzheZhu/fuel-mapping-CA).
"""List and download USGS 3DEP LiDAR point clouds for tiles (public EPT, no credentials).

USGS 3DEP point clouds are served as Entwine Point Tiles (EPT) in the public
AWS bucket ``usgs-lidar-public``. Two sub-commands:

``list``
    Intersect tiles with the public EPT resource index and write a CSV of
    candidate surveys per tile (tile_id, dataset, points, ept_url). Edit it to
    keep the survey you want for each tile (the paper used 42 surveys from
    2015-2022 with > 10 pts/m2; acquisition dates and point density are in the
    USGS WESM metadata, https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/metadata/WESM.gpkg).

``download``
    For each (tile, dataset) pair, crop the EPT to the tile plus a buffer,
    drop withheld noise classes, reproject to EPSG:6414 and write
    ``ID_<tile_id>_<dataset>.laz`` with PDAL (``pdal`` command line tool, e.g.
    from conda-forge). Pairs come from a ``list`` CSV or from ``--dataset``.

Examples::

    python download_3dep_lidar.py list --tiles tiles.gpkg --id-field tile_id --out candidates.csv
    python download_3dep_lidar.py download --tiles tiles.gpkg --candidates candidates.csv --outdir laz/
    python download_3dep_lidar.py download --tiles tiles.gpkg --dataset CA_NorthernCA_1_B22 --outdir laz/

Based on the original 3DEP download scripts (PDAL EPT pipeline).
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import geopandas as gpd

INDEX_URL = "https://raw.githubusercontent.com/hobuinc/usgs-lidar/master/boundaries/resources.geojson"
EPT_URL = "https://s3-us-west-2.amazonaws.com/usgs-lidar-public/{name}/ept.json"
OUT_CRS = "EPSG:6414"
DROP_CLASSES = (129, 135)  # classification values excluded by the original pipeline
log = logging.getLogger("download_3dep")


def read_tiles(path: str, id_field: str) -> gpd.GeoDataFrame:
    tiles = gpd.read_file(path)
    if id_field not in tiles.columns:
        raise KeyError(f"--id-field {id_field!r} not in {list(tiles.columns)}")
    tiles["tile_id"] = tiles[id_field].astype(str)
    return tiles[["tile_id", "geometry"]].to_crs(OUT_CRS)  # metric CRS for the buffer


def list_surveys(tiles: gpd.GeoDataFrame, index: str = INDEX_URL,
                 name_filter: str | None = None) -> list[dict]:
    """EPT datasets whose boundary intersects each tile."""
    eps = gpd.read_file(index)[["name", "count", "url", "geometry"]]
    if name_filter:
        eps = eps[eps["name"].map(lambda n: re.search(name_filter, n) is not None)]
    hits = gpd.sjoin(tiles.to_crs(eps.crs), eps, how="inner", predicate="intersects")
    hits = hits.sort_values(["tile_id", "name"])
    return [{"tile_id": r.tile_id, "dataset": r["name"], "points": int(r["count"]),
             "ept_url": r["url"]} for _, r in hits.iterrows()]


def ept_pipeline(tile_geom, tile_crs, dataset: str, out_laz: str, buffer_m: float = 25.0,
                 drop_classes=DROP_CLASSES, ept_url: str | None = None) -> dict:
    """PDAL pipeline that crops one EPT dataset to a buffered tile and writes LAZ."""
    area = gpd.GeoSeries([tile_geom], crs=tile_crs).buffer(buffer_m).to_crs("EPSG:3857")
    xmin, ymin, xmax, ymax = area.total_bounds  # EPT data are stored in EPSG:3857
    stages = [{"type": "readers.ept", "filename": ept_url or EPT_URL.format(name=dataset),
               "bounds": f"([{xmin}, {xmax}], [{ymin}, {ymax}])"}]
    stages += [{"type": "filters.range", "limits": f"Classification![{c}:{c}]"} for c in drop_classes]
    stages += [{"type": "filters.reprojection", "out_srs": OUT_CRS},
               {"type": "writers.las", "filename": out_laz, "compression": "laszip"}]
    return {"pipeline": stages}


def run_pipeline(pipeline: dict, pdal: str = "pdal") -> None:
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(pipeline, f)
    try:
        subprocess.run([pdal, "pipeline", f.name], check=True, capture_output=True, text=True)
    finally:
        Path(f.name).unlink(missing_ok=True)


def download(tiles: gpd.GeoDataFrame, pairs: list[dict], outdir: str, buffer_m: float = 25.0,
             workers: int = 4, dry_run: bool = False, pdal: str = "pdal") -> None:
    """Fetch every (tile_id, dataset) pair that is not on disk yet."""
    Path(outdir).mkdir(parents=True, exist_ok=True)
    geoms = dict(zip(tiles["tile_id"], tiles.geometry))

    def job(pair):
        tile_id, dataset = pair["tile_id"], pair["dataset"]
        out = Path(outdir) / f"ID_{tile_id}_{dataset}.laz"
        if out.exists():
            return f"exists: {out.name}"
        if tile_id not in geoms:
            return f"unknown tile {tile_id}"
        pipe = ept_pipeline(geoms[tile_id], tiles.crs, dataset, str(out), buffer_m,
                            ept_url=pair.get("ept_url") or None)
        if dry_run:
            return json.dumps(pipe)
        try:
            run_pipeline(pipe, pdal)
            return f"wrote {out.name}"
        except subprocess.CalledProcessError as e:
            return f"FAILED {out.name}: {e.stderr.strip()[-300:]}"

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for msg in pool.map(job, pairs):
            log.info(msg)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("list", "download"):
        s = sub.add_parser(name)
        s.add_argument("--tiles", required=True, help="tile polygons (e.g. from make_tiles.py)")
        s.add_argument("--id-field", default="tile_id", help="tile id column (default tile_id)")
    ls = sub.choices["list"]
    ls.add_argument("--index", default=INDEX_URL, help="EPT resource index (URL or file)")
    ls.add_argument("--name-filter", help="regex on dataset names, e.g. '^(CA_|USGS_LPC_CA_)'")
    ls.add_argument("--out", required=True, help="output CSV of candidate surveys")
    dl = sub.choices["download"]
    grp = dl.add_mutually_exclusive_group(required=True)
    grp.add_argument("--candidates", help="CSV with tile_id,dataset[,ept_url] (from 'list')")
    grp.add_argument("--dataset", help="one EPT dataset name used for all tiles")
    dl.add_argument("--outdir", required=True, help="folder for the LAZ files")
    dl.add_argument("--buffer", type=float, default=25.0, help="buffer around each tile, m (default 25)")
    dl.add_argument("--workers", type=int, default=4, help="parallel downloads (default 4)")
    dl.add_argument("--pdal", default="pdal", help="PDAL executable (default: pdal on PATH)")
    dl.add_argument("--dry-run", action="store_true", help="print the PDAL pipelines only")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    tiles = read_tiles(args.tiles, args.id_field)
    if args.cmd == "list":
        rows = list_surveys(tiles, args.index, args.name_filter)
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["tile_id", "dataset", "points", "ept_url"])
            w.writeheader()
            w.writerows(rows)
        n_tiles = len({r["tile_id"] for r in rows})
        log.info("%d tile/survey pairs for %d of %d tiles -> %s", len(rows), n_tiles, len(tiles), args.out)
        return

    if not args.dry_run and shutil.which(args.pdal) is None:
        p.error(f"PDAL executable {args.pdal!r} not found (install PDAL, e.g. conda install -c conda-forge pdal)")
    if args.candidates:
        with open(args.candidates, newline="") as f:
            pairs = list(csv.DictReader(f))
    else:
        pairs = [{"tile_id": t, "dataset": args.dataset} for t in tiles["tile_id"]]
    download(tiles, pairs, args.outdir, args.buffer, args.workers, args.dry_run, args.pdal)


if __name__ == "__main__":
    main()
