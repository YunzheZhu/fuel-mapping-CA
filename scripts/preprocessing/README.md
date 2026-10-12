# Preprocessing: tiles, LiDAR and NAIP

These scripts prepare the inputs of the labelling workflow in
[`../labeling/README.md`](../labeling/README.md), which lists the
requirements and a full quick start. All outputs are in EPSG:6414
(NAD83(2011) / California Albers).

| Script | What it does |
|---|---|
| [`make_tiles.py`](make_tiles.py) | Splits each sampled grid cell (1 km cells for the paper's wildland and validation tiles) into 3 x 3 tiles of 307.2 m (512 px at 0.6 m), centred on the cell centroid and named `<cell_id>_<NW, NN, NE, WW, CE, EE, SW, SS, SE>`; `--n 2` gives the quadrants NW, NE, SW, SE. With `--aoi` it first builds a fishnet (1 km cells by default) over an area of interest. |
| [`download_3dep_lidar.py`](download_3dep_lidar.py) | `list` intersects tiles with the public USGS 3DEP EPT index (one row per tile and survey). `download` uses PDAL to crop each survey to the tile plus a 25 m buffer, drops classification values 129 and 135 (as the original pipeline did), reprojects to EPSG:6414 and writes LAZ. |
| [`clip_naip_tiles.py`](clip_naip_tiles.py) | Warps local NAIP scenes (bands R, G, B, NIR) onto each tile's 512 x 512 grid at 0.6 m with nearest-neighbour resampling, mosaicking where scenes overlap. Output is `ID_<tile>_NAIP_<year>.tif`, uint8. |

Checks:

- `make_tiles.py` reproduces the 4,050 tiles of the paper's 450 sampled
  wildland cells exactly (same tile ids, corner coordinates within 1e-10 m).
- `download_3dep_lidar.py` found and downloaded the 3DEP survey of a test tile
  from the public bucket.
- `clip_naip_tiles.py` output equals nearest-neighbour sampling of the source
  scene at the tile's pixel centres (99.98 % of pixels; the rest come from
  GDAL's approximate transformer). The original pipeline reprojected each
  scene to 0.6 m first and then cut the tile window. That two-step resampling
  can shift individual pixels by up to one source pixel, so tiles made with
  this script are not byte-identical to the published tiles.

## Getting the data

- **USGS 3DEP LiDAR.** Public domain, read anonymously from the AWS bucket
  `usgs-lidar-public` (EPT format). Survey dates and point densities are in the
  USGS Work Unit Extent Spatial Metadata (WESM):
  <https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/metadata/WESM.gpkg>.
  The paper used 42 surveys from 2015-2022 with more than 10 points/m2.
- **NAIP.** Public domain. The paper downloaded the 2016, 2018, 2020 and 2022
  California DOQQs (4 bands, 0.6 m) through the USGS Machine-to-Machine (M2M)
  API (<https://m2m.cr.usgs.gov/>), which requires a free USGS account. The
  same scenes are on USGS EarthExplorer, Microsoft Planetary Computer (`naip`
  collection) and Google Earth Engine (`USDA/NAIP/DOQQ`). No downloader is
  included here. If you script M2M access, read your username and token from
  environment variables and never commit them.
- **County building footprints** (used as building labels; Napa, San Luis
  Obispo and Shasta). See the links in [`../labeling/README.md`](../labeling/README.md).
