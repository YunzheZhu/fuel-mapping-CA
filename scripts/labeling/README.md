# Label generation (paper Sec. 2.4, Fig. 2)

These scripts implement the semi-automatic labelling workflow of
Zhu, Dixon & Jin (2026), *Remote Sensing of Environment* 333, 115091
(<https://doi.org/10.1016/j.rse.2025.115091>). LiDAR segmentation makes object
polygons with initial classes. A person checks them against NAIP imagery.
The checked polygons are then rasterized to 0.6 m, masked with NAIP NDVI and
stacked into training and testing arrays.

## Pipeline at a glance

| Stage | Paper Fig. 2 step | Script | Main output |
|---|---|---|---|
| Tile grid: 307.2 m tiles (512 px at 0.6 m), EPSG:6414 | Sec. 2.3 | [`../preprocessing/make_tiles.py`](../preprocessing/make_tiles.py) | `tiles.gpkg` |
| USGS 3DEP point clouds per tile | input to (1) | [`../preprocessing/download_3dep_lidar.py`](../preprocessing/download_3dep_lidar.py) | `ID_<tile>_<survey>.laz` |
| NAIP tiles (R, G, B, NIR) | input to (3), (4) | [`../preprocessing/clip_naip_tiles.py`](../preprocessing/clip_naip_tiles.py) | `ID_<tile>_NAIP_<year>.tif` |
| Automatic object segmentation from the point cloud | (1) | [`01_lidar_segmentation.R`](01_lidar_segmentation.R) | CHM, DTM, planar mask, object tops, object polygons |
| Initial labels from planar mask and height | (2) | [`01_lidar_segmentation.R`](01_lidar_segmentation.R) | `Clss_95` attribute of the polygons |
| Manual check and correction against same-year NAIP | (3) | by hand in a GIS (e.g. QGIS) | edited polygons |
| Rasterize to 0.6 m, NAIP NDVI masking | (4) | [`02_rasterize_labels.py`](02_rasterize_labels.py) | `ID_<tile>_NAIP_<year>_Label.tif` |
| Stack tiles into training/testing arrays | Sec. 2.4-2.5 | [`03_build_training_arrays.py`](03_build_training_arrays.py) | `*_images.npz`, `*_labels.npz`, `*_tiles.csv` |

Label values used by all scripts and by the released model:

| Value | Class |
|---|---|
| 0 | other (background) |
| 1 | tree |
| 2 | shrub |
| 3 | building |
| 255 | nodata / not labelled; ignored in training and evaluation |

## Requirements

These tools are separate from the `fuelmap` model environment:

- **Python** >= 3.9 with numpy, geopandas (>= 0.12), shapely (>= 2.0), rasterio (>= 1.3), pyproj.
  Tested with Python 3.12.3, geopandas 0.14.3, shapely 2.1.2, rasterio 1.3.9 and numpy 1.26.4.
- **R** >= 4.1 with lidR (>= 4.0), terra, sf, exactextractr and data.table, installed with
  `install.packages(c("lidR", "terra", "sf", "exactextractr", "data.table"))`.
  Tested with R 4.6.0 and lidR 4.3.2.
- **PDAL** command-line tool, needed only for the LiDAR download. Install it with
  `conda install -c conda-forge pdal`. Tested with PDAL 2.4.2.

No step needs credentials. The 3DEP point clouds come from a public AWS bucket.
NAIP scenes must be downloaded separately (see
[`../preprocessing/README.md`](../preprocessing/README.md)).

## Quick start

```bash
# 0. tiles, LiDAR and NAIP
python scripts/preprocessing/make_tiles.py --cells sampled_cells.gpkg --id-field cell_id --out tiles.gpkg
python scripts/preprocessing/download_3dep_lidar.py list --tiles tiles.gpkg --out lidar.csv
#    edit lidar.csv so that it keeps one survey per tile, then:
python scripts/preprocessing/download_3dep_lidar.py download --tiles tiles.gpkg --candidates lidar.csv --outdir laz/
python scripts/preprocessing/clip_naip_tiles.py --tiles tiles.gpkg --scenes naip/2018/ --year 2018 --outdir tiles/

# 1-2. LiDAR segmentation and initial labels
Rscript scripts/labeling/01_lidar_segmentation.R --input laz/ --outdir seg/ --threads 4

# 3. check and edit seg/*_crowns.gpkg over the NAIP tiles in a GIS (see below)

# 4. rasterize (one tile; use --manifest for many)
python scripts/labeling/02_rasterize_labels.py \
    --naip tiles/ID_123_CE_NAIP_2018.tif \
    --crowns seg/ID_123_CE_<survey>_crowns.gpkg \
    --chm seg/ID_123_CE_<survey>_chm.tif \
    --out tiles/ID_123_CE_NAIP_2018_Label.tif

# 5. arrays
python scripts/labeling/03_build_training_arrays.py --naip-dir tiles/ --split train --out arrays/train
```

## Steps in detail

### 1-2. LiDAR object segmentation and initial labels: `01_lidar_segmentation.R`

The script uses lidR (Roussel et al., 2020) and follows the object delineation
approach the paper builds on (Dixon et al., 2023; Dalponte & Coomes, 2016;
Coomes et al., 2017). It runs on one LAZ file or a folder of them. Every
parameter can be changed on the command line (`--th_seed 0.45`, ...).

| Step | Paper (Sec. 2.4) | Script default |
|---|---|---|
| Noise removal | drop points > 3 SD from the mean distance to their 10 nearest neighbours | `classify_noise(sor(k = 10, m = 3))`. No elevation filter by default, so ground below sea level is kept. `--drop_z_below 0` repeats the absolute-elevation filter of the 2023–24 code; `--min_z_agl -0.5` drops points more than 0.5 m below ground after normalisation |
| Terrain | IDW DTM smoothed with a 3x3 median | `rasterize_terrain(knnidw(k = 10))` at 0.6 m with a 3x3 median, written as `_dtm.tif`. Heights are normalised with `normalize_height(knnidw())`, as in every preserved code version. `--normalize dtm` subtracts the smoothed DTM instead, which follows the paper's wording |
| CHM | 0.6 m | `rasterize_canopy(p2r(subcircle = 0.2))` at 0.6 m with a 3x3 median |
| Object tops | local maximum filter with an adaptive window | `lmf(ws = f(height), hmin = 0.5)`. The window is 2 m for objects under 2 m and grows smoothly to 10 m for objects of 30 m or more |
| Region growing | seed pixels > 40 % of the top height; keep pixels > 50 % of the region's mean height | `dalponte2016(th_seed = 0.4, th_cr = 0.5, th_tree = 0.5, max_cr = 40)` |
| Object polygons | convex hull of the first-return points | `crown_metrics(geom = "convex")` on first returns. `--hull all` uses all points, as the 2024 code did. Polygons < 0.36 m2 (one pixel) are dropped |
| Planar mask | planar-region detection (Limberger & Oliveira, 2015) | `segment_shapes(shp_plane(k = 25))`. A pixel is planar if more than 95 % of its points are planar; 3x3 median |
| Initial class `Clss_95` | objects lower than 1 m are other; of the taller objects, planar cover >= 50 % gives building, and the rest are trees if the 95th-percentile height is above 5 m, else shrubs | zq95 < 1 m gives **0** (planar or not); otherwise planar >= 0.5 gives **3**, zq95 >= 5 m gives **1**, and 1 m <= zq95 < 5 m gives **2** |

The polygons carry these attributes: `treeID`, `Z` (top height), `zq95`,
`npoints`, `area`, `planar` (mean planar-mask value) and `Clss_95`.

> **Provenance.** This script is a cleaned **reference re-implementation**.
> The original 2023-24 script that produced the paper's polygons was not
> preserved verbatim. The workflow was rebuilt from later copies of the same
> code and from 2024 R session histories, and its parameters were aligned with
> the paper text.
>
> We ran it on the original 2024 LiDAR file of one tile (ID 067789_00, San Luis
> Obispo County) and compared the results with the 2024 outputs:
>
> - The CHM and the planar mask were practically identical: the CHM differed
>   by less than 1e-6 m on average, and the planar masks agreed on 99.999 % of
>   pixels.
> - Object tops and polygons were not identical. The 2024 run found fewer
>   local maxima than the script (2,670 vs 4,466 tops); its exact window
>   settings and lidR version could not be recovered.
> - The rasterized initial classes agreed on 94.0 % of tile pixels (94.5 % with
>   `--hull all`).
>
> The manual check in step 3 is part of the method, so treat the output as a
> starting point and not as a final label.

### 3. Manual check (by hand)

Overlay the polygons on the NAIP tile from the year closest to the LiDAR
survey. Use a true-colour composite (R, G, B) and a false-colour composite
(NIR, R, G). Then correct the polygons:

- fix `Clss_95` where it is wrong (0 other, 1 tree, 2 shrub, 3 building);
- delete spurious polygons;
- reshape polygons or draw missing ones.

When `02_rasterize_labels.py` runs, it re-checks tree vs shrub against `zq95`
for every polygon that has a height. Hand-drawn polygons without `zq95` keep
the class you give them.

### 4. Rasterization and NDVI masking: `02_rasterize_labels.py`

The output label is written on exactly the pixel grid of the NAIP tile, as
uint8 with nodata = 255. The rules are ported from the original label builders:

- **Tree vs shrub.** Woody polygons with `zq95 >= 5 m` are trees; the rest are shrubs.
- **Overlaps.** Where polygons overlap, tree beats shrub, and shrub beats building.
- **NDVI mask.** Tree and shrub pixels whose NDVI is not above 0.11 become other.
  NDVI = (NIR - R) / (NIR + R) on DN/255. The mask is applied before overlaps
  are resolved, as in the original code. Buildings are never NDVI-masked.
- **County footprints** (`--buildings`). These are authoritative for buildings.
  Every pixel inside a footprint becomes building. LiDAR "building" polygons
  outside the footprints are re-assigned to tree or shrub by height and then
  NDVI-masked, which is how the paper's footprint update was implemented.
  `--keep-lidar-buildings` disables the re-assignment. Only the footprint
  geometry is used.
- **Footprints only** (`--buildings-only`). Pixels inside footprints become
  building and every other pixel becomes 255. Some urban tiles in the arrays
  used for the released model were labelled this way.
- **No data.** Pixels without NAIP data become 255. With `--chm`, pixels the
  LiDAR does not cover (no CHM value) also become 255.
- **Other NAIP years.** To label another NAIP year, rasterize the same checked
  polygons onto that year's NAIP tile. The NDVI mask then follows that year's
  vegetation (Sec. 2.4, temporal augmentation).

The paper used county building footprints for Napa (Napa County GIS, 2016),
San Luis Obispo (County of San Luis Obispo, 2015) and Shasta (Shasta County,
2018):

- Napa: <https://www.arcgis.com/home/item.html?id=53c8a567d0e646f8ab4789e5d601eeaf>
- San Luis Obispo: <https://opendata-slocounty.hub.arcgis.com/datasets/SLOCounty::building-footprints/about>
  (a snapshot extract of OpenStreetMap)
- Shasta: <https://data-shasta.opendata.arcgis.com/datasets/Shasta::buildingfootprints/about>

Each layer has its own terms of use; see [`LICENSE-DATA.md`](../../LICENSE-DATA.md#third-party-sources).
The San Luis Obispo layer may not be redistributed, so the tiles labelled from
it are not in the release arrays.

The Microsoft building footprints were used only to compute sampling
attributes, never as labels.

**Checks against the 2024 labels of tile 067789_00**

- Rasterizing the 2024 checked, footprint-merged polygons reproduces the 2024
  label tiles **pixel for pixel for all four NAIP years** (2016, 2018, 2020
  and 2022).
- Rasterizing the 2024 un-merged polygons with `--buildings` differs by at
  most 10 pixels per tile.
- `--buildings-only` reproduces this tile's label in the 2024 test array
  exactly. The legacy no-label value 4 is now 255. (This San Luis Obispo
  County tile is not in the public test package.)
- Running the whole chain (public download, `01`, `02` with footprints) gives
  97.2 % pixel agreement with the 2024 checked label of 2016.

### 5. Training and testing arrays: `03_build_training_arrays.py`

The script stacks (NAIP tile, label tile) pairs into these files:

- `<out>_images.npz`: (N, 512, 512, 4) uint8, bands R, G, B, NIR;
- `<out>_labels.npz`: (N, 512, 512) uint8, values 0-3 and 255;
- `<out>_tiles.csv`: the tile id, NAIP year, split, source files and
  per-class pixel counts for each array index.

The original 2023-24 arrays did not record which tile each index came from;
arrays built with this script do. The npz files use the same layout as the
arrays the released model was trained on (uncompressed, key `arr_0`), so
`fuelmap.dataset` reads them directly. Label values outside 0-3, for example
the legacy no-label values 4 and 5, are written as 255.

Pairs come from a CSV manifest (`naip,label[,tile_id,year,split]`) or from a
folder of `ID_<tile>_NAIP_<year>.tif` files with matching `..._Label.tif`
files. You choose the rows, which sets what kind of array you get:

- **matched-year array:** one row per tile, using the NAIP year that matches the LiDAR survey;
- **all-years array:** one row per tile and NAIP year.

The paper augmented the WUI and testing tiles to all four NAIP years.

Images written by this script are byte-identical to the corresponding entry
of the 2024 test array (checked for the 2016 image of tile 067789_00).

## References

- Zhu, Y., Dixon, D. J., & Jin, Y. (2026). Scalable sub-meter mapping of woody vegetation and structures across California's heterogeneous landscape using deep learning. *Remote Sensing of Environment*, 333, 115091. <https://doi.org/10.1016/j.rse.2025.115091>
- Dixon, D. J., Zhu, Y., Brown, C. F., & Jin, Y. (2023). Satellite detection of canopy-scale tree mortality and survival from California wildfires with spatio-temporal deep learning. *Remote Sensing of Environment*, 298, 113842. <https://doi.org/10.1016/j.rse.2023.113842>
- Roussel, J.-R., et al. (2020). lidR: An R package for analysis of Airborne Laser Scanning (ALS) data. *Remote Sensing of Environment*, 251, 112061. <https://doi.org/10.1016/j.rse.2020.112061>
- Dalponte, M., & Coomes, D. A. (2016). Tree-centric mapping of forest carbon density from airborne laser scanning and hyperspectral data. *Methods in Ecology and Evolution*, 7(10), 1236-1245. <https://doi.org/10.1111/2041-210X.12575>
- Coomes, D. A., et al. (2017). Area-based vs tree-centric approaches to mapping forest carbon in Southeast Asian forests from airborne laser scanning data. *Remote Sensing of Environment*, 194, 77-88. <https://doi.org/10.1016/j.rse.2017.03.017>
- Limberger, F. A., & Oliveira, M. M. (2015). Real-time detection of planar regions in unorganized point clouds. *Pattern Recognition*, 48(6), 2043-2053. <https://doi.org/10.1016/j.patcog.2014.12.020>
