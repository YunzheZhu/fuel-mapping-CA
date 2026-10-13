# Data

The files in this folder are small and are kept in git. The image and label arrays are large, so they are attached to the
GitHub release [`v1.0.0`](https://github.com/YunzheZhu/fuel-mapping-CA/releases/tag/v1.0.0); see
[Release data packages](#release-data-packages).

Everything here and in the release packages is licensed **CC BY 4.0**. If you use it, please cite the paper
(doi:[10.1016/j.rse.2025.115091](https://doi.org/10.1016/j.rse.2025.115091)). The images are NAIP clips and some
building labels derive from county footprint layers; their terms and credits are listed in
[`LICENSE-DATA.md`](../LICENSE-DATA.md#third-party-sources). Tiles whose building labels derive from the San Luis
Obispo County footprint layer are not in the release packages, because that layer's terms do not allow
redistribution; the tile index keeps them and marks them with `in_release_arrays = False`.

| File | What it is |
|---|---|
| `tile_index.gpkg` | Footprints of all 8,256 labelled tiles (layer `tiles`, EPSG:6414), with split, sampling stratum and stratification attributes |
| `tile_index.csv` | The same table without geometry; tile centroids are in `centroid_x` and `centroid_y` (EPSG:6414, m) |
| `test_array_index.csv` | The tile and NAIP year of each of the 693 rows of the released one-year test array |
| `sample/` | 4 georeferenced NAIP and label GeoTIFF pairs from 4 test tiles, for the installation check ([docs/PIPELINE.md](../docs/PIPELINE.md#check-the-installation-on-the-sample-tiles); details [below](#sample)) |

## Conventions

| Item | Value |
|---|---|
| Classes | 0 = other (background), 1 = tree, 2 = shrub, 3 = building, **255 = nodata / no label** |
| Imagery | NAIP, 4 bands, `uint8` digital numbers, band order **R, G, B, NIR**; the model input is DN / 255 |
| Tile | 512 × 512 px at 0.6 m (307.2 m × 307.2 m) |
| CRS of the tile index | EPSG:6414 (NAD83(2011) / California Albers) |

Each image tile is clipped on the pixel grid of its source NAIP image, so its CRS is the source CRS. Where the
georeferencing was recovered, that is UTM (EPSG:26910 or 26911) for WUI tiles and EPSG:6414 for urban-sample tiles. The footprints in `tile_index.gpkg` are in
EPSG:6414 and match the image tiles to within the reprojection.

## `tile_index.gpkg` / `tile_index.csv`

There is one row per labelled tile.

| Split | Tiles | WUI | urban | wildland | validation |
|---|---:|---:|---:|---:|---:|
| train | 7,015 | 1,557 | 1,725 | 3,733 | – |
| test | 863 | 271 | 275 | 317 | – |
| validation | 378 | – | – | – | 378 |

| Column | Description |
|---|---|
| `tile_id` | Public tile identifier: `WUI_<cell>_<quadrant>`, `WLD_<cell>_<quadrant>`, `VAL_<cell>_<quadrant>` or `URB_<nnnn>` |
| `split` | `train`, `test` or `validation` (the train/test split used in the paper) |
| `stratum` | Sampling stratum: `WUI`, `urban` (building-footprint sample in Napa, San Luis Obispo and Shasta counties), `wildland`, or `validation` (378 tiles = 42 cells × 9 tiles) |
| `cell_id` | Grid-cell identifier as used in the original per-tile file names (empty for urban tiles) |
| `quadrant` | Tile position inside its cell: `NE`/`NW`/`SE`/`SW` (WUI) or one of the 3 × 3 positions `NW NN NE WW CE EE SW SS SE` (wildland, validation) |
| `urban_sample_no` | 1–2000, urban-sample number (urban tiles only) |
| `wui_class` | `Interface`, `Intermix`, `Urban`, `Wildland` or `Unknown`: the WUI class used to stratify the accuracy assessment. Train and test values are the final attributes used for the paper |
| `wui_group` | `WUI` (Interface + Intermix), `Urban`, `Wildland` or `Unknown` |
| `ecoregion_l3` | EPA Level III ecoregion (empty for 6 tiles) |
| `rrk_region` | California Regional Resource Kit region, as in the source layer: `Northern California Region`, `Central California Region`, `Sierra Nevada Region`, `Southern California Region`, `Limited mapping`, `Not Mapped` (empty for 6 tiles) |
| `label_naip_year` | NAIP year matched to the LiDAR acquisition and used for the labels and the one-year arrays |
| `label_scope` | Label scope in the released 2024 arrays: `full` (all four classes) or `building_only` (urban-sample tiles: building = 3, every other pixel = 255) |
| `footprint_county` | Urban tiles only: county building-footprint layer (Napa, San Luis Obispo, Shasta) |
| `in_release_arrays` | `True` if at least one row of the release packages (`fuelmap_test_1y.npz`, `fuelmap_val_4y.npz`, `fuelmap_train_4y_part*.npz`) comes from this tile; see [below](#in_release_arrays) |
| `lidar_workunit`, `lidar_project` | USGS 3DEP work unit and project of the LiDAR used for labelling (empty for urban tiles) |
| `lidar_start`, `lidar_end` | Collection start and end dates of the work unit (WUI and validation tiles) |
| `lidar_mid` | Collection mid date (wildland tiles) |
| `naip_2016` … `naip_2022` | NAIP scene (quarter-quad) covering the tile in each year; empty when none was recorded |
| `geometry` | Tile footprint, EPSG:6414 (GeoPackage only) |

`wui_class` counts (train / test / validation):

| Class | train | test | validation |
|---|---:|---:|---:|
| Interface | 839 | 163 | 5 |
| Intermix | 797 | 156 | 113 |
| Urban | 888 | 76 | 3 |
| Wildland | 4,491 | 458 | 257 |
| Unknown | 0 | 10 | 0 |

The paper reports 782 test tiles. The test split itself has 863 tiles (853 with a known WUI class). Not every test tile
has a valid NAIP–label pair in every year, so evaluation counts are lower than 863.

Attributes not needed to use the data were dropped. These include internal join fields, local file paths, and the
building-count and LANDFIRE attributes used only during sampling.

### `in_release_arrays`

| Split | `True` | `False` | Tiles with `False` |
|---|---:|---:|---|
| train | 5,950 | 1,065 | 751 urban-sample tiles from San Luis Obispo County (left out); 314 tiles that are not in the 2024 training array (71 WUI, 243 wildland) |
| test | 693 | 170 | 113 urban-sample tiles from San Luis Obispo County (left out); 57 tiles that are not in the 2024 test array (21 WUI, 36 wildland) |
| validation | 378 | 0 | – |

The urban-sample tiles from San Luis Obispo County were labelled from the county's building-footprint layer, whose terms
do not allow redistribution, so their rows were removed from the release packages. Their footprints and attributes are
kept here.

Rows of the training array carry no tile IDs, so their tiles were identified after the fact. Every one of the 10,969
rows of the 2024 training array was assigned to a tile:
- urban-sample rows (1,725): byte-identical image and building mask to the georeferenced urban-sample tile;
- WUI rows (5,754 rows of 1,486 tiles): byte-identical to the tile re-clipped from the per-cell NAIP image of that
  year;
- wildland rows (3,490 rows of 3,490 tiles): matched to the tiles' archived quality-check images. For the 3,437 tiles
  with their own image name and a non-empty image, the 32 × 32 thumbnail RMSE is ≤ 4.1 DN and at least 5 DN lower
  than for any other row. 26 pairs of tiles share an image name; each pair has two byte-identical rows in the array,
  and both tiles are marked `True`. 19 wildland rows (rows 7855–7872 and 8567 of the 2024 array) have all-zero images
  and labels.

## `test_array_index.csv`

The one-year test array of 2024 (806 rows) was saved without tile IDs. We recovered the tile behind every row. The
release package `fuelmap_test_1y.npz` holds 693 of these rows: the 113 urban-sample tiles from San Luis Obispo County
are left out (see [`in_release_arrays`](#in_release_arrays)). `test_array_index.csv` lists the 693 rows of the package,
in package order.

| `index_method` | Rows | Evidence |
|---|---:|---|
| `pixel_exact` | 412 | The row is byte-identical to the tile re-clipped from its georeferenced source image: 250 WUI tiles from per-cell NAIP clips, 162 urban tiles from per-tile GeoTIFFs. `crs` and `transform` give the exact georeferencing |
| `image_match` | 271 | A 32 × 32 thumbnail of the row matches the tile's archived quality-check image for the label year (RMSE < 8 DN). Where both methods applied (241 rows), this method agreed with `pixel_exact` every time |
| `location_match` | 10 | Shift-tolerant correlation identifies the tile location unambiguously (≥ 0.25 above any other tile), but no archived image of that year is identical. `naip_year` is set to `label_naip_year` |

All 693 rows map to distinct test-split tiles. In every row, `naip_year` equals `label_naip_year`.

The package contains:
- 250 WUI tiles, 281 wildland tiles and 162 urban tiles (56 from Napa County, 106 from Shasta County);
- the 162 urban tiles carry building labels only (`label_scope = building_only`).

Test-split tiles that are not in the package: the 113 urban-sample tiles from San Luis Obispo County, and 57 tiles
that are not in the 2024 test array (21 WUI, 36 wildland).

The paper's final evaluation used full four-class labels for the urban-sample test tiles; those labels are not in the
release.

| Column | Description |
|---|---|
| `index` | Position of the row in `fuelmap_test_1y.npz` (0–692). `scripts/evaluate.py` uses this position as the tile id of npz inputs, so pass `--id-column index` |
| `row` | Row in the original 2024 test array (0–805); the same as the `row` member of the package |
| `tile_id` | Tile in `tile_index.gpkg` |
| `naip_year` | NAIP year of the image in this row |
| `label_scope` | `full` or `building_only` (taken from the row's labels) |
| `stratum`, `wui_class`, `wui_group`, `ecoregion_l3`, `rrk_region`, `label_naip_year` | Copied from the tile index |
| `index_method` | `pixel_exact`, `image_match` or `location_match` (see above) |
| `crs`, `transform` | `pixel_exact` rows only: CRS and affine transform `a,b,c,d,e,f` (GDAL order: pixel width, row rotation, x origin, column rotation, pixel height, y origin) of the 512 × 512 image |

## `sample/`

These are GeoTIFFs for the installation check ([docs/PIPELINE.md](../docs/PIPELINE.md#check-the-installation-on-the-sample-tiles)), 2.9 MB in total, listed in `sample/sample_index.csv`. All four tiles are from the
test split, and their images and labels are the same as in the test package.

| Tile | NAIP year | WUI class | Region | Labels |
|---|---|---|---|---|
| `WUI_682706_NE` | 2016 | Interface | Southern California | all four classes (test package `index` 206, `row` 208) |
| `WUI_27384_NW` | 2018 | Intermix | Northern California | all four classes (`index` 225, `row` 227) |
| `WUI_33371_SE` | 2016 | Urban | Mojave Basin and Range | all four classes (`index` 137, `row` 138) |
| `URB_0014` | 2016 | Urban (urban sample, Napa County) | Northern California | building only: footprints are 3, every other pixel 255 (`index` 560, `row` 578) |

File layout:

| File pattern | Content |
|---|---|
| `<tile_id>_<year>_naip.tif` | 4-band `uint8` (R, G, B, NIR) |
| `<tile_id>_<year>_label.tif` | `uint8` labels 0–3, nodata 255, with an embedded colour table |

## Release data packages

These files are assets of the GitHub release `v1.0.0`:
`https://github.com/YunzheZhu/fuel-mapping-CA/releases/download/v1.0.0/<file>`.

Sizes and SHA-256 checksums are listed in [`docs/DATA.md`](../docs/DATA.md) and in `SHA256SUMS.txt` on the release.

All packages are `np.savez_compressed` archives with these arrays:
- `images`: `(n, 512, 512, 4)`, `uint8`, R, G, B, NIR
- `labels`: `(n, 512, 512)`, `uint8`, 0–3, with 255 = no label
- `row`: the row index in the original 2024 array

| File | Rows | Content |
|---|---:|---|
| `fuelmap_test_1y.npz` | 693 | One-year test array (one NAIP year per tile, the label year), without the 113 San Luis Obispo urban-sample tiles. Also contains `tile_id` and `naip_year` per row, the same as `test_array_index.csv` |
| `fuelmap_val_4y.npz` | 1,512 | Four-year validation array (378 validation tiles × 4 NAIP years) |
| `fuelmap_train_4y_partNNof06.npz` | 10,218 in total | Four-year training array that the release model was trained on, without the 751 San Luis Obispo urban-sample tiles, in 6 parts |

```python
import numpy as np

d = np.load("fuelmap_test_1y.npz")
x = d["images"][0].astype("float32") / 255   # (512, 512, 4), R, G, B, NIR
y = d["labels"][0]                           # 0 other, 1 tree, 2 shrub, 3 building, 255 no label
print(d["tile_id"][0], d["naip_year"][0])
```

The 2024 arrays were built in June 2024 from the same train/test split as the paper, before the split was finalised.
Their tile counts differ slightly from those in the paper. The four-year training array has 10,969 tile-years: 6,649
tiles at their LiDAR-matched NAIP year (1,725 urban building-only, 1,434 WUI and 3,490 wildland tiles), plus 4,320
rows of WUI tiles at the other NAIP years. It holds 6,701 of the 7,015 training tiles in `tile_index.gpkg`, because 52
WUI tiles appear only at years other than their LiDAR-matched year. The release packages hold 10,218 of its rows from
5,950 tiles (974 urban building-only, 1,486 WUI and 3,490 wildland). Rows of the training and validation arrays carry
no tile IDs.
