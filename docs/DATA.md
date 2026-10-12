# Data guide

This page explains where every data product of the project lives and how to read it. The column-by-column data
dictionary of the files in `data/` is in [`data/README.md`](../data/README.md).

All data, labels, model weights and map products are licensed **CC BY 4.0**
([`LICENSE-DATA.md`](../LICENSE-DATA.md), which also lists the third-party terms and credits); the code is GPL-3.0.

## Overview

| Product | Where | Size | Version |
|---|---|---|---|
| Tile index (8,256 labelled tiles, footprints and attributes) | repo: `data/tile_index.gpkg`, `data/tile_index.csv` | 5.0 MB + 2.7 MB | paper train/test split |
| Test-array index (tile and NAIP year of each test-array row) | repo: `data/test_array_index.csv` | 0.1 MB | v1.0.0 |
| Sample NAIP + label tiles | repo: `data/sample/` | 2.9 MB | v1.0.0 |
| Model weights + metadata | GitHub release `v1.0.0` (`models/fuelmap_unet_v1.json` also in repo) | 124.2 MB | v1.0.0 |
| Test, validation and training arrays | GitHub release `v1.0.0` | 11.8 GB | v1.0.0 |
| Statewide 0.6 m county class maps, 2016–2024 | [Box share](https://ucdavis.box.com/s/l3rssv9aiiigojzt0h7ql6gteqfdrogt) | 392.6 GB | **v2 (2026)** |
| Statewide 0.6 m maps used in the paper, 2016–2022 | from the authors on request | – | v1 (paper) |
| Interactive viewer | [Google Earth Engine app](https://fuel-type-naip.projects.earthengine.app/view/fuel-mapping) | – | – |

## Conventions

| Item | Value |
|---|---|
| Classes | 0 = other (background), 1 = tree, 2 = shrub, 3 = building; **255 = nodata / no label**. The v2 Box maps add **4 = uncertain** |
| Imagery | NAIP 4-band `uint8` digital numbers, band order **R, G, B, NIR**. The model input is DN / 255, with no other normalization |
| Tile | 512 × 512 px at 0.6 m = 307.2 m × 307.2 m |
| CRS | Tile index and v2 maps: EPSG:6414 (NAD83(2011) / California Albers). Each image tile keeps the grid of its source NAIP image (see `data/README.md`) |
| Tile IDs | `WUI_<cell>_<quadrant>`, `WLD_<cell>_<quadrant>`, `VAL_<cell>_<quadrant>`, `URB_<nnnn>`. `scripts/preprocessing/make_tiles.py` names tiles `<cell_id>_<quadrant>`. Drop the prefix (or use the `cell_id` and `quadrant` columns) to match the two |

## GitHub release `v1.0.0`

URL pattern: `https://github.com/YunzheZhu/fuel-mapping-CA/releases/download/v1.0.0/<file>`. Every file is under 2 GiB.

| File | Bytes | SHA-256 | Content |
|---|---:|---|---|
| `fuelmap_unet_v1.pth` | 124,238,486 | `35e9f387de0017e7cbddbdfc7fc8cbeb5645eb114a7fc03cdd8ab8acdde5706a` | Release model weights (plain float32 `state_dict`) |
| `fuelmap_unet_v1.json` | 5,086 | `24741fe81730f46f18f70aa9881ff22fbfd24e6f425ddff6be7ea6d6ec6215b2` | Model metadata |
| `fuelmap_test_1y.npz` | 714,705,083 | `ca20e9a58d828bba0d14b468b55e0e948d987e55bc690869abfe72684f381253` | 806 test tiles, one NAIP year each (the LiDAR-matched year) |
| `fuelmap_val_4y.npz` | 1,353,273,076 | `6f8425935538ca10743a4a85087500571ce05395af81e83eac831441fd4f9c10` | 1,512 validation tile-years (378 tiles × 4 NAIP years) |
| `fuelmap_train_4y_part01of06.npz` | 1,568,603,674 | `0a412971dbecd76262d78a13ea403623c0c884a6f818b1387b52d0a6d55190ae` | training rows 0–1827 |
| `fuelmap_train_4y_part02of06.npz` | 1,594,551,284 | `3e9131b6160b71d2e07f42f18cf711517be58040df7cab15372876e3432a234e` | training rows 1828–3655 |
| `fuelmap_train_4y_part03of06.npz` | 1,657,631,055 | `214af5444e840fe7a8b83fbe1a2c059b39943e0b0c60029c019cdc939f38c619` | training rows 3656–5483 |
| `fuelmap_train_4y_part04of06.npz` | 1,656,818,937 | `e9f3bd584a9b29bf4eabab6a49006597562aba325fb74cd876403d5e7da3df83` | training rows 5484–7312 |
| `fuelmap_train_4y_part05of06.npz` | 1,644,618,650 | `b7a819e6b04fded96797873fc03712d06595c927b19c7933cc02fd18d12ba9d3` | training rows 7313–9140 |
| `fuelmap_train_4y_part06of06.npz` | 1,645,584,144 | `96384afd288171b3a8306ed1d18b63e233dea8b3cc86434a0c2d9e9633cc594b` | training rows 9141–10968 |
| `SHA256SUMS.txt` | – | – | The checksums above, for `sha256sum -c SHA256SUMS.txt` |

The training parts total 9,767,807,744 bytes, and all assets total 11,960,029,392 bytes.

### Array format

The arrays are compressed `.npz` archives (`np.savez_compressed`):

| Member | Shape, dtype | Content |
|---|---|---|
| `images` | (n, 512, 512, 4) `uint8` | NAIP digital numbers, bands R, G, B, NIR |
| `labels` | (n, 512, 512) `uint8` | 0 other, 1 tree, 2 shrub, 3 building, 255 = no label |
| `row` | (n,) `int32` | Row index in the original (2024) array |
| `tile_id`, `naip_year` | (n,) | Test array only; the same as `data/test_array_index.csv` |

```python
import numpy as np

d = np.load("fuelmap_test_1y.npz")
x = d["images"][0].astype("float32") / 255   # (512, 512, 4): R, G, B, NIR
y = d["labels"][0]                           # 0-3, 255 = no label
print(d["tile_id"][0], d["naip_year"][0])
```

Things to know about the arrays:

- **What the arrays are.** They are the arrays the release model was trained and evaluated with. They were built in June
  2024 from the paper's train/test split. Some counts differ from the numbers in the paper:

  | Array | Here | Paper |
  |---|---:|---:|
  | Training tiles at the LiDAR-matched NAIP year | 6,649 | 7,015 |
  | Training tile-years | 10,969 | 10,086 |
  | Test tiles | 806 | 782 |

  The training array covers 6,649 of the 7,015 training tiles in `data/tile_index.gpkg` (1,725 urban building-only
  and 4,924 WUI or wildland tiles), each at its LiDAR-matched NAIP year, plus 4,320 rows of WUI tiles at the other
  NAIP years. It was cut before the split was finalised.
- **Building-only tiles.** Urban-sample tiles from Napa, San Luis Obispo and Shasta counties carry
  **building-only labels**: footprint pixels are 3 and every other pixel is 255. There are 275 such rows in the test
  array; in the training array they sit mostly in part 06. The paper's final evaluation used full four-class labels
  for these test tiles; those labels are not in the release. `URB_0014` in `data/sample/` is a building-only tile.
- **No-label pixels.** In the source arrays they had the value 4. The packages store them as 255.
- **Tile IDs.** Rows of the training and validation arrays do not carry tile IDs. The test array does.
- **Memory.** Compressed members are decompressed into memory when read. For training, convert each part to
  uncompressed single-member files first. `fuelmap` memory-maps uncompressed files, so DataLoader workers do not each
  hold a copy of the data:

  ```python
  import numpy as np
  for i in range(1, 7):
      d = np.load(f"fuelmap_train_4y_part{i:02d}of06.npz")
      np.savez(f"train_4y_part{i:02d}_images.npz", arr_0=d["images"])   # ~1.9 GB each, uncompressed
      np.savez(f"train_4y_part{i:02d}_labels.npz", arr_0=d["labels"])
  ```

## Statewide maps on Box (v2)

Share: https://ucdavis.box.com/s/l3rssv9aiiigojzt0h7ql6gteqfdrogt

```
NAIP_2016/  CA_Alameda_2016_class.zip … CA_Yuba_2016_class.zip   (58 counties, 75.8 GB)
NAIP_2018/  …                                                     (58 counties, 73.8 GB)
NAIP_2020/  …                                                     (58 counties, 90.9 GB)
NAIP_2022/  …                                                     (58 counties, 79.9 GB)
NAIP_2024/  …                                                     (58 counties, 72.2 GB)
```

- County names use underscores, as in `CA_San_Luis_Obispo_2022_class.zip`.
- Each zip holds one GeoTIFF, `CA_<County>_<YYYY>_class.tif`: single band, `uint8`, EPSG:6414, about 0.6 m pixels,
  tiled, ZSTD-compressed, nodata 255.
- Values: 0 other, 1 tree, 2 shrub, 3 building, **4 uncertain** (pixels the v2 post-processing flags as uncertain),
  255 nodata.
- **Version.** These are the improved **v2 (2026)** maps. A later fine-tuned model produced them, and it is not the
  `v1.0.0` release model. They are **not** the v1 maps analysed in the paper. The v1 maps (2016–2022) are available
  from the authors on request. The v2 model and its accuracy assessment will be released separately.

You can read a map without unzipping it. This needs GDAL (or rasterio) built with ZSTD support, as the standard wheels
and conda-forge packages are:

```bash
gdalinfo /vsizip/CA_Alpine_2022_class.zip/CA_Alpine_2022_class.tif
```

To compute fractional cover at 30 m, run `scripts/aggregate_30m.py`. It treats values outside 0–3, including the v2
class 4, as invalid pixels and reports their share in the `valid` band.

## Third-party inputs

The source layers below are not redistributed. Data derived from them are: NAIP clips (the `images` arrays and the
sample tiles) and rasterised building masks from the county footprints (in the labels). Terms and credits are in
[`LICENSE-DATA.md`](../LICENSE-DATA.md#third-party-sources).

| Data | Source | Used for |
|---|---|---|
| NAIP 4-band imagery, 0.6 m (2016, 2018, 2020, 2022) | USGS M2M API (used in the paper); also USGS EarthExplorer, Microsoft Planetary Computer (`naip`) and Google Earth Engine (`USDA/NAIP/DOQQ`) | model input; clips are in the released arrays and samples; see [`scripts/preprocessing/README.md`](../scripts/preprocessing/README.md) |
| USGS 3DEP aerial LiDAR (42 surveys, 2015–2022, > 10 pts/m²) | public AWS bucket `usgs-lidar-public` (EPT); metadata in [WESM](https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/metadata/WESM.gpkg) | labels |
| County building footprints (Napa, San Luis Obispo, Shasta) | county open-data portals (links in [`scripts/labeling/README.md`](../scripts/labeling/README.md)); the San Luis Obispo layer is an OpenStreetMap extract | building labels |
| CAL FIRE FRAP WUI 2015, EPA Level III ecoregions, Regional Resource Kit regions | CAL FIRE FRAP, U.S. EPA, California Wildfire & Forest Resilience Task Force | sampling strata and tile attributes |
| LANDFIRE EVC (30 m) | LANDFIRE | sampling attributes; comparison in the paper |
| Microsoft US Building Footprints | Microsoft | sampling attributes only; not used as labels and not redistributed |
