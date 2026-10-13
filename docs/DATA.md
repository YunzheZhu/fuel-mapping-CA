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
| Test, validation and training arrays | GitHub release `v1.0.0` | 11.1 GB | v1.0.0 |
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
| `fuelmap_unet_v1.json` | 5,655 | `4513b0474ab3273c3ba090800df2b56d131124e9ec5633781b409357a9b403fa` | Model metadata |
| `fuelmap_test_1y.npz` | 614,826,992 | `4a6bdad558b6fcf5805f0fb2120c970f446fd7f6ce7edb1a5a5dd937fb043e33` | 693 test tiles, one NAIP year each (the LiDAR-matched year) |
| `fuelmap_val_4y.npz` | 1,353,273,076 | `6f8425935538ca10743a4a85087500571ce05395af81e83eac831441fd4f9c10` | 1,512 validation tile-years (378 tiles × 4 NAIP years) |
| `fuelmap_train_4y_part01of06.npz` | 1,462,868,348 | `58408c29fe016fc7221cb67e34c3aa3b4128cb7fff3df7fbfd0ee127f3ddbbbf` | 1,703 training rows from source rows 0–1707 |
| `fuelmap_train_4y_part02of06.npz` | 1,471,268,193 | `85496cb89d3e7eb5a1a28b6007b31a26d1d2d280082924397f7bb5695a4cefcd` | 1,703 training rows from source rows 1708–3413 |
| `fuelmap_train_4y_part03of06.npz` | 1,544,610,038 | `9505369bda215c9206cf66060afde0d95e7665731a058db6907a34df26d17855` | 1,703 training rows from source rows 3414–5116 |
| `fuelmap_train_4y_part04of06.npz` | 1,545,147,747 | `b1069bd18ae050360f10c012d8f9cdc0dd57bf96612488217f88601a4b2dea10` | 1,703 training rows from source rows 5117–6819 |
| `fuelmap_train_4y_part05of06.npz` | 1,534,515,782 | `ae190a8c2d054bf0a77b657f6bdef599190bee921ac0b14e52e9a1e96f8b008f` | 1,703 training rows from source rows 6820–8522 |
| `fuelmap_train_4y_part06of06.npz` | 1,546,167,079 | `0b84cd3c1da6a3670eed076f3847d6778fae7357ea30c90d8b9dd41e67d2ffc8` | 1,703 training rows from source rows 8523–10968 |
| `SHA256SUMS.txt` | – | – | The checksums above, for `sha256sum -c SHA256SUMS.txt` |

The training parts total 9,104,577,187 bytes, and all assets total 11,196,921,396 bytes. The training parts hold
consecutive rows of the source array, without the rows that were left out (see
[Tiles left out of the arrays](#tiles-left-out-of-the-arrays)); the `row` member gives each row's source index.

### Array format

The arrays are compressed `.npz` archives (`np.savez_compressed`):

| Member | Shape, dtype | Content |
|---|---|---|
| `images` | (n, 512, 512, 4) `uint8` | NAIP digital numbers, bands R, G, B, NIR |
| `labels` | (n, 512, 512) `uint8` | 0 other, 1 tree, 2 shrub, 3 building, 255 = no label |
| `row` | (n,) `int32` | Row index in the original (2024) array |
| `tile_id`, `naip_year` | (n,) | Test array only; the same as `data/test_array_index.csv` (package position = column `index`, source row = column `row`) |

```python
import numpy as np

d = np.load("fuelmap_test_1y.npz")
x = d["images"][0].astype("float32") / 255   # (512, 512, 4): R, G, B, NIR
y = d["labels"][0]                           # 0-3, 255 = no label
print(d["tile_id"][0], d["naip_year"][0])
```

Things to know about the arrays:

- **What the arrays are.** They come from the arrays the release model was trained and evaluated with. Those were built
  in June 2024 from the paper's train/test split. Some counts differ from the numbers in the paper, and the public
  packages leave out the San Luis Obispo tiles (next section):

  | Array | 2024 array (release model) | Public package | Paper |
  |---|---:|---:|---:|
  | Training tiles | 6,701 | 5,950 | 7,015 |
  | Training tile-years (rows) | 10,969 | 10,218 | 10,086 |
  | Test tiles | 806 | 693 | 782 |
  | Validation tile-years (rows) | 1,512 | 1,512 | – |

  The 2024 training array has one row for each of 6,649 tiles at the tile's LiDAR-matched NAIP year (1,725 urban
  building-only, 1,434 WUI and 3,490 wildland tiles) plus 4,320 rows of WUI tiles at the other NAIP years. In all,
  1,486 WUI tiles have 2 to 4 NAIP years; 52 of them appear only at years other than their LiDAR-matched year. The
  array was cut before the split was finalised, so it holds fewer tiles than the 7,015 training tiles in
  `data/tile_index.gpkg`. The public training packages hold 10,218 of its rows: 974 urban building-only rows (Napa
  and Shasta counties), 5,754 WUI rows and 3,490 wildland rows. The column `in_release_arrays` of the tile index marks
  the tiles that have at least one row in the public packages.
- **Building-only tiles.** Urban-sample tiles from Napa and Shasta counties carry **building-only labels**: footprint
  pixels are 3 and every other pixel is 255. There are 162 such rows in the test package and 974 in the training
  packages, most of them in part 06. The paper's final evaluation used full four-class labels for these test tiles;
  those labels are not in the release. `URB_0014` in `data/sample/` is a building-only tile.
- **No-label pixels.** In the source arrays they had the value 4. The packages store them as 255.
- **Tile IDs.** Rows of the training and validation arrays do not carry tile IDs. The test array does.
- **Memory.** Compressed members are decompressed into memory when read. For training, convert each part to
  uncompressed single-member files first. `fuelmap` memory-maps uncompressed files, so DataLoader workers do not each
  hold a copy of the data:

  ```python
  import numpy as np
  for i in range(1, 7):
      d = np.load(f"fuelmap_train_4y_part{i:02d}of06.npz")
      np.savez(f"train_4y_part{i:02d}_images.npz", arr_0=d["images"])   # ~1.8 GB each, uncompressed
      np.savez(f"train_4y_part{i:02d}_labels.npz", arr_0=d["labels"])
  ```

### Tiles left out of the arrays

The building-only labels of the urban-sample tiles from San Luis Obispo County were rasterized from the county's
building-footprint layer. That layer's terms do not allow redistribution, so these tiles are not in the public arrays:

| Array | Rows left out | Tiles |
|---|---:|---|
| Test (`fuelmap_test_1y.npz`) | 113 | 113 urban-sample tiles (`footprint_county = San Luis Obispo`) |
| Training (`fuelmap_train_4y_part*.npz`) | 751 | 751 urban-sample tiles (`footprint_county = San Luis Obispo`) |
| Validation (`fuelmap_val_4y.npz`) | 0 | – |

- Their footprints and attributes stay in `data/tile_index.gpkg`, with `in_release_arrays = False`.
- The other released tiles inside San Luis Obispo County (189 validation tiles and 24 WUI training tiles) are
  included: their building labels were checked against the LiDAR segmentation and come from it, not from the
  footprint layer.
- The release model was trained (and its checkpoint selected) on the complete 2024 arrays, including the tiles left
  out here. Its accuracy on the public test package is given in the [model card](MODEL_CARD.md#re-evaluation-of-the-release-model).

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
sample tiles) and rasterised building masks from the Napa and Shasta county footprints (in the labels). No released
label is derived from the San Luis Obispo County footprints (see
[Tiles left out of the arrays](#tiles-left-out-of-the-arrays)). Terms and credits are in
[`LICENSE-DATA.md`](../LICENSE-DATA.md#third-party-sources).

| Data | Source | Used for |
|---|---|---|
| NAIP 4-band imagery, 0.6 m (2016, 2018, 2020, 2022) | USGS M2M API (used in the paper); also USGS EarthExplorer, Microsoft Planetary Computer (`naip`) and Google Earth Engine (`USDA/NAIP/DOQQ`) | model input; clips are in the released arrays and samples; see [`scripts/preprocessing/README.md`](../scripts/preprocessing/README.md) |
| USGS 3DEP aerial LiDAR (42 surveys, 2015–2022, > 10 pts/m²) | public AWS bucket `usgs-lidar-public` (EPT); metadata in [WESM](https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/metadata/WESM.gpkg) | labels |
| County building footprints (Napa, San Luis Obispo, Shasta) | county open-data portals (links in [`scripts/labeling/README.md`](../scripts/labeling/README.md)); the San Luis Obispo layer is an OpenStreetMap extract | building labels; labels derived from the San Luis Obispo layer are not in the release |
| CAL FIRE FRAP WUI 2015, EPA Level III ecoregions, Regional Resource Kit regions | CAL FIRE FRAP, U.S. EPA, California Wildfire & Forest Resilience Task Force | sampling strata and tile attributes |
| LANDFIRE EVC (30 m) | LANDFIRE | sampling attributes; comparison in the paper |
| Microsoft US Building Footprints | Microsoft | sampling attributes only; not used as labels and not redistributed |
