# Scalable Sub-meter Mapping of Woody Vegetation and Structures Across California

[![Paper](https://img.shields.io/badge/Paper-RSE_2026-blue)](https://doi.org/10.1016/j.rse.2025.115091)
[![Code license: GPL-3.0](https://img.shields.io/badge/Code-GPL--3.0-blue.svg)](LICENSE)
[![Data and weights: CC BY 4.0](https://img.shields.io/badge/Data_%26_weights-CC_BY_4.0-lightgrey.svg)](LICENSE-DATA.md)
[![Earth Engine app](https://img.shields.io/badge/Earth_Engine-Web_App-brightgreen)](https://fuel-type-naip.projects.earthengine.app/view/fuel-mapping)

This repository accompanies the paper:

> **Zhu, Y., Dixon, D. J., & Jin, Y. (2026).** Scalable sub-meter mapping of woody vegetation and structures across California's heterogeneous landscape using deep learning. *Remote Sensing of Environment*, 333, 115091. https://doi.org/10.1016/j.rse.2025.115091

| Resource | Link |
|---|---|
| Paper (open access, CC BY 4.0) | https://doi.org/10.1016/j.rse.2025.115091 |
| Interactive map viewer (Google Earth Engine) | https://fuel-type-naip.projects.earthengine.app/view/fuel-mapping |
| Model weights and training/validation/test arrays | GitHub release [`v1.0.0`](https://github.com/YunzheZhu/fuel-mapping-CA/releases/tag/v1.0.0) |
| Statewide 0.6 m county maps (v2, 2016–2024) | Box: https://ucdavis.box.com/s/l3rssv9aiiigojzt0h7ql6gteqfdrogt |

---

## Overview

This project uses a deep learning model (U-Net) to map **trees, shrubs and building footprints at 0.6 m** across
California from open-access NAIP aerial imagery (R, G, B, NIR). Coarse land-cover products such as LANDFIRE (30 m) miss
canopy-scale detail. Maps at this scale can support wildfire risk assessment, defensible-space evaluation,
habitat-fragmentation analysis and planning in the wildland–urban interface (WUI).

**Key contributions**

- A **semi-automatic labeling workflow**. It segments objects from USGS 3DEP aerial LiDAR (42 surveys, 2015–2022) and
  sets their initial classes. A person then checks the labels against NAIP imagery from the same year. The result is a
  large set of wall-to-wall labels across urban, WUI and wildland landscapes.
- A **single unified U-Net** that classifies tree, shrub, building and other (background) pixels across the
  urban–WUI–wildland continuum.
- **Statewide biennial maps (2016, 2018, 2020, 2022)** of woody vegetation and structures at 0.6 m, viewable in the
  Earth Engine app.
- Detection of **fine-scale dynamics**: post-fire mortality, regrowth, selective logging and clear-cutting, and
  rebuilding after wildfire.

**Case studies in the paper**

- Testing-tile examples (Fig. 4): urban, Paso Robles; WUI interface, north of Trinity Center; WUI intermix, north
  Alpine community; wildland, northeast Platina and a shrub-dominated ridge south of Los Gatos.
- Defensible space (Fig. 5), with 5 / 30 / 100 ft zones around buildings: north Nimshew (Butte County); northeast Santa
  Rosa (Sonoma County); south of Ojai (printed "Ojala" in the paper), Ventura County; north of Santa Barbara.
- Large-area check (Fig. 6): a 3,000 ha region in San Luis Obispo County.
- Disturbance plots (Fig. 7): 600 plots of 30 × 30 m near the 2018 Camp Fire. There are 100 plots each of minimal
  change, regrowth, clear-cut, selective logging, severe burn and delayed mortality.
- Wildfires: the 2021 Dixie Fire (Fig. 8) and rebuilding after the 2017 Tubbs Fire in Santa Rosa, 2016–2022 (Fig. 9).
- County and state scale: Sonoma County compared with LiDAR and LANDFIRE EVC (Fig. 10); statewide maps (Fig. 11).

---

## Repository layout

```
fuel-mapping-CA/
├── fuelmap/                    # Python package: U-Net, datasets, augmentation, loss, metrics, inference
├── scripts/
│   ├── preprocessing/          # tile grid, USGS 3DEP LiDAR download, NAIP clipping (README inside)
│   ├── labeling/               # LiDAR segmentation (R/lidR), rasterization + NDVI mask, arrays (README inside)
│   ├── train.py                # model training
│   ├── predict_tiles.py        # whole-tile prediction (512 x 512 px tiles, GeoTIFF folder or npz array)
│   ├── predict_naip.py         # sliding-window prediction on NAIP rasters of any size
│   ├── evaluate.py             # confusion matrix, OA, UA/PA, F1, IoU, stratified metrics
│   ├── mosaic_cog.py           # mosaic / reproject class maps into Cloud-Optimized GeoTIFFs
│   └── aggregate_30m.py        # fractional cover on a 30 m grid
├── configs/release_v1.yaml     # training and inference settings of the release model
├── models/                     # model metadata (fuelmap_unet_v1.json); the weights are a release asset
├── data/                       # tile index, test-array index, sample tiles (data dictionary inside)
├── docs/                       # MODEL_CARD.md, PIPELINE.md, DATA.md
├── tests/test_smoke.py         # smoke tests, no data needed
├── environment.yml, requirements.txt
└── CITATION.cff, LICENSE (GPL-3.0), LICENSE-DATA.md (CC BY 4.0)
```

---

## Installation

```bash
git clone https://github.com/YunzheZhu/fuel-mapping-CA.git
cd fuel-mapping-CA
conda env create -f environment.yml      # creates the env "fuelmap"
conda activate fuelmap
# or, in an existing environment:  pip install -r requirements.txt

python tests/test_smoke.py               # 9 tests, about 5 s on CPU (or: pytest tests)
```

- Requires Python ≥ 3.9 and PyTorch ≥ 2.0. Other dependencies: numpy, rasterio, PyYAML and tqdm. GDAL's Python
  bindings are needed only for `scripts/mosaic_cog.py`. For a CUDA build of PyTorch see
  https://pytorch.org/get-started/locally/.
- The scripts run from the repository root. You do not need to install the package.
- Tested with Python 3.12.3, torch 2.3.1 (CPU), numpy 1.26.4, rasterio 1.3.9 and GDAL 3.8.4, and with the minimum
  versions Python 3.10.9, torch 2.0.0, numpy 1.24.2 and rasterio 1.3.6 (identical predictions). numpy 2 has not been
  tested. The paper's training environment was Python 3.10.13 with torch 2.1.1.
- The labeling tools need more software: geopandas, shapely ≥ 2, pyproj, PDAL, and R with lidR. See
  [`scripts/labeling/README.md`](scripts/labeling/README.md).

---

## Quick start

**1. Download the release model** (124 MB) and check it:

```bash
curl -L -o models/fuelmap_unet_v1.pth \
  https://github.com/YunzheZhu/fuel-mapping-CA/releases/download/v1.0.0/fuelmap_unet_v1.pth
sha256sum models/fuelmap_unet_v1.pth   # macOS: shasum -a 256
# 35e9f387de0017e7cbddbdfc7fc8cbeb5645eb114a7fc03cdd8ab8acdde5706a
```

**2. Predict the sample tiles** in [`data/sample`](data/sample) (4 NAIP test tiles):

```bash
python scripts/predict_tiles.py --config configs/release_v1.yaml --weights models/fuelmap_unet_v1.pth \
    --images data/sample --pattern "*_naip.tif" --output outputs/sample_pred
```

**3. Evaluate against the sample labels:**

```bash
python scripts/evaluate.py --pred-dir outputs/sample_pred --label-dir data/sample \
    --stem-regex "^(.+)_naip$" --label-template "{stem}_label.tif" \
    --out-csv outputs/sample_metrics.csv --per-tile-csv outputs/sample_per_tile.csv
```

This should print OA 0.769 over the 4 tiles (per tile 0.68–0.98; the urban tile `URB_0014` has building labels only).
The sample is too small to show the model's accuracy; use it to check that your installation works. The
[accuracy section](#accuracy) lists the full test results.

**4. Map a NAIP scene** (GeoTIFF, COG or VRT of any size; bands R, G, B, NIR):

```bash
python scripts/predict_naip.py --config configs/release_v1.yaml --weights models/fuelmap_unet_v1.pth \
    --input m_3912127_ne_10_060_20180718.tif --output m_3912127_ne_10_060_20180718_class.tif
# --probs also writes per-class probabilities; --window COL ROW W H predicts a subset;
# a folder as --input (with --skip-existing) processes many scenes.
```

The output is a uint8 GeoTIFF on the input grid: 0 other, 1 tree, 2 shrub, 3 building, 255 nodata. It is tiled,
DEFLATE-compressed and has a colour table. The script slides 1,666 px windows with 25 % overlap. On a 12-thread CPU a
full NAIP quarter-quad (about 9,600 × 12,200 px) takes roughly 15 minutes.

**From Python:**

```python
import torch
from fuelmap import load_model, predict_batch
from fuelmap.dataset import read_raster

model = load_model("models/fuelmap_unet_v1.pth")                      # CPU, eval mode
img, profile = read_raster("data/sample/WUI_27384_NW_2018_naip.tif")   # (4, 512, 512) uint8: R, G, B, NIR
x = torch.from_numpy(img).float().div(255).unsqueeze(0)
with torch.no_grad():
    classes, probs = predict_batch(model, x)                          # paper decision rule
```

---

## Pipeline overview

[`docs/PIPELINE.md`](docs/PIPELINE.md) has the end-to-end commands for every stage.

| Stage | Paper | Script |
|---|---|---|
| Stratified sampling of grid cells / tiles (WUI type, ecoregion, urban/wildland) | §2.3, Fig. 1 | not scripted; the result is [`data/tile_index.gpkg`](data/README.md) |
| Tile grid: 307.2 m tiles (512 px at 0.6 m), EPSG:6414 | §2.3 | `scripts/preprocessing/make_tiles.py` |
| USGS 3DEP point clouds per tile | §2.2 | `scripts/preprocessing/download_3dep_lidar.py` |
| NAIP tiles (R, G, B, NIR) | §2.2 | `scripts/preprocessing/clip_naip_tiles.py` |
| LiDAR object segmentation; initial labels from planar mask and height | Fig. 2 (1)–(2) | `scripts/labeling/01_lidar_segmentation.R` |
| Manual check of the labels against same-year NAIP | Fig. 2 (3) | by hand in a GIS |
| Rasterize to 0.6 m, NAIP NDVI mask | Fig. 2 (4) | `scripts/labeling/02_rasterize_labels.py` |
| Training/testing arrays | §2.4–2.5 | `scripts/labeling/03_build_training_arrays.py` |
| U-Net training | §2.5, Fig. 3 | `scripts/train.py` + `configs/release_v1.yaml` |
| Prediction on test tiles | §2.6 | `scripts/predict_tiles.py` |
| Pixel-based accuracy (Tables 1–2) | §2.6–2.7 | `scripts/evaluate.py` |
| County / statewide prediction | §2.8, §3.5 | `scripts/predict_naip.py`, `scripts/mosaic_cog.py` |
| 30 m fractional cover | §2.7–2.8, Figs. 6, 10, 11 | `scripts/aggregate_30m.py` |

Labeling summary (Fig. 2):

1. LiDAR points are cleaned with a KNN noise filter (k = 10, 3 SD).
2. A 0.6 m canopy height model is made, and object tops are found with an adaptive local-maximum filter.
3. Objects are grown from the tops by region growing.
4. Initial classes:
   - objects under 1 m → **other**;
   - taller objects with planar cover ≥ 50 % → **building**;
   - otherwise, objects whose **95th-percentile height is ≥ 5 m → tree**;
   - the remaining objects (1–5 m) → **shrub**.
5. County building footprints (Napa, San Luis Obispo, Shasta) supply the building labels of the urban sample tiles,
   which are labelled for buildings only. In the WUI, wildland and validation tiles, buildings come from the LiDAR
   planar mask.
6. A person checks the labels against NAIP.
7. The labels are rasterized to 0.6 m. Tree and shrub pixels with NAIP **NDVI ≤ 0.11** become other.

The paper words the two thresholds as "above 5 m" and "below 0.11"; both the original and the public label code use
≥ 5 m and ≤ 0.11. The R script is a reference re-implementation, not the frozen 2023–24 script; see the provenance
note in [`scripts/labeling/README.md`](scripts/labeling/README.md).

---

## Model

Full details: [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md) and [`models/README.md`](models/README.md).

**The model.** `fuelmap_unet_v1.pth` is the **release model**. It was retrained in June 2024 on the paper's training
tiles, with the test tiles held out. On the public test tiles it matches the paper's reported overall accuracy
(87.1 %, as in the paper; see [Accuracy](#accuracy)). It is not the exact checkpoint behind the paper's tables.

| Item | Value |
|---|---|
| Architecture | U-Net (Ronneberger et al., 2015; milesial/Pytorch-UNet layout), 5 levels, 64–1024 kernels, transposed-convolution upsampling, 31,038,469 parameters |
| Input | NAIP 4-band uint8, band order R, G, B, NIR, divided by 255; trained on 512 × 512 px tiles at 0.6 m |
| Output | 5 logit channels: 0 other, 1 tree, 2 shrub, 3 building, 4 = legacy "unlabeled" channel (never used; dropped at inference) |
| Decision rule | The rule used for the paper's test-tile accuracy assessment; the paper text does not spell it out. Softmax over all outputs, then: tree if p_tree > 0.4 and NDVI > 0.11; shrub if p_shrub > 0.4 and NDVI > 0.11; building if p_building > 0.6; other if p_other > 0.5. Each later rule overrides the earlier ones, and all remaining pixels are other. Plain argmax is also available (`--rule argmax`) |
| Training data | 10,969 training tile-years: 6,649 tiles at their LiDAR-matched NAIP year (1,725 urban building-only + 4,924 WUI/wildland) plus 4,320 rows of WUI tiles at the other NAIP years (2016–2022). The array is the June 2024 cut of the training split, so it holds fewer tiles than the 7,015 training tiles in [`data/tile_index.gpkg`](data/README.md). The public training packages hold 10,218 of these rows (see [Data availability](#data-availability)) |
| Training | Weighted cross-entropy (other 0.4, tree 1.0, shrub 1.1, building 1.5), RMSprop, batch 20 (multi-GPU `nn.DataParallel`); flips, rotation and scale jitter. 50 epochs; epoch 47 kept. All settings are in [`configs/release_v1.yaml`](configs/release_v1.yaml) |
| File | plain `state_dict`, float32, 124,238,486 bytes, SHA-256 `35e9f387de0017e7cbddbdfc7fc8cbeb5645eb114a7fc03cdd8ab8acdde5706a` |

`fuelmap.model.load_model` accepts plain and `module.`-prefixed (DataParallel) state dicts. It works with 4- and
5-channel heads and infers the number of outputs from the weights.

---

## Accuracy

### As reported in the paper

These values are copied from the paper. The paper evaluated **782 testing tiles**. F1 is, to rounding, the unweighted
mean of the four class F1 scores (other included); the Wildland row is the exception (see the note below Table 1). The
paper's tables also give UA, PA and four Regional Resource Kit (RRK) regions; see the
[model card](docs/MODEL_CARD.md#results-reported-in-the-paper).

**Table 1: matched NAIP–LiDAR year, overall and by WUI stratum**

| Scenario | OA | mIoU | F1 | F1 Tree | F1 Shrub | F1 Building | F1 Other |
|---|---|---|---|---|---|---|---|
| **Overall (single NAIP year)** | **87.1%** | 70.1% | 82.1% | 83.1% | 78.9% | 75.1% | 91.4% |
| Urban | 83.2% | 52.1% | 63.4% | 73.5% | 14.2% | 77.4% | 88.7% |
| WUI interface | 84.8% | 59.4% | 72.5% | 79.3% | 45.1% | 75.5% | 90.0% |
| WUI intermix | 87.2% | 61.4% | 74.2% | 85.6% | 51.4% | 68.1% | 91.5% |
| Wildland | 88.6% | 69.0% | 80.1%¹ | 83.8% | 83.8%¹ | 62.7% | 92.5% |

¹ As printed in the paper. The four class F1 values of this row average to 80.7 %, not 80.1 %, so one value in the row
(possibly the repeated 83.8 %) is likely a typo in the paper.

**Table 2: stability across NAIP years** (testing tiles with labels carried to every NAIP year)

| NAIP year | OA | mIoU | F1 | F1 Tree | F1 Shrub | F1 Building | F1 Other |
|---|---|---|---|---|---|---|---|
| Single NAIP year | 87.1% | 70.1% | 82.1% | 83.1% | 78.9% | 75.1% | 91.4% |
| 2016 | 85.0% | 66.7% | 79.6% | 81.6% | 72.5% | 74.5% | 89.8% |
| 2018 | 87.5% | 70.5% | 82.3% | 83.3% | 80.0% | 74.3% | 91.8% |
| 2020 | 85.0% | 67.2% | 80.0% | 82.8% | 73.5% | 74.1% | 89.7% |
| 2022 | 88.1% | 69.2% | 81.3% | 81.8% | 77.8% | 72.5% | 93.0% |

**30 m fractional cover:**

- San Luis Obispo County region (3,000 ha), compared with the LiDAR reference (Fig. 6): tree R² = 0.86, shrub
  R² = 0.93, building R² = 0.71.
- Sonoma County, compared with 2013 LiDAR canopy cover (Fig. 10): the model's tree cover explains about 70 % of the
  variance (R² = 0.69). LANDFIRE 2016 EVC reaches R² = 0.35.

### Release model v1.0.0, re-evaluated

We re-evaluated `fuelmap_unet_v1.pth` with the scripts in this repository. The test set is the **693-tile public
one-year test array** (`fuelmap_test_1y.npz`), which has each tile at its LiDAR-matched NAIP year. There are 143.4
million labelled pixels; unlabelled pixels are ignored. The confusion matrix has the reference in rows and the
prediction in columns. **UA = user's accuracy = precision; PA = producer's accuracy = recall.**

| Decision rule | OA | mIoU | mean F1 | F1 Tree | F1 Shrub | F1 Building | F1 Other |
|---|---|---|---|---|---|---|---|
| **Paper rule** (thresholds + NDVI) | **87.1%** | 72.5% | 83.9% | 83.0% | 80.4% | 80.7% | 91.3% |
| Argmax | 85.9% | 70.4% | 82.5% | 82.9% | 78.8% | 77.9% | 90.3% |

| Paper rule, per class | Tree | Shrub | Building | Other |
|---|---|---|---|---|
| UA (precision) | 78.9% | 74.6% | 73.6% | 96.5% |
| PA (recall) | 87.7% | 87.2% | 89.3% | 86.7% |
| IoU | 71.0% | 67.2% | 67.7% | 84.1% |

How these numbers relate to the paper:

- **Different test array.** The June 2024 test array has 806 tiles and is not identical to the paper's 782-tile set.
  The public package holds 693 of them. The 113 urban-sample tiles from San Luis Obispo County are left out, because
  their building labels come from that county's building-footprint layer, whose terms do not allow redistribution.
- **Building-only tiles.** 162 of the 693 tiles are urban-sample tiles (Napa and Shasta counties) with
  **building-only labels**: every non-building pixel is unlabelled. These tiles add building pixels but no tree, shrub
  or other pixels. As a result, the overall building scores and mIoU are not directly comparable with Table 1. The
  paper's final evaluation used full labels for these tiles.
- **Checkpoint selection.** The training code kept epoch 47 because test-split Dice and mIoU had both improved. These
  test scores are therefore not fully independent of checkpoint selection.

To reproduce them, see [`docs/PIPELINE.md`](docs/PIPELINE.md#5-reproduce-the-release-model-accuracy). This takes about
10 minutes on a CPU.

<details>
<summary>Release model by WUI class (full-label tiles only, paper rule)</summary>

These results come from `scripts/evaluate.py --attributes data/test_array_index.csv`, with the 162 building-only tiles
excluded. The tile-to-row mapping of the test array was recovered after the fact: 683 of the 693 rows are matched with
high confidence (see [`data/README.md`](data/README.md#test_array_indexcsv)). These strata are smaller than the
paper's, so compare them with Table 1 only loosely.

| WUI class | Tiles | OA | mIoU | mean F1 | F1 Tree | F1 Shrub | F1 Building | F1 Other |
|---|---|---|---|---|---|---|---|---|
| All full-label tiles | 531 | 87.0% | 66.4% | 78.8% | 83.2% | 80.4% | 60.1% | 91.5% |
| Urban | 15 | 79.0% | 49.6% | 63.1% | 74.4% | 27.2% | 65.3% | 85.7% |
| WUI interface | 58 | 84.0% | 59.5% | 73.3% | 82.7% | 58.1% | 62.8% | 89.4% |
| WUI intermix | 86 | 87.9% | 59.1% | 71.7% | 87.3% | 56.8% | 50.6% | 92.1% |
| Wildland | 370 | 87.5% | 62.1% | 73.6% | 82.1% | 82.4% | 38.1% | 92.0% |

The total row includes 2 full-label tiles whose WUI class is unknown; they are in none of the class rows.

Building F1 is lower in the full-label tiles. They hold only about 2.0 M of the 6.2 M reference building pixels, so false
positives weigh heavily; most building-rich urban tiles are among the building-only tiles. The [model card](docs/MODEL_CARD.md#stratified-results-of-the-release-model)
also gives the results by RRK region.
</details>

---

## Data availability

Details: [`docs/DATA.md`](docs/DATA.md) (overview, checksums) and [`data/README.md`](data/README.md) (data dictionary).

**In this repository** (`data/`):

| File | Description |
|---|---|
| `tile_index.gpkg` | Footprints of all 8,256 labelled tiles (EPSG:6414): 7,015 train, 863 test and 378 validation. Columns include split, sampling stratum, WUI class, ecoregion, RRK region, LiDAR project, NAIP scenes and whether the tile is in the release arrays (`in_release_arrays`) |
| `tile_index.csv` | The same table without geometry |
| `test_array_index.csv` | Tile and NAIP year of each of the 693 rows of the public test array |
| `sample/` | 4 NAIP + label GeoTIFF pairs (2.9 MB) for the quick start |

**GitHub release [`v1.0.0`](https://github.com/YunzheZhu/fuel-mapping-CA/releases/tag/v1.0.0)** (11.20 GB in total).
Download URL: `https://github.com/YunzheZhu/fuel-mapping-CA/releases/download/v1.0.0/<file>`.

| File | Size | Content |
|---|---|---|
| `fuelmap_unet_v1.pth` | 124.2 MB | Release model weights |
| `fuelmap_unet_v1.json` | 5.7 kB | Model metadata (also in `models/`) |
| `fuelmap_test_1y.npz` | 614.8 MB | 693 test tiles at the LiDAR-matched NAIP year, with `tile_id` and `naip_year` |
| `fuelmap_val_4y.npz` | 1.35 GB | 1,512 validation tile-years (378 tiles × 4 NAIP years) |
| `fuelmap_train_4y_part01of06.npz` … `part06of06.npz` | 1.46–1.55 GB each (9.10 GB) | 10,218 training tile-years: the release model's training data without the San Luis Obispo urban-sample tiles (see below) |
| `SHA256SUMS.txt` | <1 kB | SHA-256 checksums of all assets |

The arrays are compressed `.npz` files with these members:
- `images`: (n, 512, 512, 4) uint8, bands R, G, B, NIR;
- `labels`: (n, 512, 512) uint8, values 0–3, with 255 = no label;
- `row`: the row index in the original array.

**Tiles left out of the arrays.** The building labels of the urban-sample tiles from San Luis Obispo County were
rasterized from the county's building-footprint layer, and that layer's terms do not allow redistribution. These tiles
(113 test and 751 training tiles) are therefore not in the public arrays. Their footprints and attributes stay in the
tile index, with `in_release_arrays = False`. The release model was trained on the complete arrays, so the public
training packages have 751 fewer rows than its training data.

**Statewide maps on Box (v2):** https://ucdavis.box.com/s/l3rssv9aiiigojzt0h7ql6gteqfdrogt

- Content: 0.6 m class maps for **2016, 2018, 2020, 2022 and 2024**, one zip per county and year
  (`NAIP_<YYYY>/CA_<County>_<YYYY>_class.zip`; 58 counties per year; 392.6 GB in total).
- Format: each zip holds one GeoTIFF, uint8, EPSG:6414, ZSTD-compressed, with nodata 255.
- Classes: 0–3 as below, plus **4 = uncertain**.
- **These are the improved v2 (2026) maps.** A later fine-tuned model produced them, and they are **not** the v1 maps
  used in the paper. The original v1 maps (2016–2022) are available from the authors on request.

---

## Class legend

| Value | Class | Colour in class GeoTIFFs written by this code |
|---|---|---|
| 0 | Other (background: bare ground, grass, crops, water, roads, …) | `#E6E6D2` |
| 1 | Tree (woody, 95th-percentile height ≥ 5 m) | `#186E28` |
| 2 | Shrub (woody, 1–5 m) | `#C8AA3C` |
| 3 | Building | `#DC2828` |
| 4 | Uncertain (**v2 Box maps only**) | – |
| 255 | Nodata (outside the NAIP footprint); "no label" in label files | – |

---

## Citation

If you use this code, the model or the data, please cite the paper:

```bibtex
@article{Zhu2026RSE,
  author    = {Zhu, Yunzhe and Dixon, Dan J. and Jin, Yufang},
  title     = {Scalable sub-meter mapping of woody vegetation and structures across {California}'s heterogeneous landscape using deep learning},
  journal   = {Remote Sensing of Environment},
  volume    = {333},
  pages     = {115091},
  year      = {2026},
  issn      = {0034-4257},
  doi       = {10.1016/j.rse.2025.115091},
  url       = {https://doi.org/10.1016/j.rse.2025.115091},
  publisher = {Elsevier}
}
```

APA: Zhu, Y., Dixon, D. J., & Jin, Y. (2026). Scalable sub-meter mapping of woody vegetation and structures across
California's heterogeneous landscape using deep learning. *Remote Sensing of Environment, 333*, Article 115091.
https://doi.org/10.1016/j.rse.2025.115091

GitHub's "Cite this repository" button reads [`CITATION.cff`](CITATION.cff).

---

## License

- **Code:** GNU General Public License v3.0 ([`LICENSE`](LICENSE)). The U-Net implementation (`fuelmap/model.py`) and
  parts of the training loop (`scripts/train.py`) are adapted from
  [milesial/Pytorch-UNet](https://github.com/milesial/Pytorch-UNet) (GPL-3.0).
- **Data, labels, model weights and map products:** Creative Commons Attribution 4.0 International (CC BY 4.0), see
  [`LICENSE-DATA.md`](LICENSE-DATA.md). The same file has notes on third-party sources: NAIP, USGS 3DEP and the
  county building footprints that some building labels were derived from. Labels derived from the San Luis Obispo
  County footprint layer are not distributed, because that layer's terms do not allow redistribution.

---

## Acknowledgments

This research was funded by the NASA Land Cover and Land Use Change Program (grant 80NSSC21K0295). It was also partially
supported by the USDA National Institute of Food and Agriculture (Hatch project CA-DLAW2620-H) and by AmericaView
through the National Land Remote Sensing Education, Outreach and Research Activity (NLRSEORA) grant from the U.S.
Geological Survey (USGS) National Land Imaging (NLI) program.

---

## Contact

**Yunzhe Zhu**, ypfzhu@ucdavis.edu
Department of Land, Air, and Water Resources, University of California, Davis

For bug reports or questions about the code, please open an [issue](https://github.com/YunzheZhu/fuel-mapping-CA/issues).

---

## Roadmap

- [x] **v1.0.0**: code for labeling, training, prediction and evaluation; release model; training, validation and test
  arrays; tile index.
- [x] **v2 statewide maps (2016–2024)** on Box.
- [ ] **v2 model** (forthcoming): the fine-tuned model behind the v2 maps, with its accuracy assessment. The results will
  be added here and kept separate from the paper's v1 results.
