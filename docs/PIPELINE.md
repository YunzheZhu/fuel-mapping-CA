# Pipeline guide

This guide runs every stage of the workflow, from LiDAR and NAIP to statewide maps. The stages follow the paper:
labeling (§2.4, Fig. 2), training (§2.5, Fig. 3), accuracy assessment (§2.6–2.7) and large-area mapping (§2.8).

Run all commands from the repository root. They use bash syntax. In PowerShell, put each command on one line, or end
continued lines with a backtick instead of `\`. Every script has `--help`.

| Stage | Paper | Script | Section |
|---|---|---|---|
| Stratified sampling of grid cells / tiles (WUI type, ecoregion, urban/wildland) | §2.3, Fig. 1 | not scripted; the result is [`data/tile_index.gpkg`](../data/README.md) | [1](#1-tiles-lidar-and-naip) |
| Tile grid: 307.2 m tiles (512 px at 0.6 m), EPSG:6414 | §2.3 | `scripts/preprocessing/make_tiles.py` | [1](#1-tiles-lidar-and-naip) |
| USGS 3DEP point clouds per tile | §2.2 | `scripts/preprocessing/download_3dep_lidar.py` | [1](#1-tiles-lidar-and-naip) |
| NAIP tiles (R, G, B, NIR) | §2.2 | `scripts/preprocessing/clip_naip_tiles.py` | [1](#1-tiles-lidar-and-naip) |
| LiDAR object segmentation; initial labels from planar mask and height | Fig. 2 (1)–(2) | `scripts/labeling/01_lidar_segmentation.R` | [2](#2-labels) |
| Manual check of the labels against same-year NAIP | Fig. 2 (3) | by hand in a GIS | [2](#2-labels) |
| Rasterize to 0.6 m, NAIP NDVI mask | Fig. 2 (4) | `scripts/labeling/02_rasterize_labels.py` | [2](#2-labels) |
| Training/testing arrays | §2.4–2.5 | `scripts/labeling/03_build_training_arrays.py` | [3](#3-training-arrays) |
| U-Net training | §2.5, Fig. 3 | `scripts/train.py` + `configs/release_v1.yaml` | [4](#4-train) |
| Prediction on test tiles | §2.6 | `scripts/predict_tiles.py` | [5](#5-reproduce-the-release-model-accuracy) |
| Pixel-based accuracy (Tables 1–2) | §2.6–2.7 | `scripts/evaluate.py` | [5](#5-reproduce-the-release-model-accuracy) |
| Demo figures of the README | – | `scripts/make_demo_figures.py` | [5](#readme-demo-figures) |
| County / statewide prediction | §2.8, §3.5 | `scripts/predict_naip.py`, `scripts/mosaic_cog.py` | [6](#6-map-naip-scenes), [7](#7-mosaic-and-cloud-optimized-geotiffs) |
| 30 m fractional cover | §2.7–2.8, Figs. 6, 10, 11 | `scripts/aggregate_30m.py` | [8](#8-fractional-cover-at-30-m) |
| Prediction from Python | – | `fuelmap` package | [Python API](#python-api) |

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
│   ├── make_demo_figures.py    # the README demo figures (release model on the public test tiles and a NAIP scene)
│   ├── mosaic_cog.py           # mosaic / reproject class maps into Cloud-Optimized GeoTIFFs
│   └── aggregate_30m.py        # fractional cover on a 30 m grid
├── configs/release_v1.yaml     # training and inference settings of the release model
├── models/                     # model metadata (fuelmap_unet_v1.json); the weights are a release asset
├── data/                       # tile index, test-array index, sample tiles (data dictionary inside)
├── docs/                       # MODEL_CARD.md, PIPELINE.md, DATA.md; figures/ (paper figures, demo figures)
├── tests/test_smoke.py         # smoke tests, no data needed
├── environment.yml, requirements.txt
└── CITATION.cff, LICENSE (GPL-3.0), LICENSE-DATA.md (CC BY 4.0)
```

## 0. Setup

```bash
git clone https://github.com/YunzheZhu/fuel-mapping-CA.git && cd fuel-mapping-CA
conda env create -f environment.yml && conda activate fuelmap   # creates the env "fuelmap"; or: pip install -r requirements.txt
python tests/test_smoke.py                                       # 9 tests, about 5 s on CPU, no data needed (or: pytest tests)

BASE=https://github.com/YunzheZhu/fuel-mapping-CA/releases/download/v1.0.0
curl -L -o models/fuelmap_unet_v1.pth $BASE/fuelmap_unet_v1.pth  # release model, 124 MB
sha256sum models/fuelmap_unet_v1.pth   # macOS: shasum -a 256
# 35e9f387de0017e7cbddbdfc7fc8cbeb5645eb114a7fc03cdd8ab8acdde5706a

mkdir -p release && cd release
curl -LO $BASE/SHA256SUMS.txt
curl -LO $BASE/fuelmap_test_1y.npz                                   # 615 MB, needed for section 5
# training and validation arrays (10.5 GB), needed only for section 4:
curl -LO $BASE/fuelmap_val_4y.npz
for i in 1 2 3 4 5 6; do curl -LO $BASE/fuelmap_train_4y_part0${i}of06.npz; done
sha256sum -c --ignore-missing SHA256SUMS.txt
cd ..
```

Labeling (sections 1–3) needs more tools: geopandas, shapely ≥ 2, pyproj, PDAL, and R with lidR, terra, sf,
exactextractr and data.table. See [`scripts/labeling/README.md`](../scripts/labeling/README.md).

**Requirements and tested versions**

- Requires Python ≥ 3.9 and PyTorch ≥ 2.0. Other dependencies: numpy, rasterio, PyYAML and tqdm. GDAL's Python
  bindings are needed only for `scripts/mosaic_cog.py`. For a CUDA build of PyTorch see
  https://pytorch.org/get-started/locally/. `scripts/make_demo_figures.py` also needs matplotlib ≥ 3.6 (which
  installs Pillow).
- The scripts run from the repository root. You do not need to install the package.
- Tested with Python 3.12.3, torch 2.3.1 (CPU), numpy 1.26.4, rasterio 1.3.9 and GDAL 3.8.4, and with the minimum
  versions Python 3.10.9, torch 2.0.0, numpy 1.24.2 and rasterio 1.3.6 (identical predictions). numpy 2 has not been
  tested. The paper's training environment was Python 3.10.13 with torch 2.1.1.

Conventions used by every script:
- class values 0 other, 1 tree, 2 shrub, 3 building, 255 nodata/ignore;
- NAIP bands R, G, B, NIR, as uint8 digital numbers (DN), scaled by 1/255 inside the model code;
- tiles of 512 × 512 px at 0.6 m, in EPSG:6414.

### Check the installation on the sample tiles

[`data/sample`](../data/sample) holds 4 NAIP test tiles with their labels. Predict and evaluate them:

```bash
python scripts/predict_tiles.py --config configs/release_v1.yaml --weights models/fuelmap_unet_v1.pth \
    --images data/sample --pattern "*_naip.tif" --output outputs/sample_pred

python scripts/evaluate.py --pred-dir outputs/sample_pred --label-dir data/sample \
    --stem-regex "^(.+)_naip$" --label-template "{stem}_label.tif" \
    --out-csv outputs/sample_metrics.csv --per-tile-csv outputs/sample_per_tile.csv
```

This should print OA 0.769 over the 4 tiles (per tile 0.68–0.98; the urban tile `URB_0014` has building labels only).
The sample is too small to show the model's accuracy; use it to check that your installation works. The full test
results are in [section 5](#5-reproduce-the-release-model-accuracy) and in the [model card](MODEL_CARD.md#evaluation).

## 1. Tiles, LiDAR and NAIP

The paper sampled grid cells from a statewide fishnet, stratified by WUI type, ecoregion and urban/wildland gradient
(§2.3). In the released index, wildland and validation tiles are 3 × 3 tiles of 1 km cells, WUI tiles are 2 × 2
quadrants of their cells, and urban tiles were sampled individually. That sampling was done interactively and is not
scripted here. The tiles it produced are in `data/tile_index.gpkg`; see [`data/README.md`](../data/README.md).

```bash
# 3 x 3 tiles of 307.2 m (512 px x 0.6 m) per cell (the wildland design with 1 km cells; --n 2 gives quadrants),
# named <cell_id>_<NW|NN|NE|WW|CE|EE|SW|SS|SE>
python scripts/preprocessing/make_tiles.py --cells sampled_cells.gpkg --id-field cell_id --out tiles.gpkg

# USGS 3DEP point clouds (public AWS bucket, no credentials): list candidate surveys, keep one per tile, download
python scripts/preprocessing/download_3dep_lidar.py list --tiles tiles.gpkg --out lidar.csv
python scripts/preprocessing/download_3dep_lidar.py download --tiles tiles.gpkg --candidates lidar.csv --outdir laz/

# NAIP scenes (downloaded separately) warped onto each tile's grid -> tiles/ID_<tile>_NAIP_<year>.tif
python scripts/preprocessing/clip_naip_tiles.py --tiles tiles.gpkg --scenes naip/2018/ --year 2018 --outdir tiles/
```

Where to get NAIP: the paper downloaded the scenes through the USGS M2M API. They are also on EarthExplorer, Microsoft
Planetary Computer and Google Earth Engine; see [`scripts/preprocessing/README.md`](../scripts/preprocessing/README.md).
`make_tiles.py` reproduces the paper's 4,050 wildland tiles exactly.

## 2. Labels

The steps follow paper Fig. 2 (shown with its caption in [`figures/paper`](figures/paper/README.md#fig-2-label-workflow)).

**Labeling rules (Fig. 2)**

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
note in [`scripts/labeling/README.md`](../scripts/labeling/README.md).

**Commands**

```bash
# (1)-(2) LiDAR object segmentation and initial classes (Clss_95: 0 other, 1 tree, 2 shrub, 3 building)
Rscript scripts/labeling/01_lidar_segmentation.R --input laz/ --outdir seg/ --threads 4

# (3) Check and edit seg/*_crowns.gpkg over the NAIP tile of the LiDAR year in a GIS (true and false colour).

# (4) Rasterize onto the NAIP tile grid. Tree/shrub pixels with NDVI <= 0.11 become other. --buildings: county
#     footprints override LiDAR buildings. --chm: pixels without LiDAR become 255.
python scripts/labeling/02_rasterize_labels.py \
    --naip tiles/ID_123_CE_NAIP_2018.tif \
    --crowns seg/ID_123_CE_<survey>_crowns.gpkg \
    --chm seg/ID_123_CE_<survey>_chm.tif \
    --buildings county_footprints.gpkg \
    --out tiles/ID_123_CE_NAIP_2018_Label.tif
# many tiles: --manifest pairs.csv  (columns naip,crowns,out[,buildings_only])
```

To label another NAIP year, rasterize the same checked polygons onto that year's NAIP tile. The NDVI mask then follows
that year (the paper's temporal label augmentation). `--buildings-only` writes footprint-only labels (building = 3,
everything else 255). Some urban-sample tiles in the released arrays have this kind of label.

`01_lidar_segmentation.R` is a reference re-implementation of the 2023–24 workflow. Its CHM and planar mask match the
original outputs. Its object tops and polygons differ, and the initial classes agree on 94 % of pixels on the test tile.
See the provenance note in [`scripts/labeling/README.md`](../scripts/labeling/README.md).

## 3. Training arrays

```bash
# one row per (NAIP tile, label tile) pair -> arrays/train_images.npz, arrays/train_labels.npz, arrays/train_tiles.csv
python scripts/labeling/03_build_training_arrays.py --naip-dir tiles/ --split train --out arrays/train
```

The outputs are uncompressed `.npz` files with key `arr_0`, which `scripts/train.py` memory-maps directly. Use a
manifest (`--manifest`) to choose the rows:
- one row per tile at the LiDAR-matched NAIP year;
- or extra rows for other NAIP years. The paper did this for the WUI and testing tiles.

**Using the released arrays instead.** They are compressed and have the members `images` and `labels`. Convert them
once to uncompressed single-member files. Training then memory-maps them, and DataLoader workers do not copy them:

```bash
python - <<'EOF'
import glob, numpy as np
for f in sorted(glob.glob("release/fuelmap_*.npz")):
    if f.endswith(("_images.npz", "_labels.npz")):
        continue
    d = np.load(f)
    np.savez(f[:-4] + "_images.npz", arr_0=d["images"])   # e.g. release/fuelmap_val_4y_images.npz
    np.savez(f[:-4] + "_labels.npz", arr_0=d["labels"])
    print("converted", f)
EOF
```

## 4. Train

**A new model (recommended defaults).** This selects on the validation split, ignores unlabelled pixels and uses a
4-output head:

```bash
python scripts/train.py \
    --train-images release/fuelmap_train_4y_part0*of06_images.npz \
    --train-labels release/fuelmap_train_4y_part0*of06_labels.npz \
    --val-images release/fuelmap_val_4y_images.npz --val-labels release/fuelmap_val_4y_labels.npz \
    --epochs 100 --batch-size 20 --data-parallel --workers 8 --out-dir runs/my_model
```

**The release-model recipe.** `configs/release_v1.yaml` holds the release model's settings: 5 outputs, unlabelled
pixels as other, 50 epochs, selection on the test split, and the legacy rotation fill. Its data paths are placeholders,
so pass all three splits on the command line. The release packages store "no label" as 255 (it was 4 in the original
arrays), so set `--unlabeled-value 255` to keep the original treatment:

```bash
python scripts/train.py --config configs/release_v1.yaml \
    --train-images release/fuelmap_train_4y_part0*of06_images.npz \
    --train-labels release/fuelmap_train_4y_part0*of06_labels.npz \
    --val-images release/fuelmap_val_4y_images.npz --val-labels release/fuelmap_val_4y_labels.npz \
    --test-images release/fuelmap_test_1y_images.npz --test-labels release/fuelmap_test_1y_labels.npz \
    --unlabeled-value 255 --out-dir runs/release_recipe
```

The conversion above also writes the test array as `release/fuelmap_test_1y_{images,labels}.npz`. The release model
was selected on the 4-year test array, which is not released; the one-year test array is the closest public substitute.
PowerShell does not expand `*` in arguments, so list the six part files explicitly there.

Outputs in `--out-dir`:

| File | Content |
|---|---|
| `best.pth` | Plain `state_dict` of the best checkpoint |
| `best.json` | Its metadata and SHA-256 |
| `last.pth` | The latest epoch |
| `history.csv` | Per-epoch loss and metrics |
| `config.json` | The arguments used |

Other options:
- `--amp` for mixed precision;
- `--init-weights models/fuelmap_unet_v1.pth` for fine-tuning. The release weights have a 5-output head: with the
  default `--n-outputs 4` the four class channels are copied and the legacy fifth channel is dropped; add
  `--n-outputs 5` to keep the release model's head;
- `--wandb` for Weights & Biases logging (optional; off by default);
- `--limit-train-batches N` for a quick dry run.

Bit-exact re-training is not expected; see [`MODEL_CARD.md`](MODEL_CARD.md#training-procedure).

## 5. Reproduce the release-model accuracy

Whole-tile prediction of the 693-tile public test array, then pixel-based accuracy (about 10 min on a 12-thread CPU
and about 1.5 GB RAM):

```bash
python scripts/predict_tiles.py --config configs/release_v1.yaml --weights models/fuelmap_unet_v1.pth \
    --npz release/fuelmap_test_1y.npz --npz-key images --output outputs/test_1y

python scripts/evaluate.py --pred-npy outputs/test_1y/pred_class.npy \
    --label-npz release/fuelmap_test_1y.npz --label-key labels \
    --attributes data/test_array_index.csv --id-column index \
    --group-by wui_class rrk_region ecoregion_l3 label_scope \
    --out-json outputs/test_1y/metrics.json --out-csv outputs/test_1y/metrics.csv \
    --per-tile-csv outputs/test_1y/per_tile.csv
```

Expected overall result (paper rule): **OA 0.8708, mIoU 0.7249, mean F1 0.8388**. The F1 scores for other, tree, shrub
and building are 0.9135, 0.8305, 0.8040 and 0.8071. Add `--rule argmax` to `predict_tiles.py` for the argmax result
(OA 0.8592).

`evaluate.py` identifies npz tiles by their position in the label array, which is the `index` column of
`data/test_array_index.csv`. The `row` column is the row in the original 2024 test array (806 rows). The 113
San Luis Obispo urban-sample tiles of that array are not in the public package; see
[`DATA.md`](DATA.md#tiles-left-out-of-the-arrays).

Notes:
- **Nodata masking.** Leave `--mask-nodata` off, which is the default, to reproduce these numbers exactly. One test
  tile has all-zero image pixels that carry valid labels.
- **Metric conventions.** The confusion matrix has the reference in rows and the prediction in columns. UA = precision
  and PA = recall. mIoU and mean F1 are unweighted means over classes 0–3, skipping only classes absent from both
  the reference and the prediction. A class that is predicted but absent from the reference counts as 0, so the
  `label_scope = building_only` group has a low mIoU (0.23) although its building F1 is 0.95.
- **Full-label strata.** To compute the stratified tables for full-label tiles only (as in the model card), pass an
  attribute file without the building-only rows:

  ```bash
  python -c "import csv; r=list(csv.DictReader(open('data/test_array_index.csv'))); w=csv.DictWriter(open('outputs/full_label_rows.csv','w',newline=''),fieldnames=r[0].keys()); w.writeheader(); w.writerows(x for x in r if x['label_scope']=='full')"
  python scripts/evaluate.py --pred-npy outputs/test_1y/pred_class.npy \
      --label-npz release/fuelmap_test_1y.npz --label-key labels \
      --attributes outputs/full_label_rows.csv --id-column index --group-by label_scope wui_class rrk_region \
      --out-csv outputs/test_1y/metrics_full_label.csv
  ```

  The overall row still covers all 693 tiles. The groups cover only the rows in the attribute file.

**Your own GeoTIFF tiles.** Predict a folder of tiles and pair each prediction with its label by name:

```bash
python scripts/predict_tiles.py --weights models/fuelmap_unet_v1.pth --images tiles/ --pattern "*_NAIP_????.tif" \
    --output pred/
python scripts/evaluate.py --pred-dir pred/ --label-dir tiles/ --label-template "{stem}_Label.tif" \
    --out-csv metrics.csv
```

### README demo figures

`scripts/make_demo_figures.py` makes the figures in the README section
[Try it yourself](../README.md#try-it-yourself-the-release-model-on-the-shared-test-data). It predicts the
public test package with the release model and the paper rule (the same numbers as above), and optionally runs the
sliding window of `predict_naip.py` on a window of one NAIP scene:

```bash
python scripts/make_demo_figures.py --weights models/fuelmap_unet_v1.pth \
    --test-npz release/fuelmap_test_1y.npz --index data/test_array_index.csv \
    --pred-cache outputs/demo_pred_cache.npz \
    --naip m_3912128_se_10_060_20180718.tif --naip-window 4720 6920 2000 2000 --zoom 1100 500 500
```

- **Outputs** (default `--out-dir docs/figures/demo`): `test_gallery.png`, `confusion_matrix.png`,
  `per_tile_oa.png`, `naip_scene_demo.png`, plus `demo_metrics.json` (all metrics: overall, per class, per WUI
  class, and the per-tile OA counts quoted in the README) and `demo_tiles.csv` (the gallery tiles and the rule that
  chose them).
- **Gallery tiles** are chosen by a fixed rule, not by eye: within each WUI class (full-label tiles; class "Unknown"
  left out) the tile whose per-tile OA is closest to the class median, ties broken by tile_id. The last row applies
  the same rule to the building-only urban-sample tiles, whose per-tile score is building recall. Rows group tiles
  by `wui_class`; the tile_id prefix (`URB`, `WUI`, `WLD`) is the sampling stratum, which can differ.
- **The NAIP scene** (`m_3912128_se_10_060_20180718`, near Oroville, Butte County, 2018) is not part of the release.
  Download it from USGS EarthExplorer (NAIP) or the Microsoft Planetary Computer `naip` collection; copies from
  different sources may differ slightly in pixels or grid. The scene (a 2018 quarter-quad with no labelled tile in
  it), the window and the zoom box were chosen by eye to show houses among oak woodland. The scene part gives the
  same classes as `predict_naip.py --window 4720 6920 2000 2000`. Leave out `--naip`, `--naip-window` and `--zoom` to make only the
  test-tile figures, or leave out `--test-npz` to make only the scene figure.
- **Run time.** About 8 min on a 24-core CPU without a cache. `--pred-cache` stores the test predictions
  (`outputs/` is git-ignored), so later runs only redraw the figures.

## 6. Map NAIP scenes

`predict_naip.py` handles rasters of any size (GeoTIFF, COG or VRT), and memory depends on the raster width only:
about 45 KB per raster column (65 KB with `--probs`) plus the model, so about 0.5 GB for a 10,000 px wide scene and
about 7.5 GB for a 167,000 px (100 km) wide mosaic. For very wide areas, predict scene by scene and mosaic afterwards
(section 7).

- **Windows.** It reads the raster in strips and predicts 1,666 px windows with 25 % overlap. Overlaps are merged with
  linear-ramp blending, as in the original pipeline.
- **Output.** It writes a uint8 class GeoTIFF on the input grid (0 other, 1 tree, 2 shrub, 3 building, 255 nodata):
  tiled, DEFLATE, nodata 255 where all bands are 0, with a colour table in the colours of the paper's maps (tree
  blue, shrub orange, building red, other black). `--probs` also writes 4-band uint8 probabilities (p × 255).

```bash
# one scene
python scripts/predict_naip.py --config configs/release_v1.yaml --weights models/fuelmap_unet_v1.pth \
    --input naip/2018/m_3912127_ne_10_060_20180718.tif --output pred_2018/m_3912127_ne_10_060_20180718_class.tif

# a folder of scenes; resumable
python scripts/predict_naip.py --config configs/release_v1.yaml --weights models/fuelmap_unet_v1.pth \
    --input naip/2018/ --output pred_2018/ --skip-existing

# a 2000 x 2000 px test window (column offset, row offset, width, height)
python scripts/predict_naip.py --weights models/fuelmap_unet_v1.pth --input scene.tif --output test.tif \
    --window 4000 5000 2000 2000
```

- **Speed.** About 10–12 s per 1,666 px window on a 12-thread CPU, so roughly 15 min per NAIP quarter-quad
  (about 9,600 × 12,200 px, 80 windows). For large areas use a GPU (`--device cuda`, with a larger `--batch-size`).
- **NAIP band order.** NAIP DOQQs are already R, G, B, NIR. Use `--bands` if your files differ.
- **NDVI gate.** By default the NDVI gate of the decision rule is applied at 0.6 m. The archived paper-era map
  pipeline applied it later, during 30 m aggregation. `--no-ndvi` reproduces that behaviour.
- **Single tiles.** For 512 px tiles use `predict_tiles.py`. `predict_naip.py` zero-pads edge windows, so its result on
  a single small tile differs slightly from whole-tile inference.

## 7. Mosaic and Cloud-Optimized GeoTIFFs

This step needs the GDAL Python bindings (`osgeo.gdal`).

```bash
# all scene predictions of an area -> one COG in California Albers at 0.6 m (nearest neighbour, nodata 255)
python scripts/mosaic_cog.py --input pred_2018/ --pattern "*_class.tif" \
    --output county_2018_class_cog.tif --t-srs EPSG:6414 --res 0.6 --tap

# or convert each scene to a COG on its own grid
python scripts/mosaic_cog.py --input pred_2018/ --per-file pred_2018_cog/
```

With `--t-srs`, all inputs are warped to one grid: pixel size `--res` (default: that of the first input), with pixel
edges at multiples of it (`--tap`, the default; `--no-tap` lets the parts drift by a fraction of a pixel and leave
gaps at the seams). Where inputs overlap, the last file in sorted order wins.

## 8. Fractional cover at 30 m

```bash
# on a reference grid (e.g. a LANDFIRE 30 m layer)
python scripts/aggregate_30m.py --input county_2018_class_cog.tif --reference LF_EVC_30m.tif \
    --output county_2018_fraction_30m.tif
# or a plain 30 m grid
python scripts/aggregate_30m.py --input scene_class.tif --res 30 --crs EPSG:6414 --output scene_30m.tif
```

The output is float32 with 5 bands: tree, shrub, building, woody (tree + shrub) and valid. Each band is a fraction of
the valid 0.6 m pixels in the cell.

- **Invalid pixels.** Values outside 0–3 (nodata 255, the input's declared nodata, negative values, and the v2 class 4
  "uncertain") count as invalid.
- **Nodata.** Cells with a valid fraction below 0.5 are nodata (−1); `--min-valid` changes the limit.
- **Masks.** `--exclude-mask` sets masked pixels (non-zero values, e.g. water) to other before aggregation. The
  mask's own nodata and areas outside the mask are not masked.

The R² comparisons of the paper (Figs. 6 and 10) are simple regressions of these fractions against LiDAR-derived
fractions, and are not scripted here.

## Python API

The scripts are thin wrappers around the `fuelmap` package. To predict one tile from Python:

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

`fuelmap.model.load_model` accepts plain and `module.`-prefixed (DataParallel) state dicts. It works with 4- and
5-channel heads and infers the number of outputs from the weights. For an in-memory image larger than a
tile, `fuelmap.predict_array(model, image)` runs the sliding window of `predict_naip.py` on a `(bands, H, W)` array of
0–255 values; for rasters on disk use the script itself.

## Not included

These parts are not scripted in this repository:
- the interactive sampling and train/test split;
- the 600 dynamics plots near the Camp Fire (Fig. 7);
- the Sonoma County comparison data (Fig. 10);
- the figure scripts.
