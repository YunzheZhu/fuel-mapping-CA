# Pipeline guide

This guide runs every stage of the workflow, from LiDAR and NAIP to statewide maps. The stages follow the paper:
labeling (§2.4, Fig. 2), training (§2.5, Fig. 3), accuracy assessment (§2.6–2.7) and large-area mapping (§2.8).

Run all commands from the repository root. They use bash syntax. In PowerShell, put each command on one line, or end
continued lines with a backtick instead of `\`. Every script has `--help`.

| Stage | Script | Section |
|---|---|---|
| Tiles, LiDAR and NAIP inputs | `scripts/preprocessing/{make_tiles.py, download_3dep_lidar.py, clip_naip_tiles.py}` | [1](#1-tiles-lidar-and-naip) |
| Labels | `scripts/labeling/{01_lidar_segmentation.R, 02_rasterize_labels.py}` | [2](#2-labels) |
| Training arrays | `scripts/labeling/03_build_training_arrays.py` | [3](#3-training-arrays) |
| Training | `scripts/train.py` | [4](#4-train) |
| Test-tile prediction and accuracy | `scripts/predict_tiles.py`, `scripts/evaluate.py` | [5](#5-reproduce-the-release-model-accuracy) |
| Large-area prediction | `scripts/predict_naip.py` | [6](#6-map-naip-scenes) |
| Mosaics and COGs | `scripts/mosaic_cog.py` | [7](#7-mosaic-and-cloud-optimized-geotiffs) |
| 30 m fractional cover | `scripts/aggregate_30m.py` | [8](#8-fractional-cover-at-30-m) |

## 0. Setup

```bash
conda env create -f environment.yml && conda activate fuelmap   # or: pip install -r requirements.txt
python tests/test_smoke.py                                       # 9 tests, no data needed

BASE=https://github.com/YunzheZhu/fuel-mapping-CA/releases/download/v1.0.0
curl -L -o models/fuelmap_unet_v1.pth $BASE/fuelmap_unet_v1.pth

mkdir -p release && cd release
curl -LO $BASE/SHA256SUMS.txt
curl -LO $BASE/fuelmap_test_1y.npz                                   # 715 MB, needed for section 5
# training and validation arrays (11 GB), needed only for section 4:
curl -LO $BASE/fuelmap_val_4y.npz
for i in 1 2 3 4 5 6; do curl -LO $BASE/fuelmap_train_4y_part0${i}of06.npz; done
sha256sum -c --ignore-missing SHA256SUMS.txt
cd ..
```

Labeling (sections 1–3) needs more tools: geopandas, shapely ≥ 2, pyproj, PDAL, and R with lidR, terra, sf,
exactextractr and data.table. See [`scripts/labeling/README.md`](../scripts/labeling/README.md).

Conventions used by every script:
- class values 0 other, 1 tree, 2 shrub, 3 building, 255 nodata/ignore;
- NAIP bands R, G, B, NIR, as uint8 digital numbers (DN), scaled by 1/255 inside the model code;
- tiles of 512 × 512 px at 0.6 m, in EPSG:6414.

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

The steps follow paper Fig. 2.

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

Whole-tile prediction of the 806-tile test array, then pixel-based accuracy (about 12 min on a 12-thread CPU and
about 1.5 GB RAM):

```bash
python scripts/predict_tiles.py --config configs/release_v1.yaml --weights models/fuelmap_unet_v1.pth \
    --npz release/fuelmap_test_1y.npz --npz-key images --output outputs/test_1y

python scripts/evaluate.py --pred-npy outputs/test_1y/pred_class.npy \
    --label-npz release/fuelmap_test_1y.npz --label-key labels \
    --attributes data/test_array_index.csv --id-column row \
    --group-by wui_class rrk_region ecoregion_l3 label_scope \
    --out-json outputs/test_1y/metrics.json --out-csv outputs/test_1y/metrics.csv \
    --per-tile-csv outputs/test_1y/per_tile.csv
```

Expected overall result (paper rule): **OA 0.8700, mIoU 0.7353, mean F1 0.8461**. The F1 scores for other, tree, shrub
and building are 0.9106, 0.8301, 0.8036 and 0.8399. Add `--rule argmax` to `predict_tiles.py` for the argmax result
(OA 0.8596).

Notes:
- **Nodata masking.** Leave `--mask-nodata` off, which is the default, to reproduce these numbers exactly. One test
  tile has all-zero image pixels that carry valid labels.
- **Metric conventions.** The confusion matrix has the reference in rows and the prediction in columns. UA = precision
  and PA = recall. mIoU and mean F1 are unweighted means over classes 0–3, skipping only classes absent from both
  the reference and the prediction. A class that is predicted but absent from the reference counts as 0, so the
  `label_scope = building_only` group has a low mIoU (0.22) although its building F1 is 0.94.
- **Full-label strata.** To compute the stratified tables for full-label tiles only (as in the model card), pass an
  attribute file without the building-only rows:

  ```bash
  python -c "import csv; r=list(csv.DictReader(open('data/test_array_index.csv'))); w=csv.DictWriter(open('outputs/full_label_rows.csv','w',newline=''),fieldnames=r[0].keys()); w.writeheader(); w.writerows(x for x in r if x['label_scope']=='full')"
  python scripts/evaluate.py --pred-npy outputs/test_1y/pred_class.npy \
      --label-npz release/fuelmap_test_1y.npz --label-key labels \
      --attributes outputs/full_label_rows.csv --id-column row --group-by label_scope wui_class rrk_region \
      --out-csv outputs/test_1y/metrics_full_label.csv
  ```

  The overall row still covers all 806 tiles. The groups cover only the rows in the attribute file.

**Your own GeoTIFF tiles.** Predict a folder of tiles and pair each prediction with its label by name:

```bash
python scripts/predict_tiles.py --weights models/fuelmap_unet_v1.pth --images tiles/ --pattern "*_NAIP_????.tif" \
    --output pred/
python scripts/evaluate.py --pred-dir pred/ --label-dir tiles/ --label-template "{stem}_Label.tif" \
    --out-csv metrics.csv
```

## 6. Map NAIP scenes

`predict_naip.py` handles rasters of any size (GeoTIFF, COG or VRT), and memory depends on the raster width only:
about 45 KB per raster column (65 KB with `--probs`) plus the model, so about 0.5 GB for a 10,000 px wide scene and
about 7.5 GB for a 167,000 px (100 km) wide mosaic. For very wide areas, predict scene by scene and mosaic afterwards
(section 7).

- **Windows.** It reads the raster in strips and predicts 1,666 px windows with 25 % overlap. Overlaps are merged with
  linear-ramp blending, as in the original pipeline.
- **Output.** It writes a uint8 class GeoTIFF on the input grid: tiled, DEFLATE, nodata 255 where all bands are 0, with
  a colour table. `--probs` also writes 4-band uint8 probabilities (p × 255).

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
  (80 windows). For large areas use a GPU (`--device cuda`, with a larger `--batch-size`).
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

## Not included

These parts are not scripted in this repository:
- the interactive sampling and train/test split;
- the 600 dynamics plots near the Camp Fire (Fig. 7);
- the Sonoma County comparison data (Fig. 10);
- the figure scripts.
