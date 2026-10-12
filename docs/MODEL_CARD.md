# Model card: fuelmap U-Net v1 (`fuelmap_unet_v1.pth`)

| | |
|---|---|
| Model | `fuelmap_unet_v1`, version 1.0.0 |
| File | [`fuelmap_unet_v1.pth`](https://github.com/YunzheZhu/fuel-mapping-CA/releases/download/v1.0.0/fuelmap_unet_v1.pth) (GitHub release `v1.0.0`), 124,238,486 bytes |
| SHA-256 | `35e9f387de0017e7cbddbdfc7fc8cbeb5645eb114a7fc03cdd8ab8acdde5706a` |
| Metadata | [`models/fuelmap_unet_v1.json`](../models/fuelmap_unet_v1.json) |
| Paper | Zhu, Dixon & Jin (2026), *Remote Sensing of Environment* 333, 115091, https://doi.org/10.1016/j.rse.2025.115091 |
| License | CC BY 4.0 (weights); code GPL-3.0 |
| Contact | Yunzhe Zhu, ypfzhu@ucdavis.edu |

**What it is.** This is the release model of the paper. It was retrained (run `UNet_RGBN_4y_II`, June 2024) on the
paper's training tiles (the June 2024 version of the split, with the test tiles held out). It matches the paper's
reported overall accuracy: 87.0 % on the one-year test array, against 87.1 % in the paper. It is described here as the release model, not as the exact checkpoint behind the
paper's tables.

## Model details

| Item | Value |
|---|---|
| Architecture | U-Net (Ronneberger et al., 2015) in the layout of [milesial/Pytorch-UNet](https://github.com/milesial/Pytorch-UNet). 5 levels with 64–128–256–512–1024 kernels; each level has two 3×3 conv → BatchNorm → ReLU; 2×2 max-pooling; 2×2 transposed-convolution upsampling (`bilinear=False`); skip connections; 1×1 output conv |
| Parameters | 31,038,469 (118 tensors) |
| Input | NAIP 4-band `uint8` digital numbers, band order **R, G, B, NIR**, divided by 255 (float32, NCHW). No other normalization. Trained on 512 × 512 px tiles at 0.6 m (307.2 m); the network is fully convolutional, so any size works (windows of 1,666 px are used for large rasters) |
| Output | Logits with 5 channels: 0 other, 1 tree, 2 shrub, 3 building, and 4, a legacy "unlabeled" channel. Channel 4 had loss weight 0 and never received a positive target. At inference the softmax is taken over all 5 outputs and channels 0–3 are kept |
| Weights format | plain PyTorch `state_dict` (no `module.` prefix), float32; `torch.load(path, map_location="cpu", weights_only=True)` or `fuelmap.model.load_model(path)` |
| Framework | PyTorch. Paper-era training environment: PyTorch 2.1.1, Python 3.10.13. The release was checked with PyTorch 2.3.1 |

### Decision rule

The paper's test-tile accuracy assessment used probability thresholds plus an NDVI gate. The paper text does not spell
this rule out. `fuelmap.inference.classify(rule="paper")` implements it, and it is the default of all prediction
scripts:

1. tree if p_tree > 0.4 and NDVI > 0.11
2. shrub if p_shrub > 0.4 and NDVI > 0.11
3. building if p_building > 0.6
4. other if p_other > 0.5
5. every remaining pixel is other

Each later rule overrides the earlier ones. NDVI = (NIR − R) / (NIR + R) on DN / 255, and 0 where NIR + R = 0. The
thresholds are strict. Change them with `--tree-threshold`, `--shrub-threshold`, `--building-threshold`,
`--other-threshold` and `--ndvi-min`; `--no-ndvi` drops the NDVI gate; `--rule argmax` uses a plain argmax over
channels 0–3.

The archived large-area prediction script of the paper era applied the same thresholds without the NDVI gate. In that
pipeline, the NDVI ≤ 0.11 mask on tree and shrub was applied later, during aggregation to 30 m. `scripts/predict_naip.py`
applies the gate directly at 0.6 m by default. Use `--no-ndvi` to reproduce the archived large-area behaviour.

## Intended use

- Mapping trees, shrubs, buildings and other land at 0.6 m from **NAIP 4-band imagery over California**. The paper
  evaluated its model on NAIP 2016–2022 (Table 2). The release model was re-evaluated on the one-year test array
  (NAIP 2016: 296 tiles, 2018: 433, 2020: 76, 2022: 1); it has not been evaluated across years.
- Typical uses:
  - fuel and structure mapping in the wildland–urban interface;
  - screening of vegetation around buildings (defensible-space zones);
  - canopy and shrub cover estimates aggregated to coarser grids (e.g. 30 m);
  - monitoring change between NAIP years (fire, regrowth, logging, rebuilding);
  - an independent fine-scale reference for coarser products such as LANDFIRE.

**Out of scope / use with care**

- **Other imagery.** Other sensors, other band orders, pan-sharpened or 8-bit RGB-only imagery, and NAIP resampled to a
  different resolution were not tested.
- **Other places and years.** Regions outside California and NAIP years other than 2016–2022 (e.g. 1 m NAIP before
  2016, or 2024) were not evaluated for this model. The v2 maps on Box come from a different, fine-tuned model.
- **Regulatory or legal decisions without field verification.** Examples are defensible-space compliance, insurance
  and property records. The maps are model outputs with the errors described below.
- **Object counts and species.** The output is per-pixel. Adjacent crowns or buildings merge, so the maps do not give
  tree or building counts. Tree and shrub are defined by height (see below), not by species or fuel model.

## Training data

| Item | Value |
|---|---|
| Labels | Semi-automatic labels (paper §2.4, Fig. 2). Objects come from USGS 3DEP aerial LiDAR (42 surveys, 2015–2022, > 10 pts/m²), and their initial classes from a planar mask and heights: other below 1 m; otherwise building if planar cover ≥ 50 %, tree if the 95th-percentile height is ≥ 5 m (paper: "above 5 m"), else shrub. County footprints (Napa, San Luis Obispo, Shasta) replace LiDAR buildings where available. A person checked the labels against NAIP from the LiDAR year. They were then rasterized to 0.6 m, and tree/shrub pixels with NAIP NDVI ≤ 0.11 (paper: "below 0.11") were set to other |
| Sampling | Stratified by FRAP 2015 WUI type, EPA Level III ecoregion and urban/wildland gradient (paper §2.3); the tiles are listed in [`data/tile_index.gpkg`](../data/README.md) |
| Training array | 10,969 tile-years (`fuelmap_train_4y_part01..06of06.npz`): 6,649 tiles at their LiDAR-matched NAIP year (1,725 urban building-only + 4,924 WUI/wildland) plus 4,320 rows of WUI tiles at the other NAIP years, relabelled with that year's NDVI mask. Built in June 2024 from the paper's train/test split, before the split was finalised, so it holds fewer tiles than the 7,015 training tiles in [`data/tile_index.gpkg`](../data/README.md). The paper counts 7,015 training tiles and 10,086 training tile-years |
| Validation array | 1,512 tile-years (378 validation tiles × 4 NAIP years, `fuelmap_val_4y.npz`). Released for model development; the release checkpoint was not selected on it |
| Test array | 806 tiles at the LiDAR-matched NAIP year (`fuelmap_test_1y.npz`). The paper reports 782 testing tiles |
| Label scope | Most tiles have full labels (all four classes). Urban-sample tiles from the county-footprint counties carry **building-only labels** in the 2024 arrays: footprints are building and all other pixels are unlabelled |
| Unlabelled pixels | Value 4 in the 2024 arrays, 255 in the release packages. The archived data loader encoded them as class 0 ("other"; see [differences](#differences-between-the-paper-text-and-the-release-model)), so the release model saw them as other with loss weight 0.4 |

## Training procedure

These are the settings in [`configs/release_v1.yaml`](../configs/release_v1.yaml), recovered from the archived training
code and run logs.

| Setting | Value |
|---|---|
| Loss | Cross-entropy with class weights other 0.4, tree 1.0, shrub 1.1, building 1.5 (channel 4: 0) |
| Optimizer | RMSprop, learning rate 1e-8, weight decay 1e-6, momentum 0.999 |
| LR schedule | none in practice. The original code created `ReduceLROnPlateau`, but it never reduced the rate at 1e-8 |
| Gradient clipping | max norm 1.0 |
| Batch size | 20, split across GPUs with `nn.DataParallel` (the paper reports 4 × NVIDIA RTX 2080 Ti for its training) |
| Epochs | 50; checkpoint of epoch 47 released |
| Augmentation (training only) | horizontal and vertical flips, p = 0.5 each; rotation uniform in ±45°, p = 0.2 (nearest neighbour; rotated-in corners labelled other); scale jitter p = 0.2, a nearest-neighbour resample round trip with a factor in [0.9, 1.11] |
| Checkpoint selection | After each epoch the training code kept the checkpoint if mean Dice (= mean F1) and mIoU on the test split had both improved. It used the probability thresholds, without the NDVI gate. Epoch 47 was the last such epoch |
| Seed | 2024 |

`scripts/train.py --config configs/release_v1.yaml` repeats this recipe; see [`PIPELINE.md`](PIPELINE.md#4-train).
Exact re-training is not expected. The original Dice and mIoU code differed from the corrected metrics in
`fuelmap.metrics`, and multi-GPU training is not bit-reproducible. For new models, `scripts/train.py` selects on the
validation split and ignores unlabelled pixels by default.

## Evaluation

### Results reported in the paper

These are the paper's values on **782 testing tiles** (Tables 1 and 2). F1, mIoU, UA and PA are unweighted means over
the four classes, including other; every printed F1 matches the mean of its four class F1 values to rounding, except
the Wildland row (note ¹).

**Table 1** (matched NAIP–LiDAR year):

| Scenario | OA | UA* | PA* | mIoU | F1 | F1 Tree | F1 Shrub | F1 Building | F1 Other |
|---|---|---|---|---|---|---|---|---|---|
| Single NAIP year (overall) | 87.1% | 86.6% | 78.7% | 70.1% | 82.1% | 83.1% | 78.9% | 75.1% | 91.4% |
| Urban | 83.2% | 65.7% | 68.0% | 52.1% | 63.4% | 73.5% | 14.2% | 77.4% | 88.7% |
| WUI interface | 84.8% | 74.4% | 72.0% | 59.4% | 72.5% | 79.3% | 45.1% | 75.5% | 90.0% |
| WUI intermix | 87.2% | 77.8% | 72.0% | 61.4% | 74.2% | 85.6% | 51.4% | 68.1% | 91.5% |
| Wildland | 88.6% | 87.0% | 76.6% | 69.0% | 80.1%¹ | 83.8% | 83.8%¹ | 62.7% | 92.5% |
| Northern California (RRK) | 84.7% | 74.3% | 75.4% | 60.7% | 73.4% | 84.4% | 44.0% | 76.3% | 88.9% |
| Central California (RRK) | 85.4% | 81.2% | 78.2% | 66.5% | 79.3% | 84.9% | 67.3% | 75.2% | 89.7% |
| Sierra Nevada (RRK) | 85.8% | 80.4% | 74.6% | 63.1% | 76.5% | 85.8% | 63.1% | 68.4% | 88.5% |
| Southern California (RRK) | 88.4% | 82.1% | 79.8% | 67.7% | 79.8% | 66.7% | 85.6% | 73.8% | 93.2% |

¹ As printed in the paper. The four class F1 values of the Wildland row average to 80.7 %, not 80.1 %, so one value in
this row (possibly the repeated 83.8 %) is likely a typo in the paper.

**Table 2** (testing tiles with labels carried to each NAIP year):

| NAIP year | OA | UA* | PA* | mIoU | F1 | F1 Tree | F1 Shrub | F1 Building | F1 Other |
|---|---|---|---|---|---|---|---|---|---|
| 2016 | 85.0% | 83.5% | 76.6% | 66.7% | 79.6% | 81.6% | 72.5% | 74.5% | 89.8% |
| 2018 | 87.5% | 87.0% | 78.9% | 70.5% | 82.3% | 83.3% | 80.0% | 74.3% | 91.8% |
| 2020 | 85.0% | 85.5% | 76.1% | 67.2% | 80.0% | 82.8% | 73.5% | 74.1% | 89.7% |
| 2022 | 88.1% | 86.8% | 77.2% | 69.2% | 81.3% | 81.8% | 77.8% | 72.5% | 93.0% |

\* **UA/PA convention.** This repository uses UA = precision (column-wise, from the prediction side) and PA = recall
(row-wise, from the reference side). The paper's per-class confusion matrices are in Figs. S1–S2 of the supplement.
Compared with them, the means printed under "UA" in Tables 1–2 are mean recall, and those under "PA" are mean
precision. Keep this in mind when you compare UA and PA with the release-model numbers below.

30 m fractional cover against LiDAR reference:
- San Luis Obispo County region, 3,000 ha (Fig. 6): tree R² = 0.86, shrub R² = 0.93, building R² = 0.71.
- Sonoma County tree cover (Fig. 10): R² = 0.69, compared with R² = 0.35 for LANDFIRE 2016 EVC.

### Re-evaluation of the release model

The release model was re-evaluated on the 806-tile one-year test array (`fuelmap_test_1y.npz`). The test set has
146,820,802 labelled pixels; unlabelled pixels are ignored. The commands are in
[`PIPELINE.md`](PIPELINE.md#5-reproduce-the-release-model-accuracy).

| Decision rule | OA | mIoU | mean F1 | mean UA | mean PA |
|---|---|---|---|---|---|
| Paper rule | 87.00% | 73.53% | 84.61% | 82.52% | 87.24% |
| Argmax | 85.96% | 71.96% | 83.54% | 80.51% | 87.69% |

| Paper rule | Other | Tree | Shrub | Building |
|---|---|---|---|---|
| UA (precision) | 95.92% | 78.82% | 74.52% | 80.83% |
| PA (recall) | 86.68% | 87.68% | 87.19% | 87.40% |
| F1 | 91.06% | 83.01% | 80.36% | 83.99% |
| IoU | 83.60% | 70.96% | 67.16% | 72.40% |

Confusion matrix, paper rule (pixels; rows = reference, columns = prediction):

| Reference \ Prediction | Other | Tree | Shrub | Building |
|---|---:|---:|---:|---:|
| Other | 73,078,865 | 4,843,470 | 4,473,584 | 1,915,231 |
| Tree | 750,452 | 25,097,725 | 2,722,000 | 55,110 |
| Shrub | 1,337,705 | 1,740,436 | 21,139,085 | 27,658 |
| Building | 1,020,475 | 160,747 | 33,039 | 8,425,220 |

How to read these numbers:

- The test array is the June 2024 cut of the paper's test split (806 tiles, against 782 in the paper).
- 275 of the tiles are urban-sample tiles with building-only labels. They contribute 7.6 M of the 9.6 M reference
  building pixels and no pixels of the other classes.
  - Building recall therefore mostly reflects these urban tiles.
  - Building false positives can only be counted in the full-label tiles, so building precision and F1 are higher than
    they would be with full labels everywhere.
  - The paper's final evaluation used full labels for these tiles.
- The checkpoint was selected on test-split Dice and mIoU, so these scores are not fully independent of checkpoint
  selection.

### Stratified results of the release model

These results use the paper rule on full-label tiles only (531 tiles; building-only tiles excluded). The strata come
from [`data/test_array_index.csv`](../data/test_array_index.csv). Its row-to-tile mapping was recovered after the fact,
and 796 of 806 rows are matched with high confidence. mIoU and mean F1 are unweighted means over classes 0–3; only a
class absent from both the reference and the prediction of a stratum is skipped (a class that is predicted but absent
from the reference counts as 0). These tile sets are smaller than and different from the paper's, so compare them
with Table 1 only loosely.

**By WUI class**

| WUI class | Tiles | OA | mIoU | mean F1 | F1 Tree | F1 Shrub | F1 Building | F1 Other |
|---|---:|---|---|---|---|---|---|---|
| All full-label tiles | 531 | 87.0% | 66.4% | 78.8% | 83.2% | 80.4% | 60.1% | 91.5% |
| Urban | 15 | 79.0% | 49.6% | 63.1% | 74.4% | 27.2% | 65.3% | 85.7% |
| WUI interface | 58 | 84.0% | 59.5% | 73.3% | 82.7% | 58.1% | 62.8% | 89.4% |
| WUI intermix | 86 | 87.9% | 59.1% | 71.7% | 87.3% | 56.8% | 50.6% | 92.1% |
| Wildland | 370 | 87.5% | 62.1% | 73.6% | 82.1% | 82.4% | 38.1% | 92.0% |

The total row includes 2 full-label tiles whose WUI class is unknown; they are in none of the class rows.

**By Regional Resource Kit region**

| RRK region | Tiles | OA | mIoU | mean F1 | F1 Tree | F1 Shrub | F1 Building | F1 Other |
|---|---:|---|---|---|---|---|---|---|
| Northern California | 57 | 83.9% | 62.1% | 75.5% | 88.2% | 59.5% | 68.1% | 86.0% |
| Central California | 33 | 82.1% | 58.9% | 72.2% | 86.8% | 74.6% | 44.6% | 82.8% |
| Sierra Nevada | 126 | 85.2% | 59.5% | 73.1% | 86.2% | 61.3% | 57.3% | 87.5% |
| Southern California | 226 | 87.4% | 63.4% | 76.3% | 65.0% | 84.4% | 63.2% | 92.6% |
| Outside the four regions ("Limited mapping") | 62 | 94.0% | 49.2% | 62.5% | 51.9% | 50.3% | 50.8% | 97.0% |
| Outside the four regions ("Not Mapped") | 26 | 87.5% | 56.8% | 70.1% | 81.1% | 50.1% | 57.1% | 92.3% |

The rows add up to 530 tiles; 1 full-label tile has no RRK region and is in none of them.

Among the four RRK regions, the pattern is the same as in the paper. Tree F1 is lowest in Southern California, and
shrub F1 is lowest in Northern California and highest in Southern California.

## Limitations

- **Shrubs in urban and WUI areas.** These are the weakest cases. Paper shrub F1 is 14.2 % in urban, 45.1 % in WUI
  interface and 51.4 % in WUI intermix tiles. The release model shows the same pattern (27.2 %, 58.1 % and 56.8 % on
  full-label tiles).
- **Buildings in wildland.** Buildings are rare there, so a few false positives lower building F1 (62.7 % in wildland
  in the paper).
- **NAIP acquisition differences.**
  - NAIP has no rigorous radiometric calibration. Acquisition dates, sun angles, sensors and processing differ between
    years, states of vegetation and flight lines.
  - Accuracy varies by year: paper OA is 85.0–88.1 % across 2016–2022.
  - Radiometric differences can leave discontinuities along scene boundaries.
  - Shadows (especially on steep terrain) cause vegetation to be underestimated, and illumination differences add
    uncertainty (paper §4.3).
- **Misregistration between years.** NAIP georeferencing differs slightly between acquisitions (paper §4.3), so
  per-pixel differences between years include false change along object edges. Consider aggregating (e.g. to 30 m)
  before analysing change.
- **Label noise.**
  - Labels come from LiDAR segmentation, a manual check and an NDVI threshold. Merged or split crowns, missed small
    objects and an imperfect manual check all leave errors.
  - The NAIP year can differ from the LiDAR year. Labels copied to other NAIP years were updated only through the NDVI
    mask, so real structural change between the LiDAR and NAIP dates (fire, logging, construction) is not reflected.
  - The tree/shrub split is a fixed 5 m height (95th percentile) rule, so tall shrubs and short trees are assigned by
    height.
- **Pixel-level output.** Adjacent crowns and buildings merge. The output gives no object instances.
- **Training details.**
  - The release model was trained with unlabelled pixels counted as other.
  - Its checkpoint was selected on the test split.
  - It carries an unused fifth output channel. Because the softmax runs over all five outputs, the thresholds of the
    decision rule are tied to this model. Re-tune them if you train a 4-output model.

## Differences between the paper text and the release model

| Item | Paper text | Release model / code |
|---|---|---|
| Output channels | 4 classes | 5 output channels; channel 4 (legacy "unlabeled") is never a target and is dropped at inference |
| Epochs | 100 | 50; epoch 47 released |
| Optimizer settings | RMSprop; learning rate not reported | RMSprop, lr 1e-8 (constant in practice), weight decay 1e-6, momentum 0.999, gradient clipping 1.0, seed 2024 |
| Scaling augmentation | scaling by ±20 %, p = 0.2 | nearest-neighbour resample round trip with a factor in [0.9, 1.11], p = 0.2. It does not change the field of view |
| Rotation fill | not described | rotated-in corners labelled other (`rotate_label_fill: 0`); 255 (ignore) is recommended for new training |
| Model selection | not described | the last epoch at which test-split mean Dice and mIoU both improved |
| Unlabelled pixels | not described | value 4 reached the loss as class 0 (other). The archived loader one-hot encoded classes 0–3 plus value 5 as the ignored channel, so 4 produced an all-zero vector that `argmax` turned into 0 |
| Inference decision | not described | thresholds 0.4 / 0.4 / 0.6 / 0.5 with an NDVI > 0.11 gate for tree and shrub (see [Decision rule](#decision-rule)) |
| Large-area inference | not described | 1,666 px windows, 25 % overlap, linear-ramp blending, zero-padded edge windows (`scripts/predict_naip.py`) |
| Input scaling | not described | DN / 255, no radiometric normalization |
| Data counts | 7,015 training tiles / 10,086 training tile-years; 782 testing tiles | 10,969 training tile-years; 806 test tiles (June 2024 arrays from the same split) |
| Fig. 3 decoder | last block labelled 64-32-32 | standard U-Net decoder ending in 64 channels; Fig. 3 is schematic |

## Citation

```bibtex
@article{Zhu2026RSE,
  author  = {Zhu, Yunzhe and Dixon, Dan J. and Jin, Yufang},
  title   = {Scalable sub-meter mapping of woody vegetation and structures across {California}'s heterogeneous landscape using deep learning},
  journal = {Remote Sensing of Environment},
  volume  = {333},
  pages   = {115091},
  year    = {2026},
  doi     = {10.1016/j.rse.2025.115091}
}
```
