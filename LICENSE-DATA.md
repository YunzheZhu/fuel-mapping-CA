# Data and model license (CC BY 4.0)

The **source code** in this repository is licensed under the GNU General Public License v3.0 (see [`LICENSE`](LICENSE)).

The **data, labels, model weights and map products** listed below are licensed under the
**Creative Commons Attribution 4.0 International License (CC BY 4.0)**:
https://creativecommons.org/licenses/by/4.0/ (legal code: https://creativecommons.org/licenses/by/4.0/legalcode).

Copyright © 2023–2026 Yunzhe Zhu, Dan J. Dixon and Yufang Jin.

## What this covers

- **Repository data:** everything under `data/`. That includes the tile index (`tile_index.gpkg`, `tile_index.csv`), the
  test-array index (`test_array_index.csv`) and the sample NAIP and label tiles (`data/sample/`).
- **Model metadata:** `models/fuelmap_unet_v1.json`.
- **GitHub release `v1.0.0` assets:**
  - the model weights `fuelmap_unet_v1.pth`;
  - the image and label arrays `fuelmap_test_1y.npz`, `fuelmap_val_4y.npz` and `fuelmap_train_4y_part01of06.npz` …
    `part06of06.npz`.
- **Map products:** the 0.6 m class maps distributed through the links in this repository (the Box share and the
  Google Earth Engine app).

The CC BY 4.0 license of the published article (doi:10.1016/j.rse.2025.115091) covers the article. This file is what
licenses the data and weights.

## Attribution

You may share and adapt the material for any purpose, including commercially, if you give appropriate credit. Indicate
whether you made changes, and include a link to the license. Please credit the material as follows:

> Zhu, Y., Dixon, D. J., & Jin, Y. (2026). Scalable sub-meter mapping of woody vegetation and structures across
> California's heterogeneous landscape using deep learning. *Remote Sensing of Environment*, 333, 115091.
> https://doi.org/10.1016/j.rse.2025.115091. Data and model: https://github.com/YunzheZhu/fuel-mapping-CA (v1.0.0),
> licensed under CC BY 4.0.

If you redistribute modified labels or retrained weights, say that they were modified. Do not imply that the authors
endorse your use.

## Third-party sources

The released data are derived from the sources below. CC BY 4.0 covers the authors' contribution; the terms of each
source continue to apply to the information taken from it.

| Source | How it is used here | Notes |
|---|---|---|
| **NAIP imagery** (USDA National Agriculture Imagery Program) | The image tiles in `data/sample/` and the `images` arrays are clips of NAIP 4-band imagery (2016–2022). All model inputs are NAIP | NAIP imagery is in the U.S. public domain. Please credit USDA NAIP |
| **USGS 3D Elevation Program (3DEP) aerial LiDAR** | Used to make the tree, shrub and building labels (object segmentation and heights). No point clouds are redistributed | USGS 3DEP data are in the U.S. public domain. Please credit USGS 3DEP |
| **County building footprints**: Napa County GIS (2016), Shasta County (2018), County of San Luis Obispo (2015) | Rasterised building masks derived from the Napa and Shasta layers are part of the released labels: the building-only labels of the urban-sample tiles from these two counties (`URB_*`, column `footprint_county` of the tile index). Labels derived from the San Luis Obispo layer are **not** distributed (see [below](#county-building-footprints)). Only the footprint geometry was used; the vector layers themselves are not redistributed | Terms and credits per county: see [below](#county-building-footprints) |
| **Microsoft US Building Footprints** | Used only to compute sampling attributes when the tiles were selected | **Not** redistributed. No Microsoft footprint data or attributes derived from them are in this repository or its release assets |
| **CAL FIRE FRAP WUI (2015)**, **U.S. EPA Level III ecoregions**, **Regional Resource Kit regions**, **USGS 3DEP Work Unit Extent Spatial Metadata (WESM)** | The tile index has category names and identifiers taken from these layers: WUI class, ecoregion, RRK region, LiDAR work unit and project | Only names and identifiers are included, not the layers themselves |

### County building footprints

- **Napa County** ([layer](https://www.arcgis.com/home/item.html?id=53c8a567d0e646f8ab4789e5d601eeaf)): the county's
  current public building-footprint layer is published under the Open Data Commons Public Domain Dedication and License
  (PDDL) 1.0. Credit: Napa County GIS.
- **Shasta County** ([layer](https://data-shasta.opendata.arcgis.com/datasets/Shasta::buildingfootprints/about)):
  provided "as is" under the Shasta County GIS Data Disclaimer, which asks users to acknowledge the County of Shasta
  GIS as the source. Credit: County of Shasta GIS.
- **San Luis Obispo County**
  ([layer](https://opendata-slocounty.hub.arcgis.com/datasets/SLOCounty::building-footprints/about)): the paper used
  this layer (a snapshot extract of OpenStreetMap building data, © OpenStreetMap contributors) for the building labels
  of the urban-sample tiles in San Luis Obispo County. Its terms state: "This data may not be given away, sold or
  otherwise distributed to any third party without express written permission from the County of San Luis Obispo."
  **Labels derived from this layer are therefore not distributed.** The 864 urban-sample tiles labelled from it (113
  test and 751 training tiles; `footprint_county = San Luis Obispo`, `in_release_arrays = False` in the tile index) are
  not in the release packages, and their labels are not in this repository. The building labels of the released tiles
  inside San Luis Obispo County (validation and WUI training tiles) come from the LiDAR segmentation, not from this
  layer.

## Disclaimer

As the license states, the material is provided "as is", without warranties of any kind. The maps and labels are
model- and LiDAR-derived products with known errors (see [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md)). They are not
authoritative records of property, vegetation-management compliance or building presence.
