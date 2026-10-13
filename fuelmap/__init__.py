# Copyright (C) 2023-2026 Yunzhe Zhu
# SPDX-License-Identifier: GPL-3.0-only
"""fuelmap: sub-meter mapping of trees, shrubs and buildings from NAIP imagery.

Code for Zhu, Dixon & Jin (2026), "Scalable sub-meter mapping of woody
vegetation and structures across California's heterogeneous landscape using
deep learning", Remote Sensing of Environment 333, 115091.
https://doi.org/10.1016/j.rse.2025.115091

Conventions used throughout the package:

* Input: NAIP 4-band digital numbers (uint8) in band order R, G, B, NIR,
  scaled by 1/255. Training tiles are 512 x 512 px at 0.6 m (307.2 m).
* Classes: 0 = other (background), 1 = tree, 2 = shrub, 3 = building.
* 255 = nodata in products and "ignore" in label files.
"""

__version__ = "1.0.0"

CLASS_NAMES = ("other", "tree", "shrub", "building")
"""Class names in class-id order."""

NUM_CLASSES = len(CLASS_NAMES)

IGNORE_INDEX = 255
"""Label value ignored by the loss and the metrics."""

NODATA = 255
"""Nodata value of predicted class maps."""

BAND_ORDER = ("red", "green", "blue", "nir")
"""Expected input band order."""

CLASS_COLORS = {
    0: (0, 0, 0),  # other
    1: (0, 0, 255),  # tree
    2: (255, 127, 0),  # shrub
    3: (255, 0, 0),  # building
}
"""RGB colours of the paper's class maps (Figs. 4 and 6), written as the GeoTIFF colour table of class maps."""

from .model import UNet, load_model  # noqa: E402  (convenience imports)
from .inference import classify, predict_array, predict_batch  # noqa: E402
