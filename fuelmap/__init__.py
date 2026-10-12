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
    0: (230, 230, 210),  # other
    1: (24, 110, 40),  # tree
    2: (200, 170, 60),  # shrub
    3: (220, 40, 40),  # building
}
"""RGB colours written as the GeoTIFF colour table of class maps."""

from .model import UNet, load_model  # noqa: E402  (convenience imports)
from .inference import classify, predict_array, predict_batch  # noqa: E402
