# Copyright (C) 2023-2026 Yunzhe Zhu
# SPDX-License-Identifier: GPL-3.0-only
"""Small helpers shared by the command-line scripts."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Optional, Tuple

import torch

from . import CLASS_COLORS, NODATA

__all__ = ["apply_config", "add_rule_args", "rule_kwargs", "resolve_device", "sha256_file", "class_map_profile",
           "write_colormap", "ensure_parent"]


def apply_config(parser: argparse.ArgumentParser, argv=None, section: Optional[str] = None,
                 strict: bool = True, ignore: Tuple[str, ...] = ()) -> argparse.Namespace:
    """Parse arguments, taking defaults from ``--config FILE.yaml`` when given.

    Keys of the YAML mapping (or of its ``section`` sub-mapping) must match the
    parser's option names (``-`` or ``_`` both accepted). Explicit command-line
    options override the file. With ``strict=False`` keys the parser does not
    know are reported and skipped (for sections shared by several scripts);
    keys listed in ``ignore`` are skipped silently.
    """
    pre, _ = parser.parse_known_args(argv)
    cfg_path = getattr(pre, "config", None)
    if cfg_path:
        import yaml

        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        if section is not None:
            cfg = cfg.get(section, {}) or {}
        actions = {a.dest: a for a in parser._actions}
        values, unused = {}, []
        where = f"{cfg_path}" + (f" (section '{section}')" if section else "")
        for key, value in cfg.items():
            dest = key.replace("-", "_")
            if dest in ignore:
                continue
            if dest not in actions:
                if strict:
                    parser.error(f"unknown key '{key}' in {where}")
                unused.append(key)
                continue
            conv = actions[dest].type
            if callable(conv):  # YAML may give strings such as '1e-8'
                if isinstance(value, str):
                    value = conv(value)
                elif isinstance(value, list):
                    value = [conv(v) if isinstance(v, str) else v for v in value]
            values[dest] = value
        if unused:
            print(f"note: {where}: keys not used by this script: {', '.join(unused)}")
        parser.set_defaults(**values)
    return parser.parse_args(argv)


def add_rule_args(parser: argparse.ArgumentParser) -> None:
    """Add the class-decision-rule options shared by the prediction scripts."""
    from .inference import NDVI_MIN, THRESHOLDS

    g = parser.add_argument_group("class decision rule")
    g.add_argument("--rule", choices=["paper", "argmax"], default="paper",
                   help="'paper': probability thresholds + NDVI gate (used in the paper); 'argmax': plain argmax")
    for name in ("tree", "shrub", "building", "other"):
        g.add_argument(f"--{name}-threshold", type=float, default=THRESHOLDS[name],
                       help=f"probability threshold for {name} (paper rule)")
    g.add_argument("--ndvi-min", type=float, default=NDVI_MIN,
                   help="tree/shrub require NDVI > this value (paper rule)")
    g.add_argument("--no-ndvi", action="store_true", help="disable the NDVI gate")


def rule_kwargs(args: argparse.Namespace) -> dict:
    """Keyword arguments for :func:`fuelmap.inference.classify` from :func:`add_rule_args` options."""
    return dict(rule=args.rule, ndvi_min=None if args.no_ndvi else args.ndvi_min,
                thresholds={n: getattr(args, f"{n}_threshold") for n in ("tree", "shrub", "building", "other")})


def resolve_device(name: str = "auto") -> torch.device:
    """``"auto"`` -> CUDA if available, else CPU; otherwise ``torch.device(name)``."""
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def sha256_file(path, chunk: int = 1 << 22) -> str:
    """SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def class_map_profile(profile: dict, count: int = 1, nodata: Optional[float] = NODATA, blocksize: int = 512) -> dict:
    """GeoTIFF profile for uint8 outputs derived from an input profile: tiled, DEFLATE, nodata 255."""
    out = {k: v for k, v in profile.items() if k in ("crs", "transform", "width", "height")}
    out.update(driver="GTiff", dtype="uint8", count=count, nodata=nodata, compress="deflate", predictor=1,
               tiled=True, blockxsize=blocksize, blockysize=blocksize, BIGTIFF="IF_SAFER", interleave="band")
    if out["width"] < blocksize or out["height"] < blocksize:  # GDAL requires blocks <= raster size if tiled
        out.update(tiled=False, blockxsize=None, blockysize=None)
        out = {k: v for k, v in out.items() if v is not None}
    return out


def write_colormap(dst) -> None:
    """Attach the class colour table (other/tree/shrub/building) to band 1 of an open rasterio dataset."""
    try:
        dst.write_colormap(1, {k: (*rgb, 255) for k, rgb in CLASS_COLORS.items()})
    except Exception:  # colour tables are cosmetic; ignore drivers that do not support them
        pass


def ensure_parent(path) -> Path:
    """Create the parent directory of ``path`` and return it as a Path."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p
