#!/usr/bin/env python
# Copyright (C) 2023-2026 Yunzhe Zhu
# SPDX-License-Identifier: GPL-3.0-only
"""Map trees, shrubs and buildings on NAIP rasters of any size (GeoTIFF, COG or VRT).

The raster is read in strips and predicted with a sliding window (default
1666 px windows with 25 % overlap and linear-ramp blending, as in the original
pipeline), so memory depends on the raster width, not its size: about 45 KB
per raster column (65 KB with --probs) plus the model. For very wide mosaics,
predict scene by scene and mosaic afterwards (scripts/mosaic_cog.py).

  python scripts/predict_naip.py --weights fuelmap_unet_v1.pth \
      --input m_3912127_ne_10_060_20180718.tif --output m_3912127_ne_10_060_20180718_class.tif

  # every *.tif in a folder, skipping finished outputs, with probabilities
  python scripts/predict_naip.py --weights fuelmap_unet_v1.pth --input naip_2020/ --output pred_2020/ \
      --skip-existing --probs

  # a 2000 x 2000 px window (column offset, row offset, width, height)
  python scripts/predict_naip.py --weights fuelmap_unet_v1.pth --input scene.tif --output test.tif \
      --window 4000 5000 2000 2000

Outputs: <stem>_class.tif, uint8 (0 other, 1 tree, 2 shrub, 3 building,
255 nodata where all input bands are 0), tiled, DEFLATE, same CRS and grid as
the input; with --probs also <stem>_prob.tif, 4 bands uint8 = round(p * 255)
for other/tree/shrub/building (0 where nodata). Input files are only read;
outputs are written as <name>.part.tif and renamed when complete, so an
interrupted run leaves a .part.tif behind and --skip-existing redoes that file.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fuelmap import CLASS_NAMES, NODATA, NUM_CLASSES  # noqa: E402
from fuelmap.inference import OVERLAP, TILE_SIZE, predict_large  # noqa: E402
from fuelmap.model import load_model  # noqa: E402
from fuelmap.utils import add_rule_args, apply_config, class_map_profile, resolve_device, rule_kwargs, write_colormap  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", help="YAML file whose 'inference' section provides defaults")
    p.add_argument("--weights", help="model weights (.pth)")
    p.add_argument("--input", nargs="+", help="NAIP raster(s) and/or folder(s) of rasters")
    p.add_argument("--pattern", default="*.tif", help="file pattern for input folders")
    p.add_argument("--output", help="output .tif (single input) or output folder")
    p.add_argument("--suffix", default="_class.tif", help="output name suffix when writing to a folder")
    p.add_argument("--bands", type=int, nargs=4, default=[1, 2, 3, 4], metavar=("R", "G", "B", "NIR"),
                   help="1-based band indexes of red, green, blue and NIR")
    p.add_argument("--window", type=int, nargs=4, metavar=("COL", "ROW", "WIDTH", "HEIGHT"),
                   help="predict only this pixel window of the input")
    p.add_argument("--probs", action="store_true", help="also write <stem>_prob.tif (uint8 p * 255)")
    p.add_argument("--tile-size", type=int, default=TILE_SIZE, help="sliding-window size in pixels")
    p.add_argument("--overlap", type=float, default=OVERLAP, help="fractional overlap of windows")
    p.add_argument("--blend", choices=["ramp", "mean"], default="ramp", help="how overlapping windows are merged")
    p.add_argument("--batch-size", type=int, default=1, help="windows per forward pass")
    p.add_argument("--no-mask-nodata", action="store_true", help="do not set all-zero pixels to 255")
    p.add_argument("--skip-existing", action="store_true", help="skip inputs whose output exists")
    p.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:1, ...")
    p.add_argument("--threads", type=int, default=None, help="torch CPU threads")
    add_rule_args(p)
    args = apply_config(p, argv, section="inference", strict=False)
    if not args.weights or not args.input or not args.output:
        p.error("--weights, --input and --output are required")
    return args


def list_inputs(items, pattern):
    files = []
    for item in items:
        path = Path(item)
        files += sorted(path.glob(pattern)) if path.is_dir() else [path]
    return files


class AlignedWriter:
    """Write row strips to a tiled GeoTIFF in whole block rows (avoids rewriting compressed blocks)."""

    def __init__(self, dst, block: int):
        self.dst, self.block, self.y, self.buf = dst, block, 0, []

    def add(self, data: np.ndarray):
        from rasterio.windows import Window

        self.buf.append(data)
        n = sum(a.shape[1] for a in self.buf)
        if n >= self.block:
            data = np.concatenate(self.buf, axis=1)
            k = n // self.block * self.block
            self.dst.write(data[:, :k], window=Window(0, self.y, data.shape[2], k))
            self.y += k
            self.buf = [data[:, k:]] if k < n else []

    def close(self):
        from rasterio.windows import Window

        if self.buf:
            data = np.concatenate(self.buf, axis=1)
            self.dst.write(data, window=Window(0, self.y, data.shape[2], data.shape[1]))
            self.y += data.shape[1]
            self.buf = []


def predict_file(model, src_path: Path, out_path: Path, args) -> dict:
    import rasterio
    from rasterio.windows import Window

    with rasterio.open(src_path) as src:
        if max(args.bands) > src.count:
            raise ValueError(f"{src_path} has {src.count} bands; --bands {args.bands} not available")
        if args.window:
            win = Window(*args.window).intersection(Window(0, 0, src.width, src.height))
        else:
            win = Window(0, 0, src.width, src.height)
        col0, row0, width, height = int(win.col_off), int(win.row_off), int(win.width), int(win.height)
        profile = dict(src.profile, width=width, height=height, transform=src.window_transform(win))
        nodata_in = src.nodata

        def read_rows(y0, y1):
            arr = src.read(args.bands, window=Window(col0, row0 + y0, width, y1 - y0))
            if nodata_in is not None and nodata_in != 0:  # treat the declared nodata like all-zero pixels
                arr[:, (arr == nodata_in).all(axis=0)] = 0
            return arr

        cls_prof = class_map_profile(profile)
        tmp_cls = out_path.with_name(out_path.stem + ".part.tif")
        prob_path = out_path.with_name(out_path.name.replace(args.suffix, "") + "_prob.tif") \
            if out_path.name.endswith(args.suffix) else out_path.with_name(out_path.stem + "_prob.tif")
        tmp_prob = prob_path.with_name(prob_path.stem + ".part.tif")
        counts = np.zeros(256, np.int64)
        block = cls_prof.get("blockysize", 1) if cls_prof.get("tiled") else 1
        with rasterio.open(tmp_cls, "w", **cls_prof) as dst_c:
            write_colormap(dst_c)
            dst_p = rasterio.open(tmp_prob, "w", **class_map_profile(profile, NUM_CLASSES, None)) if args.probs else None
            try:
                wc = AlignedWriter(dst_c, block)
                wp = AlignedWriter(dst_p, block) if dst_p else None

                def write_rows(y0, classes, probs):
                    counts[:] += np.bincount(classes.ravel(), minlength=256)
                    wc.add(classes[None])
                    if wp is not None:
                        wp.add(np.clip(np.rint(probs * 255.0), 0, 255).astype(np.uint8))

                predict_large(model, read_rows, height, width, write_rows, tile=args.tile_size, overlap=args.overlap,
                              batch_size=args.batch_size, blend=args.blend, mask_nodata=not args.no_mask_nodata,
                              with_probs=args.probs, progress=True, **rule_kwargs(args))
                wc.close()
                if wp is not None:
                    wp.close()
                    dst_p.descriptions = tuple(f"p_{n}" for n in CLASS_NAMES)
            finally:
                if dst_p is not None:
                    dst_p.close()
    os.replace(tmp_cls, out_path)
    if args.probs:
        os.replace(tmp_prob, prob_path)
    return {"width": width, "height": height, "counts": counts, "prob": prob_path if args.probs else None}


def main(argv=None):
    args = parse_args(argv)
    if args.threads:
        torch.set_num_threads(args.threads)
    files = list_inputs(args.input, args.pattern)
    if not files:
        sys.exit("no input rasters found")
    out = Path(args.output)
    single = len(files) == 1 and out.suffix.lower() in (".tif", ".tiff")
    if not single:
        out.mkdir(parents=True, exist_ok=True)
    else:
        out.parent.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    model = load_model(args.weights, device=device)
    print(f"model: {args.weights} ({model.n_classes} outputs), device {device}, rule {args.rule}, "
          f"tile {args.tile_size}, overlap {args.overlap}, blend {args.blend}")
    failed = []
    for n, f in enumerate(files, 1):
        dst = out if single else out / (f.stem + args.suffix)
        if args.skip_existing and dst.exists():
            print(f"[{n}/{len(files)}] skip {f.name} (output exists)")
            continue
        t0 = time.time()
        try:
            info = predict_file(model, f, dst, args)
        except Exception as err:  # keep going; inputs are never modified or removed
            print(f"[{n}/{len(files)}] FAILED {f}: {err}", file=sys.stderr)
            failed.append(str(f))
            continue
        c = info["counts"]
        total = c.sum()
        share = ", ".join(f"{name} {c[k] / total:.3f}" for k, name in enumerate(CLASS_NAMES))
        print(f"[{n}/{len(files)}] {f.name}: {info['width']} x {info['height']} px in {time.time() - t0:.1f} s; "
              f"{share}; nodata {c[NODATA] / total:.3f} -> {dst}" + (f" (+ {info['prob']})" if info["prob"] else ""))
    if failed:
        print(f"{len(failed)} file(s) failed: {failed}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
