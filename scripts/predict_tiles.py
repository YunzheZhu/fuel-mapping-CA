#!/usr/bin/env python
# Copyright (C) 2023-2026 Yunzhe Zhu
# SPDX-License-Identifier: GPL-3.0-only
"""Predict whole tiles (e.g. the 512 x 512 px training/test tiles) in one forward pass each.

Inputs are either a folder of GeoTIFF tiles or an npz array of tiles:

  # GeoTIFF tiles -> <output>/<stem>_class.tif (+ <stem>_prob.tif)
  python scripts/predict_tiles.py --weights fuelmap_unet_v1.pth --images tiles/ --output pred/

  # npz array (N, 512, 512, 4) -> <output>/pred_class.npy (N, 512, 512) + tile_index.npy
  python scripts/predict_tiles.py --weights fuelmap_unet_v1.pth \
      --npz release/fuelmap_test_1y.npz --npz-key images --start 0 --count 8 --output pred/

Class maps are uint8 (0 other, 1 tree, 2 shrub, 3 building; 255 nodata if
--mask-nodata). Probabilities are saved as uint8 = round(p * 255).
For rasters larger than a tile use scripts/predict_naip.py.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fuelmap import CLASS_NAMES, NODATA  # noqa: E402
from fuelmap.dataset import TileDataset, read_raster  # noqa: E402
from fuelmap.inference import predict_batch  # noqa: E402
from fuelmap.model import load_model  # noqa: E402
from fuelmap.utils import add_rule_args, apply_config, class_map_profile, resolve_device, rule_kwargs, write_colormap  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", help="YAML file whose 'inference' section provides defaults")
    p.add_argument("--weights", help="model weights (.pth)")
    src = p.add_mutually_exclusive_group()
    src.add_argument("--images", help="folder of GeoTIFF tiles (bands R, G, B, NIR)")
    src.add_argument("--npz", help="npz file with an (N, H, W, 4) uint8 tile array")
    p.add_argument("--pattern", default="*.tif", help="file pattern inside --images")
    p.add_argument("--npz-key", default=None, help="array name inside --npz (default arr_0 / the only member)")
    p.add_argument("--start", type=int, default=0, help="first npz tile to predict")
    p.add_argument("--count", type=int, default=None, help="number of npz tiles (default: all)")
    p.add_argument("--output", help="output folder")
    p.add_argument("--save-probs", action="store_true", help="also save class probabilities as uint8 (p * 255)")
    p.add_argument("--mask-nodata", action="store_true",
                   help="set pixels whose four bands are all 0 to 255 (the paper's tile evaluation did not)")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:1, ...")
    p.add_argument("--threads", type=int, default=None, help="torch CPU threads")
    add_rule_args(p)
    args = apply_config(p, argv, section="inference", strict=False,
                        ignore=("tile_size", "overlap", "blend"))  # sliding-window keys of predict_naip.py
    if not args.weights or not args.output or not (args.images or args.npz):
        p.error("--weights, --output and one of --images/--npz are required")
    return args


def to_uint8_probs(probs: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(probs * 255.0), 0, 255).astype(np.uint8)


def main(argv=None):
    args = parse_args(argv)
    if args.threads:
        torch.set_num_threads(args.threads)
    device = resolve_device(args.device)
    model = load_model(args.weights, device=device)
    print(f"model: {args.weights} ({model.n_classes} outputs), device {device}, rule {args.rule}")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    kw = rule_kwargs(args)
    t0 = time.time()

    if args.npz:
        ds = TileDataset.from_npz(args.npz, image_key=args.npz_key)
        stop = len(ds) if args.count is None else min(len(ds), args.start + args.count)
        index = np.arange(args.start, stop)
        loader = DataLoader(Subset(ds, index.tolist()), batch_size=args.batch_size, shuffle=False)
        classes, probs = [], []
        for x in loader:
            c, pr = predict_batch(model, x, mask_nodata=args.mask_nodata, **kw)
            classes.append(c)
            if args.save_probs:
                probs.append(to_uint8_probs(pr))
            print(f"  {sum(len(a) for a in classes)}/{len(index)} tiles", flush=True)
        classes = np.concatenate(classes)
        np.save(out / "pred_class.npy", classes)
        np.save(out / "tile_index.npy", index.astype(np.int64))
        if args.save_probs:
            np.save(out / "pred_prob.npy", np.concatenate(probs))
        written = [out / "pred_class.npy", out / "tile_index.npy"]
    else:
        import rasterio

        files = sorted(Path(args.images).glob(args.pattern))
        if not files:
            sys.exit(f"no files matching {args.pattern} in {args.images}")
        written, counts = [], np.zeros(256, np.int64)
        for i in range(0, len(files), args.batch_size):
            batch = files[i:i + args.batch_size]
            arrays = [read_raster(f) for f in batch]
            shapes = {a.shape[1:] for a, _ in arrays}
            if len(shapes) > 1:  # different tile sizes: predict one by one
                results = [predict_batch(model, torch.from_numpy(a[None, :4].astype(np.float32) / np.float32(255)),
                                         mask_nodata=args.mask_nodata, **kw) for a, _ in arrays]
                results = [(c[0], p[0]) for c, p in results]
            else:
                x = torch.from_numpy(np.stack([a[:4] for a, _ in arrays]).astype(np.float32) / np.float32(255))
                c, pr = predict_batch(model, x, mask_nodata=args.mask_nodata, **kw)
                results = list(zip(c, pr))
            for f, (_, prof), (c, pr) in zip(batch, arrays, results):
                dst_path = out / f"{f.stem}_class.tif"
                with rasterio.open(dst_path, "w", **class_map_profile(prof)) as dst:
                    dst.write(c, 1)
                    write_colormap(dst)
                written.append(dst_path)
                counts += np.bincount(c.ravel(), minlength=256)
                if args.save_probs:
                    with rasterio.open(out / f"{f.stem}_prob.tif", "w", **class_map_profile(prof, 4, None)) as dst:
                        dst.write(to_uint8_probs(pr))
                        dst.descriptions = tuple(f"p_{n}" for n in CLASS_NAMES)
            print(f"  {min(i + args.batch_size, len(files))}/{len(files)} tiles", flush=True)
        classes = None
    if classes is not None:
        counts = np.bincount(classes.ravel(), minlength=256)
    total = counts.sum()
    share = ", ".join(f"{n} {counts[k] / total:.3f}" for k, n in enumerate(CLASS_NAMES))
    print(f"done in {time.time() - t0:.1f} s; class shares: {share}; nodata {counts[NODATA] / total:.3f}")
    print("wrote " + (", ".join(str(w) for w in written) if len(written) <= 4 else f"{len(written)} files to {out}"))


if __name__ == "__main__":
    main()
