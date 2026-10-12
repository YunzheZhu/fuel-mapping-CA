#!/usr/bin/env python
# Copyright (C) 2023-2026 Yunzhe Zhu
# Portions Copyright (C) Alexandre Milesi (milesial) and the Pytorch-UNet contributors
# SPDX-License-Identifier: GPL-3.0-only
#
# Training loop derived from milesial/Pytorch-UNet train.py
# (https://github.com/milesial/Pytorch-UNet), GPL-3.0.
# Modified by Yunzhe Zhu, 2023-2026: npz/GeoTIFF tile datasets, augmentation,
# class-weighted loss with an ignore value, evaluation with the paper's decision
# rule, model selection, YAML configs, multi-GPU and fine-tuning options.
"""Train the fuelmap U-Net on 512 x 512 px NAIP tiles.

Data are npz arrays (images (N, 512, 512, 4) uint8 R, G, B, NIR; labels
(N, 512, 512) uint8) or folders of GeoTIFF tiles. Defaults follow the paper and
the released model (weighted cross-entropy 0.4/1.0/1.1/1.5, RMSprop, batch 20,
flips/rotation/scale-jitter augmentation); the exact settings of the release
model are in configs/release_v1.yaml. With the release packages converted to
uncompressed files (docs/PIPELINE.md, section 3):

  python scripts/train.py \
      --train-images release/fuelmap_train_4y_part0*of06_images.npz \
      --train-labels release/fuelmap_train_4y_part0*of06_labels.npz \
      --val-images release/fuelmap_val_4y_images.npz --val-labels release/fuelmap_val_4y_labels.npz \
      --out-dir runs/my_model

--init-weights fine-tunes from existing weights (e.g. fuelmap_unet_v1.pth). If
their head size differs from --n-outputs, the shared output channels are
copied and the others keep their fresh initialisation (with the release
weights and the default 4 outputs, the legacy fifth channel is dropped).

Each epoch the model is evaluated on the validation set (and on the test set,
if given) with the class decision rule of the paper, and the checkpoint with
the best selection metric is saved as best.pth (plain state_dict) with
best.json (metadata). last.pth is overwritten every epoch; history.csv holds
the per-epoch metrics. Logging to Weights & Biases is optional (--wandb).
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fuelmap import BAND_ORDER, CLASS_NAMES, IGNORE_INDEX, NUM_CLASSES, __version__  # noqa: E402
from fuelmap.augment import Augment  # noqa: E402
from fuelmap.dataset import build_dataset  # noqa: E402
from fuelmap.inference import NDVI_MIN, classify  # noqa: E402
from fuelmap.losses import CLASS_WEIGHTS, weighted_cross_entropy  # noqa: E402
from fuelmap.metrics import confusion_matrix, metrics_from_cm  # noqa: E402
from fuelmap.model import UNet, read_state_dict  # noqa: E402
from fuelmap.utils import apply_config, resolve_device, sha256_file  # noqa: E402

log = logging.getLogger("train")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", help="YAML file whose 'train' section provides defaults")
    g = p.add_argument_group("data (npz files or one GeoTIFF folder each)")
    g.add_argument("--train-images", nargs="+")
    g.add_argument("--train-labels", nargs="+")
    g.add_argument("--val-images", nargs="+")
    g.add_argument("--val-labels", nargs="+")
    g.add_argument("--test-images", nargs="+", help="optional test set, evaluated every epoch for reporting")
    g.add_argument("--test-labels", nargs="+")
    g.add_argument("--image-key", default=None, help="array name inside image npz files (default arr_0)")
    g.add_argument("--label-key", default=None, help="array name inside label npz files (default arr_0)")
    g.add_argument("--pattern", default="*.tif", help="GeoTIFF folders: image file pattern")
    g.add_argument("--label-template", default="{stem}.tif", help="GeoTIFF folders: label name from the image stem")
    g.add_argument("--stem-regex", default=None, help="GeoTIFF folders: regex extracting the key from the stem")
    g.add_argument("--unlabeled-value", type=int, default=None,
                   help="label value meaning 'no label' (255 in the release packages; the original 2024 arrays "
                        "used 4 and the 2023 arrays 5)")
    g.add_argument("--unlabeled-as", choices=["ignore", "other"], default="ignore",
                   help="treat 'no label' pixels as ignored (255) or as class 0 'other'")
    g = p.add_argument_group("model")
    g.add_argument("--n-outputs", type=int, choices=[4, 5], default=4,
                   help="output channels; 5 reproduces the legacy head of the released model")
    g.add_argument("--init-weights", help="start from these weights (fine-tuning); a head of another size is "
                                          "adapted to --n-outputs")
    g = p.add_argument_group("optimisation")
    g.add_argument("--epochs", type=int, default=100)
    g.add_argument("--batch-size", type=int, default=20)
    g.add_argument("--lr", type=float, default=1e-8, help="RMSprop learning rate")
    g.add_argument("--weight-decay", type=float, default=1e-6)
    g.add_argument("--momentum", type=float, default=0.999)
    g.add_argument("--grad-clip", type=float, default=1.0, help="max gradient norm (0 disables)")
    g.add_argument("--class-weights", type=float, nargs=4, default=list(CLASS_WEIGHTS),
                   metavar=("OTHER", "TREE", "SHRUB", "BUILDING"))
    g.add_argument("--scheduler", choices=["none", "plateau"], default="none",
                   help="'plateau': reduce LR when the selection metric stops improving")
    g.add_argument("--plateau-patience", type=int, default=5)
    g.add_argument("--plateau-factor", type=float, default=0.1)
    g.add_argument("--amp", action="store_true", help="mixed precision (CUDA only)")
    g = p.add_argument_group("augmentation (training set only)")
    g.add_argument("--no-augment", action="store_true")
    g.add_argument("--hflip-p", type=float, default=0.5)
    g.add_argument("--vflip-p", type=float, default=0.5)
    g.add_argument("--rotate-p", type=float, default=0.2)
    g.add_argument("--max-rotation", type=float, default=45.0, help="degrees")
    g.add_argument("--scale-p", type=float, default=0.2)
    g.add_argument("--scale-range", type=float, nargs=2, default=[0.9, 1.11], metavar=("MIN", "MAX"),
                   help="factor range of the resample round trip")
    g.add_argument("--rotate-label-fill", type=int, default=IGNORE_INDEX,
                   help="label of pixels rotated in from outside the tile (255 ignore; 0 = original code)")
    g = p.add_argument_group("evaluation and model selection")
    g.add_argument("--select-on", choices=["val", "test"], default="val",
                   help="split used to pick best.pth ('test' reproduces the legacy procedure)")
    g.add_argument("--select-metric", choices=["miou", "f1_and_miou"], default="miou",
                   help="'f1_and_miou': new best only if mean F1 and mIoU both improve (legacy)")
    g.add_argument("--eval-rule", choices=["paper", "argmax"], default="paper")
    g.add_argument("--eval-ndvi-min", type=float, default=NDVI_MIN,
                   help="NDVI gate of the paper rule during evaluation (negative disables)")
    g = p.add_argument_group("runtime")
    g.add_argument("--out-dir", help="folder for checkpoints and logs")
    g.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:1, ...")
    g.add_argument("--data-parallel", action="store_true", help="use all visible GPUs (nn.DataParallel)")
    g.add_argument("--workers", type=int, default=4, help="DataLoader worker processes")
    g.add_argument("--seed", type=int, default=2024)
    g.add_argument("--save-every", type=int, default=0, help="also keep epochNNN.pth every N epochs")
    g.add_argument("--limit-train-batches", type=int, default=0, help="debug: batches per epoch (0 = all)")
    g.add_argument("--limit-eval-batches", type=int, default=0, help="debug: evaluation batches (0 = all)")
    g.add_argument("--wandb", action="store_true", help="log to Weights & Biases (requires wandb)")
    g.add_argument("--wandb-project", default="fuelmap")
    g.add_argument("--run-name", default=None)
    args = apply_config(p, argv, section="train")
    missing = [n for n in ("train_images", "train_labels", "out_dir") if not getattr(args, n)]
    if missing:
        p.error("missing required option(s): " + ", ".join("--" + m.replace("_", "-") for m in missing))
    if args.select_on == "val" and not args.val_images and args.test_images:
        log.warning("no validation set: model selection falls back to the test set")
    return args


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model, loader, criterion, device, rule, ndvi_min, limit=0):
    """Mean loss and accuracy metrics (paper definitions) over a loader."""
    model.eval()
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), np.int64)
    loss_sum, n_loss = 0.0, 0
    for i, (x, y) in enumerate(loader):
        if limit and i >= limit:
            break
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(x)
        loss = criterion(logits.float(), y)
        if torch.isfinite(loss):  # NaN if a batch has no labelled pixel
            loss_sum, n_loss = loss_sum + loss.item(), n_loss + 1
        probs = torch.softmax(logits.float(), dim=1)[:, :NUM_CLASSES]
        cm += confusion_matrix(y, classify(probs, x, rule, None, ndvi_min))
    return loss_sum / max(n_loss, 1), metrics_from_cm(cm)


def init_weights(model: UNet, path) -> None:
    """Load ``path`` into ``model``, adapting an output head of another size.

    Output channels present in both heads are copied; extra channels of the
    model keep their fresh initialisation, extra channels of the file are dropped.
    """
    state = read_state_dict(path)
    head = model.outc.conv
    n_file = int(state["outc.conv.weight"].shape[0])
    if n_file != head.out_channels:
        k = min(n_file, head.out_channels)
        for name in ("weight", "bias"):
            t = getattr(head, name).detach().clone()
            t[:k] = state[f"outc.conv.{name}"][:k]
            state[f"outc.conv.{name}"] = t
        log.warning("%s has %d output channels; copied %d of them into the %d-output head",
                    path, n_file, k, head.out_channels)
    model.load_state_dict(state)
    log.info("initialised from %s", path)


def save_weights(model: nn.Module, path: Path):
    core = model.module if isinstance(model, nn.DataParallel) else model
    torch.save({k: v.detach().cpu() for k, v in core.state_dict().items()}, str(path))


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args(argv)
    seed_everything(args.seed)
    device = resolve_device(args.device)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=1)

    # data
    augment = None if args.no_augment else Augment(
        args.hflip_p, args.vflip_p, args.rotate_p, args.max_rotation, args.scale_p, tuple(args.scale_range),
        label_fill=args.rotate_label_fill)
    ds_kw = dict(unlabeled_value=args.unlabeled_value, unlabeled_as=IGNORE_INDEX if args.unlabeled_as == "ignore" else 0,
                 image_key=args.image_key, label_key=args.label_key, pattern=args.pattern,
                 label_template=args.label_template, stem_regex=args.stem_regex)
    train_ds = build_dataset(args.train_images, args.train_labels, transform=augment, **ds_kw)
    eval_sets = {}
    for split in ("val", "test"):
        imgs, labs = getattr(args, f"{split}_images"), getattr(args, f"{split}_labels")
        if imgs:
            eval_sets[split] = build_dataset(imgs, labs, **ds_kw)
    select_on = args.select_on if args.select_on in eval_sets else next(iter(eval_sets), None)
    log.info("tiles: train %d, %s; augmentation: %s", len(train_ds),
             ", ".join(f"{k} {len(v)}" for k, v in eval_sets.items()) or "no evaluation set", augment)
    loader_kw = dict(num_workers=args.workers, pin_memory=device.type == "cuda",
                     persistent_workers=args.workers > 0)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False,
                              generator=torch.Generator().manual_seed(args.seed), **loader_kw)
    eval_loaders = {k: DataLoader(v, batch_size=args.batch_size, shuffle=False, **loader_kw)
                    for k, v in eval_sets.items()}

    # model, loss, optimiser
    model = UNet(n_channels=4, n_classes=args.n_outputs)
    if args.init_weights:
        init_weights(model, args.init_weights)
    model.to(device)
    if args.data_parallel and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        log.info("DataParallel on %d GPUs", torch.cuda.device_count())
    criterion = weighted_cross_entropy(args.class_weights, args.n_outputs, IGNORE_INDEX, device)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
                                    momentum=args.momentum, foreach=True)
    scheduler = (torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=args.plateau_factor,
                                                            patience=args.plateau_patience)
                 if args.scheduler == "plateau" else None)
    use_amp = args.amp and device.type == "cuda"
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    except (AttributeError, TypeError):  # torch < 2.3
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    ndvi_min = None if args.eval_ndvi_min is None or args.eval_ndvi_min < 0 else args.eval_ndvi_min

    run = None
    if args.wandb:
        import wandb

        run = wandb.init(project=args.wandb_project, name=args.run_name, config=vars(args))

    history_path = out / "history.csv"
    best = {"epoch": 0, "mIoU": -math.inf, "mean_F1": -math.inf}
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        loss_sum, n_batches = 0.0, 0
        for i, (x, y) in enumerate(train_loader):
            if args.limit_train_batches and i >= args.limit_train_batches:
                break
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, enabled=use_amp):
                loss = criterion(model(x).float(), y)
            if not torch.isfinite(loss):  # batch without labelled pixels
                continue
            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            loss_sum, n_batches = loss_sum + loss.item(), n_batches + 1
        row = {"epoch": epoch, "lr": optimizer.param_groups[0]["lr"], "train_loss": loss_sum / max(n_batches, 1)}
        results = {}
        for split, loader in eval_loaders.items():
            loss, m = evaluate(model, loader, criterion, device, args.eval_rule, ndvi_min, args.limit_eval_batches)
            results[split] = m
            row.update({f"{split}_loss": loss, f"{split}_OA": m["OA"], f"{split}_mIoU": m["mIoU"],
                        f"{split}_mean_F1": m["mean_F1"]})
            row.update({f"{split}_IoU_{c}": m["IoU"][c] for c in CLASS_NAMES})
        row["seconds"] = round(time.time() - t0, 1)

        save_weights(model, out / "last.pth")
        if args.save_every and epoch % args.save_every == 0:
            save_weights(model, out / f"epoch{epoch:03d}.pth")
        sel = results.get(select_on)
        if sel is None:
            improved = True
        elif args.select_metric == "f1_and_miou":
            improved = sel["mIoU"] > best["mIoU"] and sel["mean_F1"] > best["mean_F1"]
        else:
            improved = sel["mIoU"] > best["mIoU"]
        if improved:
            best = {"epoch": epoch, "mIoU": sel["mIoU"] if sel else math.nan,
                    "mean_F1": sel["mean_F1"] if sel else math.nan}
            save_weights(model, out / "best.pth")
            meta = {
                "format": "fuelmap U-Net state_dict", "fuelmap_version": __version__,
                "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "epoch": epoch, "n_channels": 4, "n_outputs": args.n_outputs, "class_names": list(CLASS_NAMES),
                "band_order": list(BAND_ORDER), "input_scaling": "digital number / 255",
                "selected_on": select_on, "select_metric": args.select_metric,
                "metrics": {k: {kk: v[kk] for kk in ("OA", "mIoU", "mean_F1", "F1", "IoU")} for k, v in results.items()},
                "training_tiles": len(train_ds), "config": vars(args),
                "sha256": sha256_file(out / "best.pth"),
            }
            with open(out / "best.json", "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=1)
        if scheduler is not None and sel is not None:
            scheduler.step(sel["mIoU"])

        write_header = not history_path.exists() or epoch == 1
        with open(history_path, "w" if epoch == 1 else "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            if write_header:
                w.writeheader()
            w.writerow(row)
        if run is not None:
            run.log(row)
        summary = " ".join(f"{k} {v:.4g}" if isinstance(v, float) else f"{k} {v}" for k, v in row.items())
        log.info("%s%s", summary, "  [best]" if improved else "")

    log.info("best epoch %d (%s %s mIoU %.4f); weights in %s", best["epoch"], select_on, args.select_metric,
             best["mIoU"], out / "best.pth")
    if run is not None:
        run.finish()


if __name__ == "__main__":
    if os.name == "nt":  # DataLoader workers use spawn on Windows
        torch.multiprocessing.freeze_support()
    main()
