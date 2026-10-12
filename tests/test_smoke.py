# Copyright (C) 2023-2026 Yunzhe Zhu
# SPDX-License-Identifier: GPL-3.0-only
"""Fast smoke tests (CPU, no data needed). Run with ``pytest tests`` or ``python tests/test_smoke.py``."""
from __future__ import annotations

import gc
import importlib.util
import pickle
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fuelmap import IGNORE_INDEX, NODATA  # noqa: E402
from fuelmap.augment import Augment, rotate, scale_jitter  # noqa: E402
from fuelmap.dataset import TileDataset  # noqa: E402
from fuelmap.inference import classify, ndvi, predict_array, predict_large, predict_probs  # noqa: E402
from fuelmap.losses import weighted_cross_entropy  # noqa: E402
from fuelmap.metrics import confusion_matrix, metrics_from_cm  # noqa: E402
from fuelmap.model import UNet, load_model  # noqa: E402


def test_model_params_and_forward():
    torch.manual_seed(0)
    m5, m4 = UNet(4, 5).eval(), UNet(4, 4).eval()
    assert sum(p.numel() for p in m5.parameters()) == 31_038_469
    assert sum(p.numel() for p in m4.parameters()) == 31_038_404
    with torch.no_grad():
        assert m5(torch.rand(2, 4, 64, 64)).shape == (2, 5, 64, 64)
        assert m4(torch.rand(1, 4, 90, 70)).shape == (1, 4, 90, 70)  # sizes not divisible by 16


def test_load_model_variants():
    torch.manual_seed(0)
    with tempfile.TemporaryDirectory() as tmp:
        sd5 = UNet(4, 5).state_dict()
        torch.save({"module." + k: v for k, v in sd5.items()}, Path(tmp) / "dp5.pth")
        torch.save(UNet(4, 4).state_dict(), Path(tmp) / "plain4.pth")
        torch.save({"state_dict": sd5}, Path(tmp) / "wrapped5.pth")
        m = load_model(Path(tmp) / "dp5.pth")
        assert m.n_classes == 5 and not m.training
        assert all(torch.equal(m.state_dict()[k], v) for k, v in sd5.items())
        assert load_model(Path(tmp) / "plain4.pth").n_classes == 4
        assert load_model(Path(tmp) / "wrapped5.pth", n_classes=5).n_classes == 5
        try:
            load_model(Path(tmp) / "plain4.pth", n_classes=5)
        except ValueError:
            pass
        else:
            raise AssertionError("class-count mismatch not detected")


# rows = reference, columns = prediction (other, tree, shrub, building)
KNOWN_CM = np.array([[8, 2, 0, 0],
                     [1, 6, 1, 0],
                     [0, 0, 3, 1],
                     [0, 0, 0, 4]])


def test_confusion_matrix_orientation():
    ref, pred = [], []
    for r in range(4):
        for c in range(4):
            ref += [r] * KNOWN_CM[r, c]
            pred += [c] * KNOWN_CM[r, c]
    ref, pred = np.array(ref + [IGNORE_INDEX, 2, 4]), np.array(pred + [1, NODATA, 0])  # last 3 are excluded
    assert np.array_equal(confusion_matrix(ref, pred), KNOWN_CM)
    assert np.array_equal(confusion_matrix(torch.from_numpy(ref), torch.from_numpy(pred)), KNOWN_CM)


def test_metrics_known_cm():
    m = metrics_from_cm(KNOWN_CM)
    close = lambda a, b: abs(a - b) < 1e-9  # noqa: E731
    assert close(m["OA"], 21 / 26) and m["pixels"] == 26
    assert all(close(m["PA"][k], v) for k, v in dict(other=0.8, tree=0.75, shrub=0.75, building=1.0).items())
    assert all(close(m["UA"][k], v) for k, v in dict(other=8 / 9, tree=0.75, shrub=0.75, building=0.8).items())
    assert all(close(m["F1"][k], v) for k, v in dict(other=16 / 19, tree=0.75, shrub=0.75, building=8 / 9).items())
    assert all(close(m["IoU"][k], v) for k, v in dict(other=8 / 11, tree=0.6, shrub=0.6, building=0.8).items())
    assert close(m["mIoU"], (8 / 11 + 0.6 + 0.6 + 0.8) / 4)
    cm = KNOWN_CM.copy()
    cm[3, :] = 0  # no building at all -> building metrics undefined
    cm[:, 3] = 0
    assert np.isnan(metrics_from_cm(cm)["IoU"]["building"])
    assert metrics_from_cm(cm, undefined="zero")["IoU"]["building"] == 0.0


def _rule_inputs():
    #                 px0   px1   px2   px3   px4   px5   px6
    probs = np.array([[0.10, 0.10, 0.10, 0.05, 0.52, 0.30, 0.20],   # other
                      [0.60, 0.60, 0.45, 0.30, 0.41, 0.30, 0.40],   # tree
                      [0.20, 0.20, 0.45, 0.00, 0.00, 0.30, 0.00],   # shrub
                      [0.10, 0.10, 0.00, 0.65, 0.00, 0.10, 0.40]],  # building
                     np.float32)[:, None, :]
    hi, lo = (0.1, 0.5), (0.30, 0.35)  # (red, nir): NDVI 0.667 and 0.077
    rn = [hi, lo, hi, lo, hi, hi, hi]
    image = np.zeros((4, 1, 7), np.float32)
    image[0, 0] = [r for r, _ in rn]
    image[3, 0] = [n for _, n in rn]
    return probs, image


def test_paper_rule_and_argmax():
    probs, image = _rule_inputs()
    assert np.allclose(ndvi(image).numpy()[0, :2], [0.4 / 0.6, 0.05 / 0.65])
    assert classify(probs, image, "paper").tolist() == [[1, 0, 2, 3, 0, 0, 0]]
    assert classify(probs, image, "paper", ndvi_min=None).tolist() == [[1, 1, 2, 3, 0, 0, 0]]
    assert classify(probs, image, "argmax").tolist() == [[1, 1, 1, 3, 0, 0, 1]]
    out = classify(torch.from_numpy(probs)[None], torch.from_numpy(image)[None], "paper")  # batched torch input
    assert out.dtype == torch.uint8 and out.tolist() == [[[1, 0, 2, 3, 0, 0, 0]]]
    assert classify(probs, image, "paper", thresholds={"building": 0.7}).tolist() == [[1, 0, 2, 0, 0, 0, 0]]


class PixelModel(torch.nn.Module):
    """1x1-conv stand-in for the U-Net: its output for a pixel does not depend on the window."""

    def __init__(self):
        super().__init__()
        torch.manual_seed(1)
        self.conv = torch.nn.Conv2d(4, 5, 1)
        torch.nn.init.normal_(self.conv.weight, std=3.0)

    def forward(self, x):
        return self.conv(x)


def test_sliding_window_matches_pixelwise():
    rng = np.random.default_rng(0)
    image = rng.integers(1, 256, size=(4, 300, 455)).astype(np.uint8)
    image[:, 250:, 400:] = 0  # nodata corner
    model = PixelModel().eval()
    x = torch.from_numpy(image[None].astype(np.float32) / 255)
    ref_probs = predict_probs(model, x)[0].numpy()
    ref_cls = classify(ref_probs, x[0].numpy(), "paper")
    ref_cls[250:, 400:] = NODATA
    for blend in ("ramp", "mean"):
        cls, probs = predict_array(model, image, with_probs=True, tile=128, overlap=0.25, batch_size=3, blend=blend)
        assert np.allclose(probs[:, :250], ref_probs[:, :250], atol=1e-5)
        assert (cls == ref_cls).mean() > 0.9999 and (cls[250:, 400:] == NODATA).all()
    rows = []
    predict_large(model, lambda a, b: image[:, a:b], 300, 455, lambda y, c, p: rows.append((y, c.shape[0])),
                  tile=100, overlap=0.6)
    assert rows[0][0] == 0 and all(a[0] + a[1] == b[0] for a, b in zip(rows, rows[1:]))
    assert rows[-1][0] + rows[-1][1] == 300


def test_augment():
    torch.manual_seed(0)
    image = torch.rand(4, 64, 64)
    label = torch.randint(0, 4, (64, 64))
    img_r, lab_r = rotate(image, label, 45.0)
    assert lab_r[0, 0] == IGNORE_INDEX and img_r[:, 0, 0].eq(0).all()
    lab90 = rotate(image, label, 90.0)[1]  # a quarter turn moves pixels without loss
    assert any(torch.equal(lab90, torch.rot90(label, k, (0, 1))) for k in (1, -1))
    assert torch.equal(rotate(image, label, 0.0)[1], label)
    assert torch.equal(scale_jitter(image, label, 1.0)[1], label)
    aug = Augment(rotate_p=1.0, scale_p=1.0)
    for _ in range(5):
        i2, l2 = aug(image, label)
        assert i2.shape == image.shape and l2.shape == label.shape
        assert set(l2.unique().tolist()) <= {0, 1, 2, 3, IGNORE_INDEX}


def test_npz_dataset_and_loss():
    rng = np.random.default_rng(0)
    imgs = rng.integers(0, 256, size=(3, 32, 32, 4)).astype(np.uint8)
    labs = rng.integers(0, 6, size=(3, 32, 32)).astype(np.uint8)
    with tempfile.TemporaryDirectory() as tmp:
        np.savez(Path(tmp) / "img.npz", imgs)
        np.savez(Path(tmp) / "lab.npz", labs)
        ds = TileDataset.from_npz(Path(tmp) / "img.npz", Path(tmp) / "lab.npz", unlabeled_value=4, unlabeled_as=0)
        ds = pickle.loads(pickle.dumps(ds))  # as sent to DataLoader workers
        x, y = ds[1]
        assert x.shape == (4, 32, 32) and x.dtype == torch.float32
        assert torch.allclose(x, torch.from_numpy(imgs[1].transpose(2, 0, 1) / 255.0).float())
        raw = labs[1]
        assert (y.numpy()[raw == 4] == 0).all() and (y.numpy()[raw == 5] == IGNORE_INDEX).all()
        assert (y.numpy()[raw < 4] == raw[raw < 4]).all()
        loss = weighted_cross_entropy(n_outputs=5)
        assert torch.allclose(loss.weight, torch.tensor([0.4, 1.0, 1.1, 1.5, 0.0])) and loss.ignore_index == 255
        value = loss(torch.randn(1, 5, 32, 32), y[None])
        assert torch.isfinite(value)
        del ds, x, y  # release the memory map before the folder is removed
        gc.collect()


def test_aggregate_30m_matches_block_counts():
    """30 m fractions equal plain counting, also in edge blocks 1-3 cells wide (a GDAL 'average' pitfall)."""
    import rasterio
    from rasterio.transform import from_origin

    spec = importlib.util.spec_from_file_location("aggregate_30m", ROOT / "scripts" / "aggregate_30m.py")
    agg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(agg)
    k = np.random.default_rng(0).integers(0, 51, (3, 7))  # tree columns per 30 m cell of 50 x 50 px
    cls = np.zeros((150, 350), np.uint8)
    for i, j in np.ndindex(k.shape):
        cls[i * 50:(i + 1) * 50, j * 50:j * 50 + k[i, j]] = 1
    with tempfile.TemporaryDirectory() as tmp:
        src, out = str(Path(tmp) / "class.tif"), str(Path(tmp) / "frac.tif")
        with rasterio.open(src, "w", driver="GTiff", width=350, height=150, count=1, dtype="uint8", nodata=255,
                           crs="EPSG:6414", transform=from_origin(-200010.0, 30000.0, 0.6, 0.6)) as d:
            d.write(cls, 1)
        for block in (4, 5, 6, 64):  # 7 cells wide: edge blocks of 3, 2 and 1 cells; one block
            agg.main(["--input", src, "--res", "30", "--block", str(block), "--output", out])
            with rasterio.open(out) as d:
                assert np.abs(d.read(1) - k / 50).max() < 1e-4, f"--block {block}"


if __name__ == "__main__":
    tests =[(n, f) for n, f in list(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except Exception:
            failed += 1
            traceback.print_exc()
            print(f"FAIL {name}")
    print(f"{len(tests) - failed}/{len(tests)} tests passed")
    sys.exit(1 if failed else 0)
