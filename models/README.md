# Model weights

The weights are too large for git (124 MB). They are attached to the GitHub release
[`v1.0.0`](https://github.com/YunzheZhu/fuel-mapping-CA/releases/tag/v1.0.0).
This folder holds only the metadata file, `fuelmap_unet_v1.json`.

| File | Size | SHA-256 |
|---|---|---|
| [`fuelmap_unet_v1.pth`](https://github.com/YunzheZhu/fuel-mapping-CA/releases/download/v1.0.0/fuelmap_unet_v1.pth) | 124,238,486 bytes | `35e9f387de0017e7cbddbdfc7fc8cbeb5645eb114a7fc03cdd8ab8acdde5706a` |

**What it is.** This is the release model. It was retrained in June 2024 on the paper's training tiles (the June 2024
version of the split, test tiles held out) and matches the paper's reported overall accuracy (87.1 % on the public
test array, as in the paper). It is not the exact checkpoint behind the paper's tables. Details:

- training run `UNet_RGBN_4y_II` (June 2024)
- 10,969 training tile-years
- 50 epochs; epoch 47 kept
- metadata: [`fuelmap_unet_v1.json`](fuelmap_unet_v1.json); full model card: [`docs/MODEL_CARD.md`](../docs/MODEL_CARD.md)

**Licence.** The weights are released under CC BY 4.0. If you use them, please cite the paper
(doi:[10.1016/j.rse.2025.115091](https://doi.org/10.1016/j.rse.2025.115091)).

## Download and verify

```bash
mkdir -p models
curl -L -o models/fuelmap_unet_v1.pth \
  https://github.com/YunzheZhu/fuel-mapping-CA/releases/download/v1.0.0/fuelmap_unet_v1.pth
sha256sum models/fuelmap_unet_v1.pth   # macOS: shasum -a 256
# expected: 35e9f387de0017e7cbddbdfc7fc8cbeb5645eb114a7fc03cdd8ab8acdde5706a
```

## Load

```python
from fuelmap.model import load_model

model = load_model("models/fuelmap_unet_v1.pth", device="cpu")   # n_classes inferred (5) from the weights
model.eval()
```

Loading without the package (plain PyTorch):

```python
import torch

state = torch.load("models/fuelmap_unet_v1.pth", map_location="cpu", weights_only=True)
# UNet = the U-Net class shipped in the fuelmap package (milesial/Pytorch-UNet layout)
model = UNet(n_channels=4, n_classes=5, bilinear=False)
model.load_state_dict(state)
model.eval()
```

## Input and output

| Item | Value |
|---|---|
| Input | NAIP 4-band `uint8`, band order **R, G, B, NIR**, divided by 255 (float32, NCHW); 512 × 512 px tiles at 0.6 m during training |
| Output | logits with 5 channels: 0 = other, 1 = tree, 2 = shrub, 3 = building, 4 = legacy "unlabeled" channel |
| Channel 4 | Legacy channel that was never a training target (loss weight 0). The archived data loader encoded no-label pixels as class 0, so they were trained as other. Channel 4 is never a valid prediction: softmax over all 5, then keep channels 0–3 |
| Weights format | plain `state_dict` (no `module.` prefix), float32, 118 tensors, 31,038,469 parameters |

Decision rule used in the paper (after softmax):

1. tree if p_tree > 0.4 and NDVI > 0.11
2. shrub if p_shrub > 0.4 and NDVI > 0.11
3. building if p_building > 0.6
4. other if p_other > 0.5 (applied last, so it overrides 1–3)
5. every remaining pixel is other

NDVI = (NIR − R) / (NIR + R), computed on DN / 255.

Plain argmax over channels 0–3 is also supported. On the test array it scores about 1 point lower in overall accuracy (OA)
(see the JSON).

## Accuracy of the release model

These numbers are a re-evaluation on the 693-tile public one-year test array (`fuelmap_test_1y.npz`). In the confusion
matrix, rows are the reference and columns the prediction. UA is precision and PA is recall. "no label" pixels are
ignored.

| Rule | OA | mIoU | mean F1 | F1 other | F1 tree | F1 shrub | F1 building |
|---|---|---|---|---|---|---|---|
| paper rule | 0.8708 | 0.7249 | 0.8388 | 0.9135 | 0.8305 | 0.8040 | 0.8071 |
| argmax | 0.8592 | 0.7045 | 0.8245 | 0.9028 | 0.8287 | 0.7879 | 0.7786 |
