# Copyright (C) 2023-2026 Yunzhe Zhu
# SPDX-License-Identifier: GPL-3.0-only
"""Tile datasets for training and evaluation.

Two storage layouts are supported:

* **npz arrays** (the layout used to train the released model): an image file
  holding a ``(N, H, W, 4)`` uint8 array (R, G, B, NIR) and a label file holding
  a ``(N, H, W)`` uint8 array. Each file has one member (``arr_0`` by default).
  Uncompressed ``.npz`` files (``np.savez``) are memory-mapped, so tens of GB of
  tiles do not have to fit in RAM.
* **GeoTIFF folders**: one 4-band image per tile and one single-band label per
  tile, paired by file name (see :func:`find_pairs`).

Samples are returned as ``(image, label)``: ``image`` is a float32 tensor
``(C, H, W)`` scaled by 1/255 and ``label`` an int64 tensor ``(H, W)`` with
classes 0-3 and 255 = ignore. Any label value outside 0-3 becomes 255.
Legacy arrays use a dedicated "no label" value (4 in the 2024 arrays, 5 in the
2023 arrays); it is ignored by default, or can be mapped to a class with
``unlabeled_value``/``unlabeled_as``.
"""
from __future__ import annotations

import re
import struct
import warnings
import zipfile
from bisect import bisect_right
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from . import IGNORE_INDEX, NUM_CLASSES

PathLike = Union[str, Path]

__all__ = ["NpzArray", "TileDataset", "build_dataset", "find_pairs", "prepare_image", "prepare_label", "read_raster"]


class NpzArray:
    """One array member of an ``.npz`` file, memory-mapped when the file is uncompressed.

    Only the path and array metadata are pickled, so the object can be sent to
    DataLoader worker processes without copying the data.

    Args:
        path: ``.npz`` file.
        key: member name (without ``.npy``). Default: ``arr_0`` if present,
            otherwise the only member of the file.
    """

    def __init__(self, path: PathLike, key: Optional[str] = None):
        self.path = str(path)
        with zipfile.ZipFile(self.path) as zf:
            names = zf.namelist()
            if key is None:
                if "arr_0.npy" in names:
                    member = "arr_0.npy"
                elif len(names) == 1:
                    member = names[0]
                else:
                    raise ValueError(f"{self.path} has several arrays {names}; pass key=")
            else:
                member = key if key.endswith(".npy") else key + ".npy"
            info = zf.getinfo(member)
        self.key = member[:-4]
        self._data = None
        if info.compress_type == zipfile.ZIP_STORED:
            with open(self.path, "rb") as f:
                f.seek(info.header_offset)
                local = f.read(30)
                n_name, n_extra = struct.unpack("<HH", local[26:30])
                f.seek(info.header_offset + 30 + n_name + n_extra)
                version = np.lib.format.read_magic(f)
                if version == (1, 0):
                    shape, fortran, dtype = np.lib.format.read_array_header_1_0(f)
                else:
                    shape, fortran, dtype = np.lib.format.read_array_header_2_0(f)
                self._offset = f.tell()
            self.shape, self.dtype, self._fortran = tuple(shape), np.dtype(dtype), fortran
        else:  # compressed: must be loaded into memory
            warnings.warn(f"{self.path} is compressed; loading '{self.key}' into memory")
            with np.load(self.path) as z:
                self._data = z[self.key]
            self.shape, self.dtype = self._data.shape, self._data.dtype

    def _array(self) -> np.ndarray:
        if self._data is None:
            self._data = np.memmap(self.path, dtype=self.dtype, mode="r", shape=self.shape,
                                   offset=self._offset, order="F" if self._fortran else "C")
        return self._data

    def __getstate__(self):
        state = self.__dict__.copy()
        if isinstance(state.get("_data"), np.memmap):
            state["_data"] = None  # re-opened lazily in the worker
        return state

    def __len__(self) -> int:
        return self.shape[0]

    def __getitem__(self, idx):
        return self._array()[idx]


def prepare_image(image: np.ndarray, channels_last: bool, n_bands: int = 4) -> torch.Tensor:
    """Convert a tile (0-255 digital numbers) to a float32 ``(C, H, W)`` tensor scaled by 1/255."""
    arr = np.asarray(image)
    if channels_last:
        arr = arr[..., :n_bands].transpose(2, 0, 1)
    else:
        arr = arr[:n_bands]
    if arr.shape[0] != n_bands:
        raise ValueError(f"expected {n_bands} bands, got array of shape {image.shape}")
    return torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32) / np.float32(255.0))


def prepare_label(label: np.ndarray, unlabeled_value: Optional[int] = None,
                  unlabeled_as: int = IGNORE_INDEX, n_classes: int = NUM_CLASSES) -> np.ndarray:
    """Map a raw label tile to int64 classes 0..n_classes-1 with 255 = ignore.

    Values outside ``[0, n_classes)`` (including NaN in float rasters) become
    255. Pixels equal to ``unlabeled_value`` become ``unlabeled_as`` (255 by
    default; 0 treats them as "other").
    """
    raw = np.asarray(label)
    if raw.dtype.kind == "f":
        raw = np.where(np.isfinite(raw), raw, -1)
    raw = raw.astype(np.int64)
    out = np.where((raw >= 0) & (raw < n_classes), raw, IGNORE_INDEX)
    if unlabeled_value is not None and unlabeled_as != IGNORE_INDEX:
        out[raw == unlabeled_value] = unlabeled_as
    return out


def read_raster(path: PathLike, bands: Optional[Sequence[int]] = None):
    """Read a GeoTIFF as ``(array, profile)``; ``bands`` are 1-based band indexes."""
    import rasterio

    with rasterio.open(path) as src:
        arr = src.read(list(bands) if bands else None)
        return arr, src.profile.copy()


def find_pairs(image_dir: PathLike, label_dir: PathLike, pattern: str = "*.tif",
               label_template: str = "{stem}.tif", stem_regex: Optional[str] = None) -> List[Tuple[str, Path, Path]]:
    """Pair image and label GeoTIFFs by name.

    For each image matching ``pattern`` in ``image_dir``, the label is
    ``label_dir / label_template.format(stem=key)``. ``key`` is the image file
    stem, or the first group of ``stem_regex`` matched against the stem.
    Example: images ``ID_010_NAIP_2016.tif`` and labels ``ID_010_Label.tif`` ->
    ``stem_regex=r"^(ID_\\d+)"``, ``label_template="{stem}_Label.tif"``.

    Returns a sorted list of ``(key, image_path, label_path)``; images without a
    label are skipped with a warning.
    """
    pairs, missing = [], 0
    for img in sorted(Path(image_dir).glob(pattern)):
        key = img.stem
        if stem_regex:
            m = re.search(stem_regex, key)
            if m is None:
                missing += 1
                continue
            key = m.group(1) if m.groups() else m.group(0)
        lab = Path(label_dir) / label_template.format(stem=key)
        if lab.exists():
            pairs.append((img.stem, img, lab))
        else:
            missing += 1
    if missing:
        warnings.warn(f"{missing} image(s) in {image_dir} have no matching label in {label_dir}")
    return pairs


class TileDataset(Dataset):
    """Image/label tiles from npz arrays or GeoTIFF files.

    Use :meth:`from_npz`, :meth:`from_folder` or :func:`build_dataset` rather
    than the constructor.

    Args:
        images: list of array-likes ``(N, H, W, C)`` (npz layout) or list of
            GeoTIFF paths.
        labels: matching label sources, or ``None`` for image-only datasets.
        transform: callable ``(image, label) -> (image, label)`` applied to tensors
            (e.g. :class:`fuelmap.augment.Augment`).
        unlabeled_value, unlabeled_as: see :func:`prepare_label`.
        ids: optional tile identifiers (default: running index or file stem).
    """

    def __init__(self, images: Sequence, labels: Optional[Sequence] = None, transform: Optional[Callable] = None,
                 unlabeled_value: Optional[int] = None, unlabeled_as: int = IGNORE_INDEX,
                 ids: Optional[Sequence[str]] = None, n_bands: int = 4):
        self.transform = transform
        self.unlabeled_value, self.unlabeled_as, self.n_bands = unlabeled_value, unlabeled_as, n_bands
        self._files = isinstance(images[0], (str, Path)) if len(images) else False
        if self._files:
            self.images = [str(p) for p in images]
            self.labels = None if labels is None else [str(p) for p in labels]
            self._bounds = None
        else:
            self.images, self.labels = list(images), None if labels is None else list(labels)
            sizes = [len(a) for a in self.images]
            if self.labels is not None and sizes != [len(a) for a in self.labels]:
                raise ValueError("image and label arrays have different numbers of tiles")
            self._bounds = np.cumsum([0] + sizes).tolist()
        if self.labels is not None and len(self.labels) != len(self.images):
            raise ValueError("number of image and label sources differs")
        n = len(self.images) if self._files else self._bounds[-1]
        self.ids = list(ids) if ids is not None else (
            [Path(p).stem for p in self.images] if self._files else [str(i) for i in range(n)])

    @classmethod
    def from_npz(cls, images: Union[PathLike, Sequence[PathLike]], labels=None, image_key: Optional[str] = None,
                 label_key: Optional[str] = None, **kwargs) -> "TileDataset":
        """Dataset from one or more image ``.npz`` files (and matching label files), concatenated."""
        as_list = lambda x: [x] if isinstance(x, (str, Path)) else list(x)  # noqa: E731
        imgs = [NpzArray(p, image_key) for p in as_list(images)]
        labs = None if labels is None else [NpzArray(p, label_key) for p in as_list(labels)]
        return cls(imgs, labs, **kwargs)

    @classmethod
    def from_folder(cls, image_dir: PathLike, label_dir: Optional[PathLike] = None, pattern: str = "*.tif",
                    label_template: str = "{stem}.tif", stem_regex: Optional[str] = None, **kwargs) -> "TileDataset":
        """Dataset from a folder of GeoTIFF tiles (and a folder of label GeoTIFFs)."""
        if label_dir is None:
            imgs = sorted(Path(image_dir).glob(pattern))
            return cls(imgs, None, ids=[p.stem for p in imgs], **kwargs)
        pairs = find_pairs(image_dir, label_dir, pattern, label_template, stem_regex)
        if not pairs:
            raise FileNotFoundError(f"no image/label pairs found in {image_dir} and {label_dir}")
        return cls([p[1] for p in pairs], [p[2] for p in pairs], ids=[p[0] for p in pairs], **kwargs)

    def __len__(self) -> int:
        return len(self.images) if self._files else self._bounds[-1]

    def _raw(self, idx: int):
        if self._files:
            img = read_raster(self.images[idx])[0]
            lab = None if self.labels is None else read_raster(self.labels[idx], [1])[0][0]
            return img, lab, False
        part = bisect_right(self._bounds, idx) - 1
        j = idx - self._bounds[part]
        lab = None if self.labels is None else np.asarray(self.labels[part][j])
        return np.asarray(self.images[part][j]), lab, True

    def __getitem__(self, idx: int):
        if idx < 0:
            idx += len(self)
        img, lab, channels_last = self._raw(idx)
        image = prepare_image(img, channels_last, self.n_bands)
        if lab is None:
            return image
        label = torch.from_numpy(prepare_label(lab, self.unlabeled_value, self.unlabeled_as))
        if self.transform is not None:
            image, label = self.transform(image, label)
        return image, label


def build_dataset(images: Union[PathLike, Sequence[PathLike]], labels=None, **kwargs) -> TileDataset:
    """Build a :class:`TileDataset` from ``.npz`` file(s) or a GeoTIFF directory.

    ``images``/``labels`` are either ``.npz`` paths (one or a list) or one
    directory each. Extra keyword arguments go to :meth:`TileDataset.from_npz`
    or :meth:`TileDataset.from_folder` (``pattern``, ``label_template``,
    ``stem_regex``) and to the constructor (``transform``, ``unlabeled_value``...).
    """
    first = images if isinstance(images, (str, Path)) else images[0]
    if Path(first).is_dir():
        lab = labels if labels is None or isinstance(labels, (str, Path)) else labels[0]
        npz_keys = ("image_key", "label_key")
        return TileDataset.from_folder(first, lab, **{k: v for k, v in kwargs.items() if k not in npz_keys})
    folder_keys = ("pattern", "label_template", "stem_regex")
    return TileDataset.from_npz(images, labels, **{k: v for k, v in kwargs.items() if k not in folder_keys})
