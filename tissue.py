"""Shared utilities for the tissue-classification project.

This module is built up incrementally alongside the per-section walkthrough
notebooks: each commit adds only the utilities needed by the notebook it
supports, so the git history mirrors how the pipeline grew section by section.

Currently covers data loading and preprocessing (used by
``preprocessing_walkthrough.ipynb``):
    * Constants — class names, ImageNet normalization stats.
    * ``load_image`` — TIFF loader with the 16-bit → 8-bit rescale.
    * ``encode_labels`` — class name -> integer label encoding.
    * ``DiscreteRotation90`` — 90° rotation augmentation without interpolation artefacts.
    * ``build_train_transforms`` / ``build_eval_transforms`` — pipelines for training and
      for single-tile evaluation.
    * ``TissueDataset`` — thin ``torch.utils.data.Dataset`` wrapping ``(path, label)`` pairs.

Model-building, the CV training loop, ablation variants, and evaluation/diagnostics
utilities will land here as the corresponding walkthrough sections are split out.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Callable, Optional, Sequence, Union

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

PathLike = Union[str, Path]

CLASSES: tuple[str, ...] = ("class1", "class2", "class3", "class4")
CLASS_TO_IDX: dict[str, int] = {c: i for i, c in enumerate(CLASSES)}

IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)


def load_image(path: PathLike) -> np.ndarray:
    """Load a TIFF tile and return it as an 8-bit RGB array.

    The dataset mixes 8-bit and 16-bit TIFFs. For 16-bit files we right-shift by one
    byte (equivalent to ``// 256``) and cast to ``uint8``. We considered a float
    ``[0, 1]`` loader that would preserve any sub-8-bit precision in 16-bit files, but
    the downstream backbone is ImageNet-pretrained and its weights do not encode
    sub-8-bit information, so the simpler rescale is the more honest choice. See
    ``exploration.ipynb`` §1 Step 6.
    """
    try:
        import tifffile
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "tifffile is required to load TIFF images. "
            "Install it in the active environment to use load_image()."
        ) from exc
    img = tifffile.imread(str(path))
    if img.dtype == np.uint16:
        img = (img >> 8).astype(np.uint8)
    return img


def encode_labels(class_names: Sequence[str]) -> list[int]:
    """Turn ``["class1", "class3", ...]`` into ``[0, 2, ...]``."""
    return [CLASS_TO_IDX[c] for c in class_names]


class DiscreteRotation90:
    """Randomly rotate the image by one of ``{0, 90, 180, 270}`` degrees.

    We deliberately do *not* use ``transforms.RandomRotation(90)``: that samples a
    continuous angle in ``[-90, 90]`` and introduces bilinear interpolation plus black
    border padding at non-right-angle rotations. The border artefact in particular is
    a strong, class-agnostic shortcut that the network can overfit to.

    Tissue sections have no inherent orientation, so the *right* augmentation is to
    sample from the discrete rotational symmetry group ``{0°, 90°, 180°, 270°}``, which
    is an exact (no-interpolation, no-padding) operation on any image.
    """

    def __call__(self, img):
        k = random.randint(0, 3)
        if k == 0:
            return img
        return TF.rotate(img, angle=90 * k)

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


def build_train_transforms(crop_size: int = 512) -> Callable:
    """Training pipeline.

    - ``RandomCrop(crop_size)`` — gives a different patch per epoch, functioning as a
      free augmentation on top of the explicit ones.
    - ``RandomHorizontalFlip`` / ``RandomVerticalFlip`` — tissue has no preferred axis.
    - ``DiscreteRotation90`` — see its docstring; respects symmetry without artefacts.
    - ``ToTensor`` + ImageNet ``Normalize`` — matches the backbone's input distribution.

    HED / stain-aware colour jitter is deliberately *not* included here: it's evaluated
    as a separate ablation row (§4) so we can measure its effect in isolation.
    """
    return T.Compose(
        [
            T.RandomCrop(crop_size),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            DiscreteRotation90(),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def build_eval_transforms(crop_size: int = 512) -> Callable:
    """Deterministic single-tile pipeline used during training-time validation.

    The full image-level evaluation (non-overlapping grid + flip TTA + mean softmax)
    is a separate helper we'll add in §5 — this transform is only for cheap per-epoch
    validation on a single centre crop.
    """
    return T.Compose(
        [
            T.CenterCrop(crop_size),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


class TissueDataset(Dataset):
    """Dataset wrapping a list of ``(path, label)`` pairs.

    Loading goes through :func:`load_image` so the 16-bit → 8-bit rescale is applied
    consistently. The array is then wrapped as a ``PIL.Image`` so we can use the
    standard torchvision transforms.
    """

    def __init__(
        self,
        paths: Sequence[PathLike],
        labels: Sequence[int],
        transform: Optional[Callable] = None,
    ) -> None:
        if len(paths) != len(labels):
            raise ValueError(
                f"paths and labels must have the same length "
                f"(got {len(paths)} and {len(labels)})"
            )
        self.paths: list[PathLike] = list(paths)
        self.labels: list[int] = [int(y) for y in labels]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img = load_image(self.paths[idx])
        x = Image.fromarray(img)
        if self.transform is not None:
            x = self.transform(x)
        return x, self.labels[idx]
