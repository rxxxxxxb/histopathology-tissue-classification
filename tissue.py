"""Shared utilities for the tissue-classification project.

This module is built up incrementally alongside the per-section walkthrough
notebooks: each commit adds only the utilities needed by the notebook it
supports, so the git history mirrors how the pipeline grew section by section.

Currently covers data loading/preprocessing, model building/training, and the
resize-vs-crop-vs-HED-jitter ablation (used by ``preprocessing_walkthrough.ipynb``,
the model-and-training walkthrough, and the upcoming ablation walkthrough):
    * Constants — class names, ImageNet normalization stats.
    * ``load_image`` — TIFF loader with the 16-bit → 8-bit rescale.
    * ``encode_labels`` — class name -> integer label encoding.
    * ``DiscreteRotation90`` — 90° rotation augmentation without interpolation artefacts.
    * ``build_train_transforms`` / ``build_eval_transforms`` — pipelines for training and
      for single-tile evaluation.
    * ``TissueDataset`` — thin ``torch.utils.data.Dataset`` wrapping ``(path, label)`` pairs.
    * ``set_seed`` / ``pick_device`` — reproducibility and device selection helpers.
    * ``build_model`` — ResNet-50 with ImageNet weights, head swapped for N classes.
    * ``TrainConfig`` — single source of truth for training hyperparameters.
    * ``train_one_fold`` / ``run_cv`` — training loop and StratifiedKFold wrapper.
    * ``save_cv_results`` / ``load_cv_metrics`` — persistence for the CV run.
    * ``HEDColorJitter`` — Tellez-style stain-aware colour jitter augmentation.
    * ``build_train_transforms_resize224`` / ``build_eval_transforms_resize224`` —
      classic ImageNet resize/crop pipeline, the ablation baseline.
    * ``build_train_transforms_hed`` — main pipeline + HED jitter, for the ablation.
    * ``AblationVariant`` / ``run_ablation`` — declare and run each ablation row,
      with per-variant checkpoint caching.

Evaluation/diagnostics utilities (Grad-CAM, held-out prediction collection) will
land here as that section is split out.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm
import torchvision.transforms.functional as TF
import yaml
from PIL import Image
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
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


# ---------------------------------------------------------------------------
# Reproducibility and device
# ---------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and torch (CPU + CUDA). Also force deterministic cuDNN."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def pick_device() -> torch.device:
    """Return the best available device: CUDA → MPS → CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


def build_model(num_classes: int = 4) -> nn.Module:
    """ResNet-50 with ImageNet-V2 weights, ``fc`` swapped for ``num_classes`` outputs.

    Why ResNet-50: well-understood, right capacity for ``n = 400``, self-contained
    ImageNet weights (no auth or extra dependencies). Pathology-SSL foundation models
    (UNI / Virchow / CTransPath / Lunit-DINO) are the obvious next step — see Future
    Work in the README.
    """
    model = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


@dataclass
class TrainConfig:
    """Single source of truth for training hyperparameters.

    Both the notebook (single-fold demo) and ``train.py`` (full CV) read from this.
    """

    epochs: int = 30
    batch_size: int = 16
    lr: float = 1e-4
    weight_decay: float = 1e-4
    patience: int = 7  # early-stopping patience on val macro-F1
    crop_size: int = 512
    num_workers: int = 4
    seed: int = 42
    n_splits: int = 5


def _batch_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    }


def train_one_fold(
    train_ds: "TissueDataset",
    val_ds: "TissueDataset",
    cfg: TrainConfig,
    device: Optional[torch.device] = None,
    progress: Optional[Callable[[str], None]] = None,
) -> dict[str, Any]:
    """Train a fresh ResNet-50 on one fold.

    Returns a dict with ``history`` (per-epoch train/val loss and val metrics),
    ``best_state`` (model state_dict of the best epoch by val macro-F1) and
    ``best_metrics``.
    """
    device = device or pick_device()
    log = progress if progress is not None else print

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = build_model(num_classes=len(CLASSES)).to(device)
    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    criterion = nn.CrossEntropyLoss()

    history: dict[str, list[float]] = {
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_balanced_accuracy": [],
        "val_macro_f1": [],
    }
    best_macro_f1 = -float("inf")
    best_state: Optional[dict[str, torch.Tensor]] = None
    best_metrics: Optional[dict[str, float]] = None
    patience_counter = 0

    for epoch in range(cfg.epochs):
        model.train()
        running_loss = 0.0
        n_samples = 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x.size(0)
            n_samples += x.size(0)
        train_loss = running_loss / max(n_samples, 1)

        model.eval()
        running_loss = 0.0
        n_samples = 0
        pred_chunks: list[np.ndarray] = []
        true_chunks: list[np.ndarray] = []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device, non_blocking=True)
                y_dev = y.to(device, non_blocking=True)
                logits = model(x)
                loss = criterion(logits, y_dev)
                running_loss += loss.item() * x.size(0)
                n_samples += x.size(0)
                pred_chunks.append(logits.argmax(1).cpu().numpy())
                true_chunks.append(y.numpy())
        val_loss = running_loss / max(n_samples, 1)
        y_pred = np.concatenate(pred_chunks)
        y_true = np.concatenate(true_chunks)
        metrics = _batch_metrics(y_true, y_pred)

        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(metrics["accuracy"])
        history["val_balanced_accuracy"].append(metrics["balanced_accuracy"])
        history["val_macro_f1"].append(metrics["macro_f1"])

        log(
            f"  ep {epoch + 1:3d}/{cfg.epochs}  "
            f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
            f"val_acc={metrics['accuracy']:.3f}  val_macroF1={metrics['macro_f1']:.3f}"
        )

        if metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = metrics["macro_f1"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_metrics = metrics
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= cfg.patience:
                log(f"  early stopping at epoch {epoch + 1} (no val improvement for {cfg.patience} epochs)")
                break

    return {
        "history": history,
        "best_state": best_state,
        "best_metrics": best_metrics,
    }


def run_cv(
    paths: Sequence[PathLike],
    labels: Sequence[int],
    cfg: TrainConfig,
    transform_builder: Callable[[int], Callable] = build_train_transforms,
    eval_transform_builder: Optional[Callable[[int], Callable]] = None,
    device: Optional[torch.device] = None,
    progress: Optional[Callable[[str], None]] = None,
) -> dict[str, Any]:
    """Run Stratified K-fold CV. Returns per-fold results, summary stats, and split indices.

    ``transform_builder`` builds the training transform; ``eval_transform_builder`` builds
    the per-epoch validation transform (a single-tile, deterministic pipeline). If
    ``eval_transform_builder`` is not given, the default :func:`build_eval_transforms` is
    used — which matches ``transform_builder=build_train_transforms``.
    """
    device = device or pick_device()
    log = progress if progress is not None else print
    eval_builder = eval_transform_builder or build_eval_transforms
    y = np.asarray(labels)

    skf = StratifiedKFold(n_splits=cfg.n_splits, shuffle=True, random_state=cfg.seed)
    fold_results: list[dict[str, Any]] = []
    splits: list[dict[str, list[int]]] = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(np.arange(len(paths)), y)):
        log(f"\n=== Fold {fold_idx + 1}/{cfg.n_splits}  (train={len(train_idx)}, val={len(val_idx)}) ===")
        set_seed(cfg.seed + fold_idx)

        train_ds = TissueDataset(
            [paths[i] for i in train_idx],
            [labels[i] for i in train_idx],
            transform=transform_builder(cfg.crop_size),
        )
        val_ds = TissueDataset(
            [paths[i] for i in val_idx],
            [labels[i] for i in val_idx],
            transform=eval_builder(cfg.crop_size),
        )

        result = train_one_fold(train_ds, val_ds, cfg, device=device, progress=log)
        result["fold"] = fold_idx
        fold_results.append(result)
        splits.append({"train_idx": train_idx.tolist(), "val_idx": val_idx.tolist()})

        log(
            f"  fold {fold_idx + 1} best: "
            f"acc={result['best_metrics']['accuracy']:.3f} "
            f"macroF1={result['best_metrics']['macro_f1']:.3f}"
        )

    summary_keys = ("accuracy", "balanced_accuracy", "macro_f1")
    summary = {
        k: {
            "mean": float(np.mean([r["best_metrics"][k] for r in fold_results])),
            "std": float(np.std([r["best_metrics"][k] for r in fold_results])),
            "per_fold": [r["best_metrics"][k] for r in fold_results],
        }
        for k in summary_keys
    }
    best_fold = int(np.argmax([r["best_metrics"]["macro_f1"] for r in fold_results]))

    return {
        "fold_results": fold_results,
        "summary": summary,
        "best_fold": best_fold,
        "splits": splits,
        "cfg": asdict(cfg),
    }


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save_cv_results(results: dict[str, Any], out_dir: PathLike) -> None:
    """Persist the outputs of ``run_cv``.

    Writes:
        - ``final.pt``    — state_dict of the best fold's model.
        - ``splits.json`` — per-fold train/val indices + best fold index.
        - ``config.yaml`` — the ``TrainConfig`` used.
        - ``metrics.json``— per-fold metrics, histories, and summary stats.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    best_idx = results["best_fold"]
    best_state = results["fold_results"][best_idx]["best_state"]
    if best_state is None:
        raise RuntimeError("best fold has no saved state_dict")
    torch.save(best_state, out / "final.pt")

    with open(out / "splits.json", "w") as f:
        json.dump({"best_fold": best_idx, "splits": results["splits"]}, f, indent=2)

    with open(out / "config.yaml", "w") as f:
        yaml.safe_dump(results["cfg"], f)

    metrics_only = {
        "summary": results["summary"],
        "best_fold": best_idx,
        "fold_metrics": [r["best_metrics"] for r in results["fold_results"]],
        "fold_histories": [r["history"] for r in results["fold_results"]],
    }
    with open(out / "metrics.json", "w") as f:
        json.dump(metrics_only, f, indent=2)


def load_cv_metrics(out_dir: PathLike) -> Optional[dict[str, Any]]:
    """Load the serialised CV metrics if they exist, else ``None``."""
    path = Path(out_dir) / "metrics.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# §4 Ablation — extra transforms and a thin runner
# ---------------------------------------------------------------------------


class HEDColorJitter:
    """Tellez-style HED stain-aware colour jitter.

    Converts ``RGB`` → ``HED`` (haematoxylin / eosin / DAB) via a fixed stain matrix,
    applies a per-channel multiplicative ``alpha`` and additive ``beta`` perturbation,
    then converts back.

    **Why HED over RGB jitter.** H&E stain intensity varies meaningfully across
    scanners, laboratories, and batches — but that variation lives on axes aligned
    with *stain concentrations*, not with the RGB axes of a display. Perturbing in
    HED space simulates the real source of colour variability. Perturbing in RGB
    space (``ColorJitter``) can produce colour combinations that do not exist in real
    H&E images (e.g. green tissue), which is net-harmful as augmentation.

    Reference: Tellez et al., 2019, *Quantifying the effects of data augmentation and
    stain colour normalization in convolutional neural networks for computational
    pathology*.
    """

    def __init__(self, alpha: float = 0.05, beta: float = 0.05) -> None:
        # Lazy import so that skimage only becomes a hard dep when this is used.
        from skimage.color import rgb2hed, hed2rgb

        self.alpha = float(alpha)
        self.beta = float(beta)
        self._rgb2hed = rgb2hed
        self._hed2rgb = hed2rgb

    def __call__(self, img: Image.Image) -> Image.Image:
        arr = np.asarray(img, dtype=np.float32) / 255.0
        hed = self._rgb2hed(arr)
        for c in range(3):
            a = np.random.uniform(1.0 - self.alpha, 1.0 + self.alpha)
            b = np.random.uniform(-self.beta, self.beta)
            hed[..., c] = hed[..., c] * a + b
        rgb = np.clip(self._hed2rgb(hed), 0.0, 1.0)
        return Image.fromarray((rgb * 255.0).astype(np.uint8))

    def __repr__(self) -> str:
        return f"{type(self).__name__}(alpha={self.alpha}, beta={self.beta})"


def build_train_transforms_resize224(_crop_size_unused: int = 512) -> Callable:
    """Classic ImageNet-style pipeline: resize to 256 then random-crop 224.

    Used as the ablation baseline to quantify the crop-vs-resize claim. The train
    pipeline still has a random-crop augmentation (so we're not unfairly handicapping
    it by stripping augmentation) — the only difference from the main pipeline is the
    **effective resolution**: 224 px after a 2048→256 down-scale vs. 512 px native.

    The ``_crop_size_unused`` parameter is present only so the signature matches the
    shape of other builders passed into ``run_cv``.
    """
    return T.Compose(
        [
            T.Resize(256),
            T.RandomCrop(224),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            DiscreteRotation90(),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def build_eval_transforms_resize224(_crop_size_unused: int = 512) -> Callable:
    """Eval-side counterpart: resize to 256, centre-crop 224 (standard ImageNet eval)."""
    return T.Compose(
        [
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def build_train_transforms_hed(crop_size: int = 512) -> Callable:
    """Main pipeline + HED stain jitter.

    HED jitter is applied *after* ``RandomCrop`` — there's no reason to pay the
    skimage colour-conversion cost on pixels we're about to throw away — and *before*
    ``ToTensor``, because :class:`HEDColorJitter` expects a PIL image.
    """
    return T.Compose(
        [
            T.RandomCrop(crop_size),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            DiscreteRotation90(),
            HEDColorJitter(alpha=0.05, beta=0.05),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


@dataclass
class AblationVariant:
    """One row in the ablation table.

    Attributes:
        name: Short slug used as the subdirectory name for saved checkpoints.
        label: Display name for the results table.
        train_builder: Function ``(crop_size) -> Compose`` producing the train transform.
        eval_builder: Function ``(crop_size) -> Compose`` producing the eval transform.
        crop_size: Passed to the builders (ignored by the resize-224 builders).
    """

    name: str
    label: str
    train_builder: Callable[[int], Callable]
    eval_builder: Callable[[int], Callable]
    crop_size: int = 512


def run_ablation(
    paths: Sequence[PathLike],
    labels: Sequence[int],
    cfg: TrainConfig,
    variants: Sequence[AblationVariant],
    out_root: PathLike = Path("checkpoints"),
    device: Optional[torch.device] = None,
    progress: Optional[Callable[[str], None]] = None,
) -> dict[str, dict[str, Any]]:
    """Run Stratified 5-fold CV for each variant and return ``{name: metrics_dict}``.

    Each variant is persisted to ``<out_root>/<name>/`` via :func:`save_cv_results`, so
    that re-running the notebook doesn't retrain any variant whose metrics already
    exist on disk. The special name ``"main"`` reuses the top-level ``<out_root>/``
    directory where §3 wrote its results.
    """
    log = progress if progress is not None else print
    device = device or pick_device()
    out_root = Path(out_root)
    results: dict[str, dict[str, Any]] = {}

    from dataclasses import replace as _replace

    for v in variants:
        out_dir = out_root if v.name == "main" else out_root / v.name
        cached = load_cv_metrics(out_dir)
        if cached is not None:
            log(f"[{v.label}] cached at {out_dir}/metrics.json — skipping")
            results[v.name] = cached
            continue

        log(f"\n[{v.label}] running {cfg.n_splits}-fold CV (saving to {out_dir}/)")
        v_cfg = _replace(cfg, crop_size=v.crop_size)
        cv = run_cv(
            paths,
            labels,
            v_cfg,
            transform_builder=v.train_builder,
            eval_transform_builder=v.eval_builder,
            device=device,
            progress=log,
        )
        save_cv_results(cv, out_dir)
        results[v.name] = load_cv_metrics(out_dir)

    return results

