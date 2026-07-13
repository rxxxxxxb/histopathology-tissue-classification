"""Headless runner for §4 ablation. Writes to checkpoints/<variant>/.

Re-entrant: any variant with an existing ``checkpoints/<name>/metrics.json`` is skipped.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

from tissue import (
    AblationVariant,
    TrainConfig,
    build_eval_transforms,
    build_eval_transforms_resize224,
    build_train_transforms,
    build_train_transforms_hed,
    build_train_transforms_resize224,
    encode_labels,
    pick_device,
    run_ablation,
)


def _scan_dataset(data_dir: Path) -> pd.DataFrame:
    rows = []
    for cls_dir in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        for img in sorted(cls_dir.iterdir()):
            if img.suffix.lower() in {".tif", ".tiff"}:
                rows.append({"path": str(img), "class": cls_dir.name})
    return pd.DataFrame(rows)


def main() -> None:
    df = _scan_dataset(ROOT / "data")
    paths = df["path"].tolist()
    labels = encode_labels(df["class"].tolist())
    print(f"dataset: {len(df)} images across {df['class'].nunique()} classes", flush=True)

    cfg = TrainConfig()
    variants = [
        AblationVariant(
            name="resize224",
            label="ResNet-50 + resize-224",
            train_builder=build_train_transforms_resize224,
            eval_builder=build_eval_transforms_resize224,
            crop_size=224,
        ),
        AblationVariant(
            name="main",
            label="ResNet-50 + crop-512 (main)",
            train_builder=build_train_transforms,
            eval_builder=build_eval_transforms,
            crop_size=512,
        ),
        AblationVariant(
            name="hed",
            label="ResNet-50 + crop-512 + HED jitter",
            train_builder=build_train_transforms_hed,
            eval_builder=build_eval_transforms,
            crop_size=512,
        ),
    ]

    device = pick_device()
    print(f"device: {device}", flush=True)

    t0 = time.time()
    results = run_ablation(
        paths,
        labels,
        cfg,
        variants,
        out_root=ROOT / "checkpoints",
        device=device,
        progress=lambda m: print(m, flush=True),
    )

    print(f"\n=== Ablation complete in {(time.time() - t0) / 60:.1f} min ===", flush=True)
    for v in variants:
        s = results[v.name]["summary"]
        print(
            f"{v.label:40s}  "
            f"macroF1={s['macro_f1']['mean']:.3f} ± {s['macro_f1']['std']:.3f}  "
            f"acc={s['accuracy']['mean']:.3f} ± {s['accuracy']['std']:.3f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
