"""Run resize-224 CV with per-image held-out prediction collection.

Writes ``checkpoints/resize224/predictions.npz`` so §5 can load it without retraining.
Existing ``final.pt`` / ``metrics.json`` / ``splits.json`` are left untouched.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

from tissue import (
    TrainConfig,
    build_eval_transforms_resize224,
    build_train_transforms_resize224,
    collect_cv_predictions,
    encode_labels,
    pick_device,
    save_cv_predictions,
)


def _scan_dataset(data_dir: Path) -> pd.DataFrame:
    rows = []
    for cls_dir in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        for img in sorted(cls_dir.iterdir()):
            if img.suffix.lower() in {".tif", ".tiff"}:
                rows.append({"path": str(img), "class": cls_dir.name})
    return pd.DataFrame(rows)


def main() -> None:
    out_dir = ROOT / "checkpoints" / "resize224"
    if (out_dir / "predictions.npz").exists():
        print(f"predictions already exist at {out_dir / 'predictions.npz'} — nothing to do", flush=True)
        return

    df = _scan_dataset(ROOT / "data")
    paths = df["path"].tolist()
    labels = encode_labels(df["class"].tolist())
    print(f"dataset: {len(df)} images across {df['class'].nunique()} classes", flush=True)

    cfg = TrainConfig()  # same seed/protocol as §3 & §4 — deterministic re-run
    device = pick_device()
    print(f"device: {device}", flush=True)

    t0 = time.time()
    results = collect_cv_predictions(
        paths,
        labels,
        cfg,
        transform_builder=build_train_transforms_resize224,
        eval_transform_builder=build_eval_transforms_resize224,
        device=device,
        progress=lambda m: print(m, flush=True),
    )
    print(f"\n=== CV complete in {(time.time() - t0) / 60:.1f} min ===", flush=True)

    s = results["summary"]
    print(f"  macroF1 = {s['macro_f1']['mean']:.3f} ± {s['macro_f1']['std']:.3f}", flush=True)
    print(f"  acc     = {s['accuracy']['mean']:.3f} ± {s['accuracy']['std']:.3f}", flush=True)

    save_cv_predictions(results, out_dir)
    print(f"wrote {out_dir / 'predictions.npz'}", flush=True)


if __name__ == "__main__":
    main()
