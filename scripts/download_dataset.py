"""Download the public Brain Tumor MRI dataset (Kaggle mirror on GitHub)."""

from __future__ import annotations

import argparse
import random
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ml.config import CLASS_NAMES, DATA_DIR

URLS = [
    "https://github.com/SartajBhuvaji/Brain-Tumor-Classification-DataSet/archive/refs/heads/main.zip",
    "https://github.com/SartajBhuvaji/Brain-Tumor-Classification-DataSet/archive/refs/heads/master.zip",
    "https://github.com/sartajbhuvaji/brain-tumor-classification-mri/archive/refs/heads/main.zip",
]
CLASS_MAP = {
    "glioma": "glioma",
    "meningioma": "meningioma",
    "notumor": "no_tumor",
    "pituitary": "pituitary",
}
EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def prepared(data_dir: Path) -> bool:
    for name in CLASS_NAMES:
        folder = data_dir / "train" / name
        if not folder.exists() or not any(p.suffix.lower() in EXT for p in folder.iterdir()):
            return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DATA_DIR)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if prepared(args.output) and not args.force:
        print(f"Dataset ready at {args.output}")
        return

    cache = args.output / "_cache"
    zip_path = cache / "dataset.zip"
    extract = cache / "extracted"
    cache.mkdir(parents=True, exist_ok=True)

    if args.force:
        for split in ("train", "val"):
            split_dir = args.output / split
            if split_dir.exists():
                shutil.rmtree(split_dir)

    print("Downloading public Brain Tumor MRI dataset (~150 MB)...")
    last_error = None
    for url in URLS:
        try:
            print(f"  Trying {url}")
            urllib.request.urlretrieve(url, zip_path)
            last_error = None
            break
        except Exception as exc:
            last_error = exc
            print(f"  Failed: {exc}")
    if last_error is not None:
        raise last_error

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract)

    training = None
    for pattern in ("*/Training", "*/training", "*/train"):
        matches = list(extract.glob(pattern))
        if matches:
            training = matches[0]
            break
    if training is None:
        raise FileNotFoundError("Could not find Training/ folder in downloaded archive")
    rng = random.Random(args.seed)
    stats = {name: {"train": 0, "val": 0} for name in CLASS_NAMES}

    for src_name, class_name in CLASS_MAP.items():
        src = training / src_name
        if not src.exists():
            continue
        images = [p for p in src.iterdir() if p.suffix.lower() in EXT]
        rng.shuffle(images)
        val_count = max(1, int(len(images) * args.val_ratio)) if len(images) > 1 else 0
        val_set = set(images[:val_count])
        for img in images:
            split = "val" if img in val_set else "train"
            dest = args.output / split / class_name
            dest.mkdir(parents=True, exist_ok=True)
            shutil.copy2(img, dest / img.name)
            stats[class_name][split] += 1

    train_total = sum(v["train"] for v in stats.values())
    val_total = sum(v["val"] for v in stats.values())
    print(f"Done: {train_total} train, {val_total} val images")
    for name in CLASS_NAMES:
        s = stats[name]
        print(f"  {name}: {s['train']} train, {s['val']} val")


if __name__ == "__main__":
    main()
