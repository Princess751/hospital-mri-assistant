"""Generate synthetic MRI-like images when public dataset download is unavailable."""

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ml.config import CLASS_NAMES, DATA_DIR

EXT = ".png"
COUNTS = {"train": 40, "val": 10}


def make_brain(class_name: str, seed: int) -> Image.Image:
    rng = random.Random(seed)
    arr = np.zeros((256, 256), dtype=np.uint8)
    cx, cy = 128 + rng.randint(-15, 15), 128 + rng.randint(-15, 15)
    for y in range(256):
        for x in range(256):
            d = ((x - cx) ** 2 + (y - cy) ** 2) ** 0.5
            if d < 95 + rng.randint(-8, 8):
                arr[y, x] = int(40 + rng.randint(0, 30))
    img = Image.fromarray(arr).convert("RGB")
    draw = ImageDraw.Draw(img)
    if class_name != "no_tumor":
        tx, ty = cx + rng.randint(-30, 30), cy + rng.randint(-30, 30)
        r = rng.randint(12, 28)
        color = (180, 60, 60) if class_name == "glioma" else (200, 120, 60) if class_name == "meningioma" else (160, 80, 180)
        draw.ellipse((tx - r, ty - r, tx + r, ty + r), fill=color)
    return img.filter(ImageFilter.GaussianBlur(radius=1.2))


def main() -> None:
    for split, count in COUNTS.items():
        for class_name in CLASS_NAMES:
            out = DATA_DIR / split / class_name
            out.mkdir(parents=True, exist_ok=True)
            for i in range(count):
                img = make_brain(class_name, seed=hash((split, class_name, i)) % 10_000)
                img.save(out / f"{class_name}_{split}_{i:04d}{EXT}")
    print(f"Synthetic data written to {DATA_DIR}")


if __name__ == "__main__":
    main()
