from __future__ import annotations

import sys
from pathlib import Path

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import io

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from ml.config import IMAGE_SIZE, MEAN, STD


def to_rgb(image: Image.Image) -> Image.Image:
    if image.mode == "L":
        return Image.merge("RGB", (image, image, image))
    if image.mode != "RGB":
        return image.convert("RGB")
    return image


def load_image_from_bytes(data: bytes) -> Image.Image:
    return to_rgb(Image.open(io.BytesIO(data)))


def apply_clahe(image: Image.Image) -> Image.Image:
    gray = np.array(image.convert("L"))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    rgb = np.stack([enhanced, enhanced, enhanced], axis=-1)
    return Image.fromarray(rgb)


def train_transforms():
    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(8),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )


def eval_transforms():
    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )


def preprocess_for_model(image: Image.Image, use_clahe: bool = True) -> torch.Tensor:
    if use_clahe:
        image = apply_clahe(image)
    return eval_transforms()(image).unsqueeze(0)


if __name__ == "__main__":
    print("ml.preprocess is a library module - not run directly.")
    print("Start the app:  uvicorn app.main:app --host 127.0.0.1 --port 8781")
    print("Or train:       python scripts/train_model.py")
