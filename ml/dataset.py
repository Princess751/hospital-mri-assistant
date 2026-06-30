from __future__ import annotations

import sys
from pathlib import Path

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image
from torch.utils.data import Dataset

from ml.config import CLASS_NAMES
from ml.preprocess import eval_transforms, train_transforms


class MRIDataset(Dataset):
    EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

    def __init__(self, root: Path, train: bool = True):
        self.samples: list[tuple[Path, int]] = []
        class_to_idx = {name: idx for idx, name in enumerate(CLASS_NAMES)}
        for class_name in CLASS_NAMES:
            class_dir = Path(root) / class_name
            if not class_dir.exists():
                continue
            label = class_to_idx[class_name]
            for path in sorted(class_dir.iterdir()):
                if path.suffix.lower() in self.EXTENSIONS:
                    self.samples.append((path, label))
        self.transform = train_transforms() if train else eval_transforms()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        image = Image.open(path)
        if image.mode != "RGB":
            image = image.convert("RGB")
        return self.transform(image), label
