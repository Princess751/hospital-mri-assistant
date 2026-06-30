from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS_DIR = ROOT / "weights"
DATA_DIR = ROOT / "data"

IMAGE_SIZE = 224
NUM_CLASSES = 4
CLASS_NAMES = ("glioma", "meningioma", "no_tumor", "pituitary")
CLASS_LABELS = {
    "glioma": "Glioma",
    "meningioma": "Meningioma",
    "no_tumor": "No tumor detected",
    "pituitary": "Pituitary tumor",
}

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
DEFAULT_WEIGHTS = WEIGHTS_DIR / "tumor_classifier.pt"
