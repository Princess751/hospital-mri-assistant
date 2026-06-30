from __future__ import annotations

import sys
from pathlib import Path

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F

from ml.config import CLASS_LABELS, CLASS_NAMES, DEFAULT_WEIGHTS
from ml.gradcam import GradCAM, heatmap_to_png, overlay_heatmap
from ml.model import load_model
from ml.preprocess import load_image_from_bytes, preprocess_for_model


@dataclass
class PredictionResult:
    predicted_class: str
    predicted_label: str
    confidence: float
    probabilities: dict[str, float]
    tumor_detected: bool
    heatmap_png: bytes | None
    model_version: str
    limitations: list[str]

    def to_dict(self) -> dict:
        data = asdict(self)
        data.pop("heatmap_png", None)
        return data


class TumorDetector:
    def __init__(self, weights_path: Path | None = None, device: str | None = None):
        self.weights_path = Path(weights_path or DEFAULT_WEIGHTS)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = None
        self._gradcam: GradCAM | None = None
        self.model_version = "unloaded"

    def load(self) -> None:
        if not self.weights_path.exists():
            raise FileNotFoundError(
                f"Model weights not found at {self.weights_path}. Run scripts/setup_lab.py first."
            )
        self.model = load_model(self.weights_path, self.device)
        self._gradcam = GradCAM(self.model, self.model.layer4)
        checkpoint = torch.load(self.weights_path, map_location="cpu", weights_only=False)
        if isinstance(checkpoint, dict):
            self.model_version = (
                f"resnet18-epoch-{checkpoint.get('epoch', '?')}-acc-{checkpoint.get('val_acc', 0):.2f}"
            )
        else:
            self.model_version = "resnet18"

    def predict(self, image_bytes: bytes, include_heatmap: bool = True) -> PredictionResult:
        if self.model is None:
            self.load()

        image = load_image_from_bytes(image_bytes)
        tensor = preprocess_for_model(image).to(self.device)
        tensor.requires_grad_(include_heatmap)

        with torch.set_grad_enabled(include_heatmap):
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=1).squeeze(0).detach().cpu().tolist()

        class_idx = int(max(range(len(probs)), key=lambda i: probs[i]))
        predicted_class = CLASS_NAMES[class_idx]
        confidence = probs[class_idx]

        heatmap_png = None
        if include_heatmap and self._gradcam is not None:
            cam = self._gradcam.generate(tensor, class_idx)
            heatmap_png = heatmap_to_png(overlay_heatmap(image, cam))

        return PredictionResult(
            predicted_class=predicted_class,
            predicted_label=CLASS_LABELS[predicted_class],
            confidence=round(confidence, 4),
            probabilities={CLASS_NAMES[i]: round(probs[i], 4) for i in range(len(CLASS_NAMES))},
            tumor_detected=predicted_class != "no_tumor",
            heatmap_png=heatmap_png,
            model_version=self.model_version,
            limitations=[
                "Decision-support only — not a standalone diagnostic device.",
                "Validate on hospital-specific scanners and protocols before clinical use.",
                "Grad-CAM highlights model attention, not confirmed tumor boundaries.",
                "Radiologist review is required for all findings.",
            ],
        )


if __name__ == "__main__":
    from ml.config import DATA_DIR

    detector = TumorDetector()
    detector.load()
    sample = next((DATA_DIR / "train" / "glioma").glob("*.*"), None)
    if sample is None:
        print("Model loaded. No sample image in data/train/glioma to demo.")
    else:
        result = detector.predict(sample.read_bytes())
        print(f"Demo on {sample.name}: {result.predicted_label} ({result.confidence:.1%})")
