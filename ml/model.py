from __future__ import annotations

import sys
from pathlib import Path

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
from torchvision import models

from ml.config import NUM_CLASSES


def build_model(pretrained: bool = True) -> nn.Module:
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    net = models.resnet18(weights=weights)
    old_conv = net.conv1
    net.conv1 = nn.Conv2d(
        3,
        old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=False,
    )
    if pretrained:
        with torch.no_grad():
            net.conv1.weight.copy_(old_conv.weight)
    net.fc = nn.Linear(net.fc.in_features, NUM_CLASSES)
    return net


def load_model(weights_path, device: torch.device | None = None) -> nn.Module:
    device = device or torch.device("cpu")
    model = build_model(pretrained=False)
    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
    state = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


if __name__ == "__main__":
    print("ml.model is a library module - not run directly.")
    print("Start the app:  uvicorn app.main:app --host 127.0.0.1 --port 8781")
    print("Or train:       python scripts/train_model.py")
