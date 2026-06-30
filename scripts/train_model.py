from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ml.config import CLASS_NAMES, DATA_DIR, DEFAULT_WEIGHTS, WEIGHTS_DIR
from ml.dataset import MRIDataset
from ml.model import build_model


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train(train)
    total_loss = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        if train:
            optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        if train:
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / max(total, 1), correct / max(total, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DATA_DIR)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output", type=Path, default=DEFAULT_WEIGHTS)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds = MRIDataset(args.data / "train", train=True)
    val_ds = MRIDataset(args.data / "val", train=False)
    if len(train_ds) == 0:
        raise SystemExit("No training data. Run: python scripts/download_dataset.py")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = build_model(pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_acc = 0.0
    history = []
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, True)
        with torch.no_grad():
            val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, False)
        history.append(
            {"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc, "val_loss": val_loss, "val_acc": val_acc}
        )
        print(
            f"Epoch {epoch}/{args.epochs} | train acc={train_acc:.3f} | val acc={val_acc:.3f}"
        )
        if val_acc >= best_acc:
            best_acc = val_acc
            WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_names": list(CLASS_NAMES),
                    "val_acc": val_acc,
                    "epoch": epoch,
                },
                args.output,
            )

    args.output.with_suffix(".meta.json").write_text(
        json.dumps({"best_val_acc": best_acc, "history": history}, indent=2),
        encoding="utf-8",
    )
    print(f"Best val accuracy: {best_acc:.3f}. Saved to {args.output}")


if __name__ == "__main__":
    main()
