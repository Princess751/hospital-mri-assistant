from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    subprocess.check_call(cmd, cwd=ROOT, env=env)


def main() -> None:
    py = sys.executable
    try:
        run([py, str(ROOT / "scripts" / "download_dataset.py")])
    except subprocess.CalledProcessError:
        print("Public dataset download failed — using synthetic fallback for demo.")
        run([py, str(ROOT / "scripts" / "generate_synthetic_data.py")])
    run([py, str(ROOT / "scripts" / "train_model.py"), "--epochs", "8"])
    weights = ROOT / "weights" / "tumor_classifier.pt"
    if not weights.exists():
        raise SystemExit("Training failed — weights not created.")
    print(f"\nReady. Weights: {weights}")
    print("Start: .\\run.bat")


if __name__ == "__main__":
    main()
