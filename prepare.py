"""Preparation step (run once, with internet, before the timed run):

    python prepare.py

1. downloads the official YOLO11l weights (yolo11l.pt) next to run.py and checks the file's SHA-256
2. checks that PyTorch can see the NVIDIA GPU (a CPU-only run would be far too slow)
3. loads the model once on the GPU, so any first-use setup happens here and not in the timed run
"""

import hashlib
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
WEIGHTS = REPO / "yolo11l.pt"
SHA256 = "9ebd0e09d59811db4b1d61e2bc6730649608b1ac47f8dd01e2da6bca7c20023f"  # ultralytics assets v8.4.0


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    os.chdir(REPO)  # Ultralytics downloads "yolo11l.pt" into the current directory
    import numpy as np
    import torch
    from ultralytics import YOLO

    if not WEIGHTS.exists() or sha256(WEIGHTS) != SHA256:
        WEIGHTS.unlink(missing_ok=True)
        YOLO("yolo11l.pt")
    digest = sha256(WEIGHTS)
    print(f"weights: {WEIGHTS} sha256 {digest}")
    if digest != SHA256:
        print("ERROR: yolo11l.pt checksum does not match the official file", file=sys.stderr)
        return 1

    print(f"torch {torch.__version__} (CUDA {torch.version.cuda}); CUDA available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("ERROR: PyTorch cannot see an NVIDIA GPU; run.py would fall back to the CPU", file=sys.stderr)
        return 1
    for i in range(torch.cuda.device_count()):
        print(f"cuda:{i} {torch.cuda.get_device_name(i)}")

    model = YOLO(str(WEIGHTS))
    model.predict(np.zeros((360, 640, 3), np.uint8), imgsz=640, conf=0.4, iou=0.7, device="cuda:0",
                  quantize=16, verbose=False)
    print("model loads and runs on cuda:0 (FP16): ready")
    return 0


if __name__ == "__main__":
    sys.exit(main())
