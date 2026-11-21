"""Video indexing with YOLO11l: which of person, car, dog, laptop, cell phone appear in each second
of each video, and how many people are visible.

    python run.py --input <folder of .parquet files> --output results.jsonl

Adaptive sampling (K=5 frames per second; a second whose samples are not clear-cut runs every frame),
FP16 on CUDA. See README.md for how it works and experiments/ for the accuracy and speed measurements.
"""

import sys

from alethia.cli import main

if __name__ == "__main__":
    sys.exit(main("run"))
