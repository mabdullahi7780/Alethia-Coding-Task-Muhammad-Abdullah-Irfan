"""Video indexing with YOLO11l, full mode. Usage: python run_full.py --input <folder of .parquet files> --output results.jsonl"""

import sys

from alethia.cli import main

if __name__ == "__main__":
    sys.exit(main("full"))
