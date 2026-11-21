"""Video indexing with YOLO11l, skip mode. Usage: python run_skip.py --input <folder of .parquet files> --output results.jsonl"""

import sys

from alethia.cli import main

if __name__ == "__main__":
    sys.exit(main("skip"))
