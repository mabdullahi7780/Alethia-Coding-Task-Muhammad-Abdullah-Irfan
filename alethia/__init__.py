"""Video indexing with YOLO11l (Alethia AI take-home)."""

import os

# The timed run has no internet. Tell Ultralytics up front so it never tries to reach the network
# (it checks connectivity when imported). Must be set before ultralytics is imported anywhere.
os.environ.setdefault("YOLO_OFFLINE", "1")
