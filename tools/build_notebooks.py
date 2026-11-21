"""Write the Kaggle notebooks (PA3 pattern: clone the public repo, run, zip the outputs).

Standard library only:
    python3 tools/build_notebooks.py

Writes:
    kaggle/smoke.ipynb   Phase 2 smoke test on a Kaggle GPU: environment, unit tests, the batched-vs-predict
                         equivalence test on CUDA, damaged-video handling, and a small run of both
                         run_full.py and run_skip.py on the first 8 videos (30 s each) of one Parquet file,
                         scored against each other.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
KAGGLE_DIR = REPO / "kaggle"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.strip("\n").splitlines(keepends=True),
    }


def notebook(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


# ---------------------------------------------------------------- shared cells
SETUP = r'''
import json
import os
import shutil
import subprocess
import sys
import time

REPO_URL = "https://github.com/mabdullahi7780/Alethia-Coding-Task-Muhammad-Abdullah-Irfan.git"
BRANCH = "main"
REPO_DIR = "/tmp/alethia"  # outside /kaggle/working, so the output holds only results
DATA_DIR = "/tmp/finevideo"  # Parquet files live here, never in /kaggle/working


def run(cmd, cwd=None, check=True, log=None, env=None):
    """Run a command, print it, optionally tee its output to a log file in OUT_DIR."""
    print("$", " ".join(cmd), flush=True)
    t0 = time.perf_counter()
    if log is None:
        res = subprocess.run(cmd, cwd=cwd, env=env)
    else:
        res = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True)
        text = res.stdout + res.stderr
        with open(f"{OUT_DIR}/{log}", "w") as fh:
            fh.write(text)
        print("\n".join(text.splitlines()[-25:]))
    print(f"[exit {res.returncode}, {time.perf_counter() - t0:.1f}s]", flush=True)
    if check and res.returncode != 0:
        raise RuntimeError(f"command failed: {cmd}")
    return res


if not os.path.isdir(REPO_DIR):
    run(["git", "clone", "--branch", BRANCH, REPO_URL, REPO_DIR])
run(["git", "log", "-1", "--format=commit %H%nsubject %s"], cwd=REPO_DIR)

# Same versions as development; Kaggle's preinstalled torch is used (version recorded in env.json).
run([sys.executable, "-m", "pip", "install", "-q", "ultralytics==8.4.216", "av==18.1.0",
     "huggingface_hub", "imageio-ffmpeg", "pytest"])

os.chdir(REPO_DIR)
sys.path.insert(0, REPO_DIR)
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
'''

ENV = r'''
import importlib.metadata as metadata
import platform

import torch


def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout.strip()


env = {
    "commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_DIR, capture_output=True, text=True).stdout.strip(),
    "python": platform.python_version(),
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "ultralytics": metadata.version("ultralytics"),
    "av": metadata.version("av"),
    "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
    "gpu_mem_gb": [round(torch.cuda.get_device_properties(i).total_memory / 2**30, 1) for i in range(torch.cuda.device_count())],
    "cpu_model": sh("lscpu | grep 'Model name' | sed 's/.*: *//'"),
    "cpu_count": os.cpu_count(),
    "ram_gb": round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30, 1),
    "disk_free_gb": {p: round(shutil.disk_usage(p).free / 2**30, 1) for p in ["/kaggle/working", "/tmp"]},
    "disk_type": sh("lsblk -d -o NAME,ROTA,SIZE,MODEL"),
    "os": sh("cat /etc/os-release | grep PRETTY_NAME"),
}
print(json.dumps(env, indent=2))
assert torch.cuda.device_count() >= 1, "No GPU: set the accelerator to GPU T4 x2"
with open(f"{OUT_DIR}/env.json", "w") as handle:
    json.dump(env, handle, indent=2)
'''

HF_TOKEN = r'''
from kaggle_secrets import UserSecretsClient

# Read from Kaggle Secrets (Add-ons > Secrets > HF_TOKEN). Never printed, never written to disk.
os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
print("HF_TOKEN loaded:", os.environ["HF_TOKEN"].startswith("hf_"))
'''

ARCHIVE = r'''
archive = shutil.make_archive(f"/kaggle/working/{ARCHIVE_NAME}", "zip", OUT_DIR)
print("Archive:", archive, os.path.getsize(archive), "bytes")
print(sorted(os.listdir(OUT_DIR)))
'''


# ---------------------------------------------------------------- smoke.ipynb
def smoke() -> dict:
    return notebook([
        md('''
# Athletia smoke test on a Kaggle GPU

Checks the whole pipeline on real CUDA hardware before any long run:

1. environment (GPUs, CPU, RAM, disk) for `HARDWARE.md`
2. unit tests (per-second rules, decoding, rotation, letterbox)
3. equivalence test: our batched inference vs plain `model.predict()` on real frames, on CUDA
4. damaged videos: truncated, garbage and empty bytes must give error lines without stopping the run
5. a small run of `run_full.py` (every frame) and `run_skip.py` (adaptive sampling) on the first 8 videos
   of `train-00000` (first 30 s each), both GPUs, then the skip output scored against the full output

Expected GPU time: about 15 minutes.

**Settings:** Accelerator GPU T4 x2, Internet on, Add-ons > Secrets > `HF_TOKEN` attached.
Run with **Save Version, Save & Run All**, then download `smoke_outputs.zip` from the version's Output tab.
'''),
        code('''
OUT_DIR = "/kaggle/working/smoke"
ARCHIVE_NAME = "smoke_outputs"
'''),
        code(SETUP),
        code(ENV),
        code(HF_TOKEN),
        code(r'''
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

t0 = time.perf_counter()
PARQUET = hf_hub_download("HuggingFaceFV/finevideo", "data/train-00000-of-01357.parquet",
                          repo_type="dataset", local_dir=DATA_DIR)
download_s = time.perf_counter() - t0
INPUT = os.path.dirname(PARQUET)
YOLO("yolo11l.pt")  # downloads the official weights into the repo folder (run.py finds them there)
WEIGHTS = os.path.join(REPO_DIR, "yolo11l.pt")
print("parquet", PARQUET, round(os.path.getsize(PARQUET) / 2**20), "MiB in", round(download_s, 1), "s")
print("weights", WEIGHTS, os.path.exists(WEIGHTS))
'''),
        md("## 1. Unit tests"),
        code(r'''
run([sys.executable, "-m", "pytest", "-q", "tests/test_rules.py", "tests/test_decode.py"], log="unit_tests.log")
'''),
        md("## 2. Equivalence: batched inference vs `model.predict()` on CUDA (every video of train-00000, 40 frames each)"),
        code(r'''
eq_env = dict(os.environ, ALETHIA_TEST_PARQUET=PARQUET, ALETHIA_WEIGHTS=WEIGHTS, ALETHIA_DEVICE="cuda:0",
              ALETHIA_FRAMES_PER_VIDEO="40", ALETHIA_TEST_BATCH="16")
run([sys.executable, "-m", "pytest", "-q", "-s", "tests/test_infer.py"], log="equivalence.log", env=eq_env)
'''),
        md("## 3. Damaged videos"),
        code(r'''
import pyarrow as pa
import pyarrow.parquet as pq

DAMAGED_DIR = "/tmp/damaged"
os.makedirs(DAMAGED_DIR, exist_ok=True)
good = []
for i, batch in enumerate(pq.ParquetFile(PARQUET).iter_batches(batch_size=1, columns=["mp4"])):
    good.append(batch.column(0)[0].as_py())
    if i == 1:
        break
rows = [
    good[0][: len(good[0]) * 2 // 5],  # :0 truncated to 40% of its bytes
    b"this is not a video file" * 100,  # :1 garbage bytes
    None,                               # :2 missing bytes
    good[1],                            # :3 intact video, must still be processed normally
]
pq.write_table(pa.table({"mp4": pa.array(rows, pa.binary())}), f"{DAMAGED_DIR}/damaged-00000.parquet")
run([sys.executable, "run_full.py", "--input", DAMAGED_DIR, "--output", f"{OUT_DIR}/damaged.jsonl",
     "--weights", WEIGHTS, "--max-seconds", "10"], log="damaged_run.log")
run([sys.executable, "tools/check_results.py", f"{OUT_DIR}/damaged.jsonl", "--expect-videos", "4"],
    log="damaged_check.log", check=False)
for line in open(f"{OUT_DIR}/damaged.jsonl"):
    if '"error"' in line or '"second": 0,' in line:
        print(line.strip())
'''),
        md("## 4. Small run: every frame vs adaptive sampling (first 8 videos, 30 s each, both GPUs)"),
        code(r'''
SUBSET = ["--videos-per-file", "8", "--max-seconds", "30", "--stats-every", "15"]
timing = {}
for mode in ["full", "skip"]:
    t0 = time.perf_counter()
    run([sys.executable, f"run_{mode}.py", "--input", INPUT, "--output", f"{OUT_DIR}/{mode}.jsonl",
         "--weights", WEIGHTS, *SUBSET], log=f"{mode}_run.log")
    timing[mode] = round(time.perf_counter() - t0, 1)
print(timing)
'''),
        code(r'''
for mode in ["full", "skip"]:
    run([sys.executable, "tools/check_results.py", f"{OUT_DIR}/{mode}.jsonl", "--expect-videos", "8"],
        log=f"{mode}_check.log", check=False)
run([sys.executable, "tools/eval.py", "--ref", f"{OUT_DIR}/full.jsonl", "--pred", f"{OUT_DIR}/skip.jsonl",
     "--json", f"{OUT_DIR}/skip_vs_full.json"], log="skip_vs_full.log")

summary = {"timing_s": timing, "download_s": round(download_s, 1)}
for mode in ["full", "skip"]:
    st = json.load(open(f"{OUT_DIR}/{mode}.jsonl.stats.json"))
    gpus = st["gpu"].values()
    frames = sum(g["frames"] for g in gpus)
    secs = sum(g["seconds"] for g in gpus)
    summary[mode] = {
        "wall_s": st["wall_s"], "workers": st["workers"], "devices": st["devices"],
        "video_seconds": secs, "frames_inferred": frames,
        "frames_per_video_second": round(frames / max(secs, 1), 2),
        "video_seconds_per_s": round(secs / st["wall_s"], 2),
        "gpu_idle_s": [round(g["idle_s"], 1) for g in gpus],
        "gpu_infer_s": [round(g["infer_s"], 1) for g in gpus],
        "escalated_of_sampled": [g["escalated_seconds"] for g in gpus], "sampled": [g["sampled_seconds"] for g in gpus],
    }
summary["skip_vs_full"] = json.load(open(f"{OUT_DIR}/skip_vs_full.json"))
print(json.dumps(summary, indent=2))
json.dump(summary, open(f"{OUT_DIR}/summary.json", "w"), indent=2)
'''),
        code(ARCHIVE),
    ])


def main() -> None:
    KAGGLE_DIR.mkdir(parents=True, exist_ok=True)
    for name, build in {"smoke.ipynb": smoke}.items():
        out = KAGGLE_DIR / name
        out.write_text(json.dumps(build(), indent=1) + "\n")
        print("wrote", out)


if __name__ == "__main__":
    main()
