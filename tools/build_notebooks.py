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
    """Run a command, streaming its output live; with log=..., also save it to OUT_DIR/log."""
    print("$", " ".join(cmd), flush=True)
    t0 = time.perf_counter()
    proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1)
    fh = open(f"{OUT_DIR}/{log}", "w") if log else None
    for line in proc.stdout:
        print(line, end="", flush=True)
        if fh:
            fh.write(line)
    proc.wait()
    if fh:
        fh.close()
    print(f"[exit {proc.returncode}, {time.perf_counter() - t0:.1f}s]", flush=True)
    if check and proc.returncode != 0:
        raise RuntimeError(f"command failed: {cmd}")
    return proc


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


# ---------------------------------------------------------------- compare.ipynb
COMPARE_FILES = [f"data/train-{i:05d}-of-01357.parquet" for i in range(0, 48, 4)]  # 12 files spread over the 50
TOTAL_VIDEO_SECONDS = 457_200  # ~127 h in the 50 files (file 0 has 2.54 h; brief says ~125 h)


def compare() -> dict:
    return notebook([
        md('''
# Athletia Phase 3: every frame vs adaptive sampling, FP32 vs FP16

Subset: the first 8 videos of 12 Parquet files spread across the 50 (`train-00000`, `-00004`, ... `-00044`),
first 60 seconds of each (about 96 videos, 5,760 video-seconds). Three runs on both T4s, one after another:

1. `run_full.py` FP32 with `--dump-frames`: the reference (the brief's rule on every frame, identical to `predict()`)
2. `run_full.py --half`: FP16 speed, and its agreement with the FP32 reference
3. `run_skip.py --half`: adaptive sampling (K=5, 80/20, margin 0.30-0.50) speed and agreement

Then, on CPU: `tools/sweep.py` replays the skip rule from the FP32 per-frame detections for 90 settings.
Expected GPU time: about 50 minutes.

**Settings:** Accelerator GPU T4 x2, Internet on, Add-ons > Secrets > `HF_TOKEN` attached.
Run with **Save Version, Save & Run All**, then download `compare_outputs.zip` from the version's Output tab.
'''),
        code('''
OUT_DIR = "/kaggle/working/compare"
ARCHIVE_NAME = "compare_outputs"
'''),
        code(SETUP),
        code(ENV),
        code(HF_TOKEN),
        code(f'''
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

FILES = {COMPARE_FILES!r}
TOTAL_VIDEO_SECONDS = {TOTAL_VIDEO_SECONDS}
t0 = time.perf_counter()
for f in FILES:
    hf_hub_download("HuggingFaceFV/finevideo", f, repo_type="dataset", local_dir=DATA_DIR)
INPUT = os.path.join(DATA_DIR, "data")
print(len(os.listdir(INPUT)), "files,", round(sum(os.path.getsize(os.path.join(INPUT, x)) for x in os.listdir(INPUT)) / 2**30, 2),
      "GiB in", round(time.perf_counter() - t0), "s")
YOLO("yolo11l.pt")  # official weights into the repo folder
WEIGHTS = os.path.join(REPO_DIR, "yolo11l.pt")
'''),
        md("## Runs"),
        code(r'''
SUBSET = ["--input", INPUT, "--weights", WEIGHTS, "--videos-per-file", "8", "--max-seconds", "60", "--stats-every", "60"]
RUNS = {
    "full32": ["run_full.py", "--dump-frames", f"{OUT_DIR}/full32.frames"],
    "full16": ["run_full.py", "--half"],
    "skip16": ["run_skip.py", "--half"],
}
timing = {}
for name, args in RUNS.items():
    t0 = time.perf_counter()
    run([sys.executable, args[0], *args[1:], "--output", f"{OUT_DIR}/{name}.jsonl", *SUBSET], log=f"{name}_run.log")
    timing[name] = round(time.perf_counter() - t0, 1)
    run([sys.executable, "tools/check_results.py", f"{OUT_DIR}/{name}.jsonl"], log=f"{name}_check.log", check=False)
print(timing)
'''),
        md("## Accuracy against the FP32 every-frame reference"),
        code(r'''
pairs = {"full16_vs_full32": ("full32", "full16"), "skip16_vs_full32": ("full32", "skip16"),
         "skip16_vs_full16": ("full16", "skip16")}
for name, (ref, pred) in pairs.items():
    run([sys.executable, "tools/eval.py", "--ref", f"{OUT_DIR}/{ref}.jsonl", "--pred", f"{OUT_DIR}/{pred}.jsonl",
         "--json", f"{OUT_DIR}/{name}.json"], log=f"{name}.log")
'''),
        md("## Offline sweep of skip settings (CPU only)"),
        code(r'''
run([sys.executable, "tools/sweep.py", "--dump", f"{OUT_DIR}/full32.frames", "--ref", f"{OUT_DIR}/full32.jsonl",
     "--out", f"{OUT_DIR}/sweep.csv"], log="sweep.log")
'''),
        md("## Summary"),
        code(r'''
summary = {"timing_s": timing}
for name in RUNS:
    st = json.load(open(f"{OUT_DIR}/{name}.jsonl.stats.json"))
    gpus = list(st["gpu"].values())
    frames = sum(g["frames"] for g in gpus)
    secs = sum(g["seconds"] for g in gpus)
    rate = secs / st["wall_s"]
    summary[name] = {
        "wall_s": st["wall_s"], "video_seconds": secs, "frames_inferred": frames,
        "frames_per_video_second": round(frames / max(secs, 1), 2),
        "video_seconds_per_s": round(rate, 2),
        "projected_50_files_h_2xT4": round(TOTAL_VIDEO_SECONDS / rate / 3600, 1),
        "gpu_idle_s": [round(g["idle_s"], 1) for g in gpus], "gpu_infer_s": [round(g["infer_s"], 1) for g in gpus],
        "escalated": sum(g["escalated_seconds"] for g in gpus), "sampled": sum(g["sampled_seconds"] for g in gpus),
    }
for name in pairs:
    r = json.load(open(f"{OUT_DIR}/{name}.json"))
    summary[name] = {k: r[k] for k in ["seconds", "object_f1", "people_exact", "people_within1", "missing_seconds",
                                       "worst_videos_people_exact"]}
print(json.dumps(summary, indent=2))
json.dump(summary, open(f"{OUT_DIR}/summary.json", "w"), indent=2)
'''),
        code(ARCHIVE),
    ])


# ---------------------------------------------------------------- full_run.ipynb
SETUP_REQUIREMENTS = SETUP.replace(
    '''# Same versions as development; Kaggle's preinstalled torch is used (version recorded in env.json).
run([sys.executable, "-m", "pip", "install", "-q", "ultralytics==8.4.216", "av==18.1.0",
     "huggingface_hub", "imageio-ffmpeg", "pytest"])''',
    '''# Exactly the submitted requirements.txt (torch 2.14.1+cu126 etc.), so the results come from the
# environment the grader installs. huggingface_hub is only for downloading the data here.
run([sys.executable, "-m", "pip", "install", "-q", "-r", f"{REPO_DIR}/requirements.txt"])
run([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub"])''')
assert SETUP_REQUIREMENTS != SETUP


def full_run() -> dict:
    return notebook([
        md('''
# Athletia full run: all 50 Parquet files with run.py

1. installs the submitted `requirements.txt` and runs `prepare.py` (the README's preparation step)
2. downloads `train-00000` ... `train-00049` (about 25 GB) into `/tmp`
3. validation gate: `run.py` on the Phase 3 subset (96 videos x 60 s) scored against the committed FP32
   every-frame reference (`experiments/phase3_compare/full32.jsonl`); stops here if accuracy is off
4. the timed full run, exactly as the grader runs it: `python run.py --input <folder> --output results.jsonl`
5. format check of every line, `results.jsonl.gz`, and the numbers for `HARDWARE.md`

Expected: about 6 to 7 hours on GPU T4 x2.

**Settings:** Accelerator GPU T4 x2, Internet on, Add-ons > Secrets > `HF_TOKEN` attached.
Run with **Save Version, Save & Run All**. When it finishes, download `results.jsonl.gz` and
`full_run_outputs.zip` from the version's Output tab.
'''),
        code('''
OUT_DIR = "/kaggle/working/full"
ARCHIVE_NAME = "full_run_outputs"
'''),
        code(SETUP_REQUIREMENTS),
        code(ENV),
        code(HF_TOKEN),
        md("## Data and weights"),
        code(r'''
from concurrent.futures import ThreadPoolExecutor

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

FILES = [f"data/train-{i:05d}-of-01357.parquet" for i in range(50)]
t0 = time.perf_counter()
with ThreadPoolExecutor(8) as pool:
    paths = list(pool.map(lambda f: hf_hub_download("HuggingFaceFV/finevideo", f, repo_type="dataset",
                                                    local_dir=DATA_DIR), FILES))
download_s = time.perf_counter() - t0
INPUT = os.path.join(DATA_DIR, "data")
names = sorted(os.listdir(INPUT))
EXPECTED_VIDEOS = sum(pq.ParquetFile(os.path.join(INPUT, n)).metadata.num_rows for n in names)
size_gb = sum(os.path.getsize(os.path.join(INPUT, n)) for n in names) / 2**30
print(f"{len(names)} files, {size_gb:.1f} GiB, {EXPECTED_VIDEOS} videos, downloaded in {download_s:.0f} s")
assert len(names) == 50, names
run([sys.executable, "prepare.py"], log="prepare.log")
'''),
        md("## Validation gate (Phase 3 subset vs the FP32 every-frame reference)"),
        code(r'''
VAL_DIR = "/tmp/validation"
os.makedirs(VAL_DIR, exist_ok=True)
for i in range(0, 48, 4):
    name = f"train-{i:05d}-of-01357.parquet"
    if not os.path.exists(f"{VAL_DIR}/{name}"):
        os.symlink(f"{INPUT}/{name}", f"{VAL_DIR}/{name}")
run([sys.executable, "run.py", "--input", VAL_DIR, "--output", f"{OUT_DIR}/validation.jsonl",
     "--videos-per-file", "8", "--max-seconds", "60"], log="validation_run.log")
run([sys.executable, "tools/eval.py", "--ref", "experiments/phase3_compare/full32.jsonl",
     "--pred", f"{OUT_DIR}/validation.jsonl", "--json", f"{OUT_DIR}/validation_vs_full32.json"], log="validation_eval.log")
val = json.load(open(f"{OUT_DIR}/validation_vs_full32.json"))
# Our own sanity margins (from the Phase 3 sweep and the FP16 run), not the brief's thresholds.
assert val["seconds"] == 5543 and val["missing_seconds"] == 0, val
assert val["object_f1"] >= 0.998 and val["people_exact"] >= 0.995 and val["people_within1"] >= 0.999, val
print("validation passed")
'''),
        md("## The timed full run"),
        code(r'''
t0 = time.perf_counter()
run([sys.executable, "run.py", "--input", INPUT, "--output", f"{OUT_DIR}/results.jsonl", "--stats-every", "300"],
    log="full_run.log")
full_run_s = time.perf_counter() - t0
print(f"full run: {full_run_s:.0f} s = {full_run_s / 3600:.2f} h")
'''),
        md("## Checks, results.jsonl.gz, numbers for HARDWARE.md"),
        code(r'''
import gzip

run([sys.executable, "tools/check_results.py", f"{OUT_DIR}/results.jsonl", "--expect-videos", str(EXPECTED_VIDEOS)],
    log="results_check.log", check=False)
with open(f"{OUT_DIR}/results.jsonl", "rb") as src, gzip.open("/kaggle/working/results.jsonl.gz", "wb") as dst:
    shutil.copyfileobj(src, dst)
shutil.copy("/kaggle/working/results.jsonl.gz", f"{OUT_DIR}/results.jsonl.gz")

st = json.load(open(f"{OUT_DIR}/results.jsonl.stats.json"))
gpus = list(st["gpu"].values())
errors = [json.loads(l) for l in open(f"{OUT_DIR}/results.jsonl") if '"error"' in l]
summary = {
    "env": env,
    "download_s": round(download_s), "data_gib": round(size_gb, 1), "videos": EXPECTED_VIDEOS,
    "full_run_s": round(full_run_s), "full_run_h": round(full_run_s / 3600, 2),
    "lines": st["lines"], "video_seconds": sum(g["seconds"] for g in gpus),
    "frames_inferred": sum(g["frames"] for g in gpus),
    "frames_per_video_second": round(sum(g["frames"] for g in gpus) / max(sum(g["seconds"] for g in gpus), 1), 2),
    "escalated_seconds": sum(g["escalated_seconds"] for g in gpus), "sampled_seconds": sum(g["sampled_seconds"] for g in gpus),
    "gpu_idle_s": [round(g["idle_s"]) for g in gpus], "gpu_infer_s": [round(g["infer_s"]) for g in gpus],
    "error_videos": len(errors), "errors": errors[:50],
    "validation": val,
    "results_gz_mb": round(os.path.getsize("/kaggle/working/results.jsonl.gz") / 2**20, 1),
}
print(json.dumps({k: v for k, v in summary.items() if k != "errors"}, indent=2))
json.dump(summary, open(f"{OUT_DIR}/summary.json", "w"), indent=2)
'''),
        code(r'''
os.remove(f"{OUT_DIR}/results.jsonl")  # the gzip copy is kept; keeps the archive small
archive = shutil.make_archive(f"/kaggle/working/{ARCHIVE_NAME}", "zip", OUT_DIR)
print("Archive:", archive, os.path.getsize(archive), "bytes")
print(sorted(os.listdir(OUT_DIR)))
'''),
    ])


def main() -> None:
    KAGGLE_DIR.mkdir(parents=True, exist_ok=True)
    for name, build in {"smoke.ipynb": smoke, "compare.ipynb": compare, "full_run.ipynb": full_run}.items():
        out = KAGGLE_DIR / name
        out.write_text(json.dumps(build(), indent=1) + "\n")
        print("wrote", out)


if __name__ == "__main__":
    main()
