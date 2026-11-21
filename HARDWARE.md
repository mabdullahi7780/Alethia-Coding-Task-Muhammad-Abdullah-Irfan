# Hardware

`results.jsonl.gz` was produced by one run of `python run.py --input <folder> --output results.jsonl`
(commit `3018200`) on a Kaggle notebook (`kaggle/full_run.ipynb`). The logs, environment record and
statistics of that run are in `experiments/phase4_full_run/`.

| | |
|---|---|
| Machine | Kaggle notebook, accelerator "GPU T4 x2" |
| GPU | 2x NVIDIA Tesla T4, 14.6 GB each (both used, one GPU process each) |
| CPU | Intel Xeon @ 2.00 GHz, 4 vCPUs (run.py used 2 decode processes: cores minus GPUs) |
| RAM | 31.3 GB |
| Disk | Google Cloud Persistent Disk (network block storage; the kernel reports it as non-rotational). The 50 Parquet files (23.1 GiB) were read from `/tmp` |
| OS | Ubuntu 24.04.4 LTS |
| Software | Python 3.13.16 (Kaggle's image), torch 2.14.1+cu126 (CUDA 12.6), ultralytics 8.4.216, av 18.1.0, installed with `pip install -r requirements.txt` |

## Run time

| | |
|---|---|
| Full run (`run.py`, start to finish) | **21,497 s = 5 h 58 min 17 s** (05:50:05 to 11:48:19, Kaggle clock, 21 Nov 2026) |
| Input | first 50 Parquet files, 1,650 videos, 461,757 video-seconds (128.3 h of video) |
| Output | 461,757 lines, 0 error lines, 0 format problems (`tools/check_results.py`) |
| Throughput | 21.5 video-seconds per second; 3,955,538 frames through the model (8.57 per video-second) |
| GPU time | per GPU about 17,300 s inferring and 3,500 to 3,700 s waiting for decoded frames |

Not included in the run time (separate, untimed steps in the notebook): downloading the 50 files
(109 s), `prepare.py`, and a validation run on a 96-video subset before the full run.

The GPUs waited for frames about 17% of the time, so on this machine the run was partly limited by
the 4 vCPUs that decode video. It was not measured on the grading machine (RTX 4090, 32 threads).
