"""Parallel video indexing engine shared by run_full.py and run_skip.py.

Processes
    parent        reads the Parquet files (one row group at a time) and hands each video's bytes to a
                  bounded task queue; watches workers (restarts a crashed or hung one); merges output.
    decode worker one per CPU core (auto). Takes whole videos, decodes every frame, converts to
                  letterboxed BGR and writes each frame into its own shared-memory slot pool. Sends one
                  small message per second of video ("block") to its GPU process.
    GPU process   one per GPU. Batches frames from all of its workers by frame shape, runs YOLO11l,
                  applies the per-second rule, writes each video's lines, and hands the slots back.

Modes
    full  every frame of every second goes through the model; the brief's exact section 5 rule.
    skip  K evenly spaced frames per second first; if the second is clear-cut (alethia.rules.adaptive_decision)
          it is decided, otherwise the remaining frames of that second are run and the exact rule applied.

Frames of different videos share a batch; every frame carries (video, second, index) so results go back
to the right second. A video's lines are written once the whole video is done (or failed), together with
an entry in a "done" file, which is what makes the run resumable.
"""

from __future__ import annotations

import json
import os
import queue
import sys
import threading
import time
import traceback
from collections import deque
from dataclasses import dataclass, field
from multiprocessing import shared_memory
from pathlib import Path

import numpy as np

SLOT_H = SLOT_W = 640
SLOT_BYTES = SLOT_H * SLOT_W * 3
MAX_BLOCK = 64  # frames per message; a second with more frames is sent in several chunks


@dataclass
class Options:
    input: str
    output: str
    mode: str = "full"  # "full" or "skip"
    weights: str = "yolo11l.pt"
    device: str | None = None  # None: every CUDA GPU, else MPS, else CPU
    half: bool = False
    workers: int = 0  # 0: one per CPU core, minus one per GPU process
    batch: int = 32
    slots: int = 160  # frame slots per worker (each up to 640x640x3)
    decode_threads: int = 1
    samples: int = 5
    hi: float = 0.8
    lo: float = 0.2
    margin: tuple[float, float] = (0.3, 0.5)
    use_margin: bool = True
    people_rule: str = "all-equal"  # or "majority"
    dump_frames: str | None = None  # write per-frame detections here (for offline tuning)
    max_seconds: float = 0  # evaluation only: stop each video after this many seconds
    videos_per_file: int = 0  # evaluation only: first N videos of each Parquet file
    hang_seconds: float = 300
    stats_every: float = 30
    quiet: bool = False


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


# ====================================================================== decode worker
def _debug_hooks() -> None:
    """kill -USR1 <pid> prints every thread's Python stack to stderr (for diagnosing stalls)."""
    import faulthandler
    import signal

    faulthandler.register(signal.SIGUSR1, all_threads=True)


def worker_main(wid, task_q, out_q, free_q, shm_name, n_slots, gens, beats, waiting, opts: Options):
    _debug_hooks()
    import cv2

    cv2.setNumThreads(1)
    from .decode import iter_frames, to_model_input

    shm = shared_memory.SharedMemory(name=shm_name)
    buf = np.ndarray((n_slots, SLOT_BYTES), np.uint8, shm.buf)
    gen = gens[wid]
    # Send a second's frames early once half the slot pool is in it, so a worker can never hold
    # every slot while waiting for one more (a long second at a high frame rate would deadlock).
    chunk = max(1, min(MAX_BLOCK, n_slots // 2))

    def get_slot() -> int:
        waiting[wid] = 1
        while True:
            g, s = free_q.get()
            if g == gen:
                break  # slots from an earlier, crashed incarnation of this worker are ignored
        waiting[wid] = 0
        beats[wid] = time.time()
        return s

    while True:
        waiting[wid] = 1
        task = task_q.get()
        waiting[wid] = 0
        if task is None:
            break
        key, data, err = task
        out_q.put(("start", wid, gen, key))
        beats[wid] = time.time()
        if data is None:
            out_q.put(("end", wid, gen, key, err or "empty mp4 bytes"))
            continue
        cur_sec, frames, chunked = None, [], False

        def flush(final: bool):
            nonlocal frames, chunked
            if frames or (final and chunked):
                out_q.put(("block", wid, gen, key, cur_sec, frames, final))
            frames = []
            chunked = not final

        error = None
        try:
            for sec, frame in iter_frames(data, opts.decode_threads):
                beats[wid] = time.time()
                if opts.max_seconds and sec >= opts.max_seconds:
                    break
                if cur_sec is not None and sec != cur_sec:
                    flush(final=True)
                cur_sec = sec
                img = to_model_input(frame)
                h, w = img.shape[:2]
                slot = get_slot()
                buf[slot, : h * w * 3] = img.reshape(-1)
                frames.append((slot, h, w))
                if len(frames) >= chunk:
                    flush(final=False)
        except Exception as exc:  # damaged video: keep what was decoded, then one error line
            error = f"could not decode video: {type(exc).__name__}: {exc}"[:200]
        if cur_sec is not None:
            flush(final=True)
        out_q.put(("end", wid, gen, key, error))
    del buf
    shm.close()


# ====================================================================== GPU process
@dataclass
class Second:
    key: str
    sec: int
    frames: list = field(default_factory=list)  # (wid, gen, slot, h, w)
    dets: list = field(default_factory=list)  # FrameDet or None
    raw: list = field(default_factory=list)  # per-frame [(bit, conf)] when dumping
    pending: int = 0
    final: bool = False
    chunked: bool = False
    stage: str = "full"  # "sample" while waiting for the K sampled frames (skip mode)
    samples: list = field(default_factory=list)
    queued: int = 0  # frames already put in a bucket
    dropped: bool = False


@dataclass
class Video:
    key: str
    wid: int
    gen: int
    lines: dict = field(default_factory=dict)  # second -> (objects, people)
    open: dict = field(default_factory=dict)  # second -> Second
    ended: bool = False
    error: str | None = None


def gpu_main(g, device, opts: Options, in_q, free_qs, shm_info, part_path, done_path, ctrl_q, stats_path):
    _debug_hooks()
    try:
        _gpu_loop(g, device, opts, in_q, free_qs, shm_info, part_path, done_path, ctrl_q, stats_path)
    except Exception:
        ctrl_q.put(("gpu_error", g, traceback.format_exc()))
        raise


def _gpu_loop(g, device, opts, in_q, free_qs, shm_info, part_path, done_path, ctrl_q, stats_path):
    from .infer import Detector
    from .rules import CLASS_TO_BIT, adaptive_decision, exact_second, sample_indices

    if opts.dump_frames:
        floor = 0.25  # keep near-threshold detections for offline tuning
    elif opts.mode == "skip" and opts.use_margin:
        floor = opts.margin[0]
    else:
        floor = 0.4  # exactly what model.predict(conf=0.4) runs
    half = opts.half and str(device).startswith("cuda")  # FP16 only on CUDA; FP32 elsewhere
    det = Detector(opts.weights, device=device, half=half, conf_floor=floor, margin=opts.margin)
    shms = {w: shared_memory.SharedMemory(name=name) for w, (name, _) in shm_info.items()}
    bufs = {w: np.ndarray((n, SLOT_BYTES), np.uint8, shms[w].buf) for w, (_, n) in shm_info.items()}
    ctrl_q.put(("ready", g, f"{det.device} {'fp16' if det.fp16 else 'fp32'}"))

    part = open(part_path, "a")
    done = open(done_path, "a")
    dump = open(f"{opts.dump_frames}.gpu{g}", "a") if opts.dump_frames else None
    videos: dict[str, Video] = {}
    current: dict[int, str] = {}  # worker -> key of the video it is decoding
    buckets: dict[tuple[int, int], deque] = {}
    st = dict(frames=0, batches=0, seconds=0, sampled_seconds=0, escalated_seconds=0, videos=0,
              errors=0, idle_s=0.0, infer_s=0.0, decided_from_samples=0)
    t_start = last_report = time.time()
    finishing = False

    def enqueue(s: Second, idxs, front=False):
        for i in idxs:
            h, w = s.frames[i][3], s.frames[i][4]
            b = buckets.setdefault((h, w), deque())
            (b.appendleft if front else b.append)((s, i))
            s.pending += 1

    def release(s: Second):
        """Hand back the slots of frames that never went through the model (inferred ones are freed at once)."""
        for (wid, gen, slot, _, _), d in zip(s.frames, s.dets):
            if d is None:
                free_qs[wid].put((gen, slot))

    def emit(v: Video):
        lines = []
        last = max(v.lines) if v.lines else -1
        for sec in range(last + 1):
            objects, people = v.lines.get(sec, ([], 0))
            lines.append(json.dumps({"video": v.key, "second": sec, "objects": objects, "people": people}))
        error = v.error
        if not v.lines and not error:
            error = "could not decode video: no frames"
        if error:
            lines.append(json.dumps({"video": v.key, "error": error}))
            st["errors"] += 1
        part.write("\n".join(lines) + "\n")
        part.flush()
        done.write(v.key + "\n")
        done.flush()
        st["videos"] += 1
        videos.pop(v.key, None)

    def maybe_emit(v: Video):
        if v.ended and not v.open:
            emit(v)

    def finalize(s: Second, result):
        v = videos[s.key]
        v.lines[s.sec] = result
        st["seconds"] += 1
        release(s)
        if dump is not None:
            dump.write(json.dumps({"video": s.key, "second": s.sec, "frames": s.raw}) + "\n")
        del v.open[s.sec]
        maybe_emit(v)

    def resolve(s: Second):
        if s.stage == "sample":
            decision = adaptive_decision([s.dets[i] for i in s.samples], opts.hi, opts.lo, opts.use_margin,
                                         opts.people_rule)
            if decision is not None:
                st["decided_from_samples"] += 1
                finalize(s, decision)
                return
            st["escalated_seconds"] += 1
            s.stage = "full"
            rest = [i for i in range(len(s.frames)) if s.dets[i] is None]
            enqueue(s, rest, front=True)
            if rest:
                return
        finalize(s, exact_second(s.dets))

    def on_block(wid, gen, key, sec, frames, final):
        v = videos.get(key)
        if v is None or v.gen != gen:
            for slot, _, _ in frames:
                free_qs[wid].put((gen, slot))
            return
        s = v.open.get(sec)
        if s is None:
            s = v.open[sec] = Second(key, sec)
        start = len(s.frames)
        s.frames.extend((wid, gen, slot, h, w) for slot, h, w in frames)
        s.dets.extend([None] * len(frames))
        s.raw.extend([None] * len(frames))
        if not final:
            s.chunked = True
        s.final = final
        new = list(range(start, len(s.frames)))
        if opts.mode == "full" or s.chunked:
            enqueue(s, new)
        elif final:
            n = len(s.frames)
            if n <= opts.samples:
                enqueue(s, new)
            else:
                st["sampled_seconds"] += 1
                s.stage = "sample"
                s.samples = sample_indices(n, opts.samples)
                enqueue(s, s.samples)
        if s.final and s.pending == 0:
            resolve(s)

    def on_message(msg):
        nonlocal finishing
        kind = msg[0]
        if kind == "block":
            on_block(*msg[1:])
        elif kind == "start":
            _, wid, gen, key = msg
            videos[key] = Video(key, wid, gen)
            current[wid] = key
        elif kind == "end":
            _, wid, gen, key, error = msg
            v = videos.get(key)
            if v is not None and v.gen == gen:
                v.ended, v.error = True, error
                maybe_emit(v)
        elif kind == "dead":
            _, wid, old_gen = msg
            key = current.pop(wid, None)
            v = videos.get(key) if key else None
            if v is not None and v.gen == old_gen:
                for s in v.open.values():
                    s.dropped = True  # its slots belong to the dead worker; never handed back
                v.open.clear()
                v.ended = True
                v.error = "could not decode video: decoder crashed or hung"
                emit(v)
        elif kind == "finish":
            finishing = True

    def run_batch():
        (h, w), b = max(buckets.items(), key=lambda kv: len(kv[1]))
        items = []
        while b and len(items) < opts.batch:
            s, i = b.popleft()
            if not s.dropped:
                items.append((s, i))
        if not b:
            del buckets[(h, w)]
        if not items:
            return
        batch = np.empty((len(items), h, w, 3), np.uint8)
        for j, (s, i) in enumerate(items):
            wid, _, slot, _, _ = s.frames[i]
            batch[j] = bufs[wid][slot, : h * w * 3].reshape(h, w, 3)
        t0 = time.perf_counter()
        dets, raw = det(batch)
        st["infer_s"] += time.perf_counter() - t0
        st["frames"] += len(items)
        st["batches"] += 1
        for (s, i), d, r in zip(items, dets, raw):
            s.dets[i] = d
            wid, gen, slot, _, _ = s.frames[i]
            free_qs[wid].put((gen, slot))  # pixels no longer needed: free the slot right away
            if dump is not None:
                s.raw[i] = [[CLASS_TO_BIT[int(c)], round(float(p), 6)]
                            for p, c in r[:, 4:6].tolist() if int(c) in CLASS_TO_BIT]
            s.pending -= 1
        for s in {id(s): s for s, _ in items}.values():
            if s.final and s.pending == 0 and not s.dropped and s.sec in videos.get(s.key, Video("", 0, 0)).open:
                resolve(s)

    def report(final=False):
        el = time.time() - t_start
        log(f"gpu{g} {'FINAL ' if final else ''}{el:7.0f}s | videos {st['videos']} | video-seconds {st['seconds']} "
            f"({st['seconds'] / max(el, 1e-9):.1f}/s) | frames inferred {st['frames']} ({st['frames'] / max(el, 1e-9):.0f}/s) "
            f"| escalated {st['escalated_seconds']}/{st['sampled_seconds']} | idle {st['idle_s']:.0f}s | infer {st['infer_s']:.0f}s")

    while True:
        pending = sum(len(b) for b in buckets.values())
        # take every message that is already waiting; block only when there is nothing to run
        while True:
            try:
                if pending == 0 and not finishing:
                    t0 = time.perf_counter()
                    msg = in_q.get(timeout=0.2)
                    st["idle_s"] += time.perf_counter() - t0
                else:
                    msg = in_q.get_nowait()
            except queue.Empty:
                break
            on_message(msg)
            pending = sum(len(b) for b in buckets.values())
            if pending >= opts.batch:
                break
        if pending:
            run_batch()
        elif finishing and not videos:
            break
        elif finishing:
            # workers are gone; anything still open can never complete
            for v in list(videos.values()):
                v.open.clear()
                v.ended = True
                v.error = v.error or "could not decode video: worker stopped"
                emit(v)
        if not opts.quiet and time.time() - last_report >= opts.stats_every:
            report()
            last_report = time.time()

    report(final=True)
    st["wall_s"] = time.time() - t_start
    st["device"] = det.device
    with open(stats_path, "w") as fh:
        json.dump(st, fh, indent=2)
    part.close()
    done.close()
    if dump is not None:
        dump.close()
    for s in shms.values():
        s.close()


# ====================================================================== parent
def _load_done(parts_dir: Path) -> set[str]:
    done = set()
    for f in parts_dir.glob("*.done"):
        done.update(line.strip() for line in f.read_text().splitlines() if line.strip())
    return done


def _sanitize_parts(parts_dir: Path, done: set[str]) -> None:
    """After a crash, keep only lines of videos recorded as done (drops half-written tails)."""
    for f in parts_dir.glob("*.jsonl"):
        keep = []
        for line in f.read_text().splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("video") in done:
                keep.append(line)
        f.write_text("\n".join(keep) + ("\n" if keep else ""))


def _feed(files, task_q, done, opts: Options, n_workers, counters):
    try:
        _feed_files(files, task_q, done, opts, counters)
    except Exception:
        log("reader failed:\n" + traceback.format_exc())
    finally:
        for _ in range(n_workers):
            task_q.put(None)


def _feed_files(files, task_q, done, opts: Options, counters):
    import pyarrow.parquet as pq

    for path in files:
        name = path.name
        try:
            pf = pq.ParquetFile(path)
        except Exception as exc:
            task_q.put((f"{name}:0", None, f"could not read parquet file: {type(exc).__name__}"))
            counters["videos"] += 1
            continue
        row = 0
        for rg in range(pf.num_row_groups):
            n_rows = pf.metadata.row_group(rg).num_rows
            if opts.videos_per_file and row >= opts.videos_per_file:
                break
            try:
                column = pf.read_row_group(rg, columns=["mp4"]).column("mp4")
            except Exception as exc:
                column, err = None, f"could not read parquet row group: {type(exc).__name__}"
            for i in range(n_rows):
                if opts.videos_per_file and row >= opts.videos_per_file:
                    break
                key = f"{name}:{row}"
                row += 1
                if key in done:
                    continue
                if column is None:
                    task_q.put((key, None, err))
                else:
                    try:
                        data = column[i].as_py()
                        task_q.put((key, data, None if data else "empty mp4 bytes"))
                    except Exception as exc:
                        task_q.put((key, None, f"could not read video bytes: {type(exc).__name__}"))
                counters["videos"] += 1
            column = None


def _detect_devices() -> list[str]:
    """Every CUDA GPU, else MPS, else CPU. Probed in a short-lived child so the parent never holds torch."""
    import subprocess

    code = ("import torch;"
            "print(' '.join(f'cuda:{i}' for i in range(torch.cuda.device_count())) if torch.cuda.is_available() "
            "else ('mps' if torch.backends.mps.is_available() else 'cpu'))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    return out.split() or ["cpu"]


def run(opts: Options) -> int:
    import multiprocessing as mp
    import signal

    # On SIGTERM, unwind normally so the finally-block and daemon cleanup stop every child process.
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    _debug_hooks()
    t0 = time.time()
    src = Path(opts.input)
    files = sorted(src.glob("*.parquet")) if src.is_dir() else [src]
    if not files:
        log(f"no .parquet files in {src}")
        return 2
    out = Path(opts.output)
    parts_dir = out.parent / (out.name + ".parts")
    parts_dir.mkdir(parents=True, exist_ok=True)
    done = _load_done(parts_dir)
    if done:
        _sanitize_parts(parts_dir, done)
        log(f"resuming: {len(done)} videos already done")

    if opts.device:
        devices = [opts.device]  # no torch import in the parent: it is only needed in GPU processes
    else:
        devices = _detect_devices()
    n_workers = opts.workers or max(1, (os.cpu_count() or 2) - len(devices))

    log(f"mode={opts.mode} devices={devices} workers={n_workers} batch={opts.batch} files={len(files)}"
        + (f" samples={opts.samples} hi={opts.hi} lo={opts.lo} margin={opts.margin if opts.use_margin else 'off'}"
           f" people={opts.people_rule}" if opts.mode == "skip" else "") + (" half" if opts.half else ""))

    ctx = mp.get_context("spawn")
    task_q = ctx.Queue(maxsize=max(4, 2 * n_workers))
    ctrl_q = ctx.Queue()
    gin = [ctx.Queue() for _ in devices]
    free_qs = [ctx.Queue() for _ in range(n_workers)]
    gens = ctx.Array("i", n_workers, lock=False)
    beats = ctx.Array("d", n_workers, lock=False)
    waiting = ctx.Array("b", n_workers, lock=False)
    group = [w % len(devices) for w in range(n_workers)]
    shms = [shared_memory.SharedMemory(create=True, size=opts.slots * SLOT_BYTES) for _ in range(n_workers)]

    def fill_slots(w):
        for s in range(opts.slots):
            free_qs[w].put((gens[w], s))

    gpus = []
    try:
        for g, dev in enumerate(devices):
            mine = {w: (shms[w].name, opts.slots) for w in range(n_workers) if group[w] == g}
            p = ctx.Process(target=gpu_main, name=f"gpu{g}", args=(
                g, dev, opts, gin[g], {w: free_qs[w] for w in mine}, mine,
                str(parts_dir / f"gpu{g}.jsonl"), str(parts_dir / f"gpu{g}.done"), ctrl_q,
                str(parts_dir / f"gpu{g}.stats.json")))
            p.start()
            gpus.append(p)
        ready = 0
        while ready < len(gpus):
            msg = ctrl_q.get()
            if msg[0] == "ready":
                ready += 1
                log(f"gpu{msg[1]} ready on {msg[2]}")
            elif msg[0] == "gpu_error":
                log(f"gpu{msg[1]} failed:\n{msg[2]}")
                return 1

        def start_worker(w):
            beats[w] = time.time()
            waiting[w] = 0
            p = ctx.Process(target=worker_main, name=f"worker{w}", daemon=True, args=(
                w, task_q, gin[group[w]], free_qs[w], shms[w].name, opts.slots, gens, beats, waiting, opts))
            p.start()
            return p

        for w in range(n_workers):
            fill_slots(w)
        workers = [start_worker(w) for w in range(n_workers)]
        counters = {"videos": 0}
        feeder = threading.Thread(target=_feed, args=(files, task_q, done, opts, n_workers, counters), daemon=True)
        feeder.start()

        finished = [False] * n_workers
        while not all(finished):
            time.sleep(0.5)
            try:
                msg = ctrl_q.get_nowait()
                if msg[0] == "gpu_error":
                    log(f"gpu{msg[1]} failed:\n{msg[2]}")
                    return 1
            except queue.Empty:
                pass
            for w, p in enumerate(workers):
                if finished[w]:
                    continue
                hung = (not waiting[w]) and time.time() - beats[w] > opts.hang_seconds
                if p.is_alive() and not hung:
                    continue
                if not p.is_alive() and p.exitcode == 0:
                    finished[w] = True
                    continue
                if hung:
                    p.terminate()
                    p.join(5)
                log(f"worker{w} {'hung' if hung else f'died (exit {p.exitcode})'}; restarting it")
                old = gens[w]
                gens[w] = old + 1
                gin[group[w]].put(("dead", w, old))
                fill_slots(w)
                workers[w] = start_worker(w)
            for g, p in enumerate(gpus):
                if not p.is_alive():
                    log(f"gpu{g} exited unexpectedly (exit {p.exitcode})")
                    return 1
        for q in gin:
            q.put(("finish",))
        for p in gpus:
            p.join()
            if p.exitcode != 0:
                log(f"{p.name} exit code {p.exitcode}")
                return 1
    finally:
        for p in gpus + [w for w in locals().get("workers", []) if w is not None]:
            if p.is_alive():
                p.terminate()
        for s in shms:
            s.close()
            s.unlink()

    # merge the per-GPU parts into the output file
    n_lines = 0
    with open(out, "w") as fh:
        for f in sorted(parts_dir.glob("*.jsonl")):
            for line in f.read_text().splitlines():
                if line.strip():
                    fh.write(line + "\n")
                    n_lines += 1
    stats = {}
    for f in sorted(parts_dir.glob("*.stats.json")):
        stats[f.name.split(".")[0]] = json.loads(f.read_text())
    import shutil

    shutil.rmtree(parts_dir, ignore_errors=True)  # merged into the output; only needed for resuming
    wall = time.time() - t0
    summary = {"mode": opts.mode, "wall_s": round(wall, 1), "lines": n_lines, "workers": n_workers,
               "devices": devices, "gpu": stats, "options": {k: v for k, v in vars(opts).items()}}
    Path(str(out) + ".stats.json").write_text(json.dumps(summary, indent=2, default=str))
    log(f"done: {n_lines} lines in {wall:.1f}s -> {out}")
    return 0
