"""
Lightweight profiler shared by every pipeline stage.

A stage opens one Profiler, which:
  - samples VRAM / RAM / GPU utilisation on a background thread -> profile/samples.csv
  - records one row per timed span (epoch, case, tile batch)    -> profile/events.jsonl
  - stores the resolved config and machine environment          -> profile/config.json
  - writes per-span aggregates when the stage ends              -> profile/summary.json

Every stage of a run appends to the same profile/ directory, tagged by stage name,
so one run produces one comparable timeline end to end.

VRAM is reported three ways, because they answer different questions:
  torch_alloc    tensors currently held by torch
  torch_reserved what torch's caching allocator took from the driver
  proc_vram      what the driver charges this PID, including the CUDA context
                 -- this is the number that has to fit in the SLURM --gres request
"""

import csv
import json
import os
import platform
import socket
import subprocess
import threading
import time
from collections import defaultdict
from contextlib import contextmanager

import numpy as np
import torch

try:
    import psutil
except ImportError:
    psutil = None

MB = 1024 ** 2
SAMPLE_INTERVAL = 2.0


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def _cuda_active() -> bool:
    """
    True only once this process actually uses the GPU.

    Deliberately not just is_available(): querying torch's memory stats forces a
    CUDA context, which would reserve VRAM on a shared device for CPU-only stages
    like make_shells.
    """
    return torch.cuda.is_available() and torch.cuda.is_initialized()


def model_summary(model: torch.nn.Module) -> dict:
    params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "params": params,
        "params_trainable": trainable,
        "params_mb": round(params * 4 / MB, 2),
    }


class _GpuProbe:
    """Per-process GPU telemetry. Degrades to torch-only, then to nothing."""

    def __init__(self):
        self.pid     = os.getpid()
        self.nvml    = None
        self.handle  = None
        self._tried  = False

    def _ensure_handle(self):
        """
        Attach to NVML lazily, so a CPU-only stage never initialises CUDA.

        Resolves the device by UUID, not by torch.cuda.current_device(). SLURM's
        gres/gpu plugin sets CUDA_VISIBLE_DEVICES to remap the assigned GPU to
        index 0 from CUDA's point of view, but NVML is not masked by that env var
        and still enumerates every physical GPU on the node — indexing it with
        torch's (masked) index silently queries the wrong physical device on any
        node with more than one GPU, which made every per-process reading inside
        a batch job come back empty. The UUID is stable across that remapping.
        """
        if self._tried or not _cuda_active():
            return
        self._tried = True
        try:
            import pynvml

            pynvml.nvmlInit()
            self.nvml = pynvml

            uuid = torch.cuda.get_device_properties(torch.cuda.current_device()).uuid
            self.handle = pynvml.nvmlDeviceGetHandleByUUID(f"GPU-{uuid}".encode())
        except Exception:
            self.nvml = None

    def proc_vram_mb(self):
        self._ensure_handle()
        if not self.handle:
            return None
        try:
            procs = self.nvml.nvmlDeviceGetComputeRunningProcesses(self.handle)
        except Exception:
            return None
        for p in procs:
            if p.pid == self.pid and p.usedGpuMemory is not None:
                return round(p.usedGpuMemory / MB, 1)
        return None

    def util_pct(self):
        self._ensure_handle()
        if not self.handle:
            return None
        try:
            return self.nvml.nvmlDeviceGetUtilizationRates(self.handle).gpu
        except Exception:
            return None

    def sample(self) -> dict:
        if not _cuda_active():
            return {}
        return {
            "torch_alloc_mb":    round(torch.cuda.memory_allocated() / MB, 1),
            "torch_reserved_mb": round(torch.cuda.memory_reserved() / MB, 1),
            "proc_vram_mb":      self.proc_vram_mb(),
            "gpu_util_pct":      self.util_pct(),
        }


SAMPLE_FIELDS = [
    "wall_time", "elapsed_s", "stage", "span",
    "torch_alloc_mb", "torch_reserved_mb", "proc_vram_mb", "gpu_util_pct",
    "host_rss_mb", "host_ram_used_pct", "cpu_pct",
]


class Profiler:

    def __init__(self, stage: str, config: dict = None, profile_dir: str = None,
                 sample_interval: float = SAMPLE_INTERVAL):
        self.stage    = stage
        self.dir      = profile_dir
        self.interval = sample_interval
        self.gpu      = _GpuProbe()

        self.config = {
            "stage":       stage,
            "args":        _jsonable(config or {}),
            "environment": self._environment(),
            "started":     time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        self._spans      = defaultdict(list)
        self._span_stack = []
        self._peak_alloc = 0.0
        self._peak_proc  = 0.0
        self._t0         = None
        self._stop       = threading.Event()
        self._thread     = None
        self._csv_file   = None
        self._csv        = None

        self._proc = psutil.Process() if psutil else None
        if self._proc:
            self._proc.cpu_percent()  # prime; first call always returns 0.0

        if self.dir:
            os.makedirs(self.dir, exist_ok=True)

    # ---------------------------------------------------------------- lifecycle

    def __enter__(self):
        self._t0 = time.perf_counter()

        if _cuda_active():
            torch.cuda.reset_peak_memory_stats()

        if self.dir:
            self._merge_json("config.json", {self.stage: self.config})

            path      = os.path.join(self.dir, "samples.csv")
            is_new    = not os.path.exists(path) or os.path.getsize(path) == 0
            self._csv_file = open(path, "a", newline="")
            self._csv      = csv.DictWriter(self._csv_file, fieldnames=SAMPLE_FIELDS)
            if is_new:
                self._csv.writeheader()

        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=self.interval + 1.0)
        if self._csv_file:
            self._csv_file.close()
        self.write_summary()
        return False

    # ------------------------------------------------------------------ capture

    @contextmanager
    def span(self, name: str, **meta):
        """
        Time a block. Peak VRAM is measured from the start of this span.

        Yields the metadata dict, so a caller can attach counts it only learns
        while the block runs:  with prof.span("epoch") as s: ...; s["items"] = n
        """
        if _cuda_active():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

        self._span_stack.append(name)
        t0 = time.perf_counter()
        try:
            yield meta
        finally:
            if _cuda_active():
                torch.cuda.synchronize()
            duration = time.perf_counter() - t0
            self._span_stack.pop()

            record = {"span": name, "duration_s": round(duration, 4), **meta}

            if _cuda_active():
                peak_alloc    = torch.cuda.max_memory_allocated() / MB
                peak_reserved = torch.cuda.max_memory_reserved() / MB
                proc          = self.gpu.proc_vram_mb()

                record["peak_alloc_mb"]    = round(peak_alloc, 1)
                record["peak_reserved_mb"] = round(peak_reserved, 1)
                record["proc_vram_mb"]     = proc

                self._peak_alloc = max(self._peak_alloc, peak_alloc)
                if proc:
                    self._peak_proc = max(self._peak_proc, proc)

            if self._proc:
                record["host_rss_mb"] = round(self._proc.memory_info().rss / MB, 1)

            self._spans[name].append(record)
            self.event("span", **record)

    def event(self, kind: str, **fields):
        if not self.dir:
            return
        row = {
            "t":     round(time.perf_counter() - self._t0, 3) if self._t0 else 0.0,
            "stage": self.stage,
            "kind":  kind,
            **_jsonable(fields),
        }
        with open(os.path.join(self.dir, "events.jsonl"), "a") as f:
            f.write(json.dumps(row) + "\n")

    def config_update(self, extra: dict):
        self.config.update(_jsonable(extra))
        if self.dir:
            self._merge_json("config.json", {self.stage: self.config})

    # ------------------------------------------------------------------ sampling

    def _sample_loop(self):
        while not self._stop.is_set():
            try:
                self._write_sample()
            except Exception:
                pass  # telemetry must never take the training run down
            self._stop.wait(self.interval)

    def _write_sample(self):
        if not self._csv:
            return

        row = {
            "wall_time": round(time.time(), 2),
            "elapsed_s": round(time.perf_counter() - self._t0, 2),
            "stage":     self.stage,
            "span":      self._span_stack[-1] if self._span_stack else "",
            **self.gpu.sample(),
        }

        if self._proc:
            row["host_rss_mb"]       = round(self._proc.memory_info().rss / MB, 1)
            row["host_ram_used_pct"] = psutil.virtual_memory().percent
            row["cpu_pct"]           = self._proc.cpu_percent()

        self._csv.writerow({k: row.get(k) for k in SAMPLE_FIELDS})
        self._csv_file.flush()

    # ------------------------------------------------------------------- summary

    def summary(self) -> dict:
        spans = {}
        for name, records in self._spans.items():
            durations = np.array([r["duration_s"] for r in records], dtype=float)
            entry = {
                "count":      len(records),
                "total_s":    round(float(durations.sum()), 3),
                "mean_s":     round(float(durations.mean()), 4),
                "median_s":   round(float(np.median(durations)), 4),
                "p95_s":      round(float(np.percentile(durations, 95)), 4),
                "min_s":      round(float(durations.min()), 4),
                "max_s":      round(float(durations.max()), 4),
            }

            peaks = [r["peak_alloc_mb"] for r in records if r.get("peak_alloc_mb")]
            if peaks:
                entry["peak_alloc_mb"] = round(max(peaks), 1)

            items = sum(r.get("items", 0) or 0 for r in records)
            if items:
                entry["items"]      = items
                entry["items_per_s"] = round(items / max(durations.sum(), 1e-9), 2)

            waits = [r["data_wait_s"] for r in records if r.get("data_wait_s") is not None]
            if waits:
                total_wait = float(sum(waits))
                entry["data_wait_s"]   = round(total_wait, 3)
                entry["data_wait_pct"] = round(100 * total_wait / max(float(durations.sum()), 1e-9), 1)

            spans[name] = entry

        return {
            "stage":             self.stage,
            "wall_s":            round(time.perf_counter() - self._t0, 2) if self._t0 else 0.0,
            "peak_alloc_mb":     round(self._peak_alloc, 1) if self._peak_alloc else None,
            "peak_proc_vram_mb": round(self._peak_proc, 1) if self._peak_proc else None,
            "spans":             spans,
        }

    def write_summary(self):
        data = self.summary()
        if self.dir:
            self._merge_json("summary.json", {self.stage: data})
        self.print_summary(data)

    def print_summary(self, data: dict = None):
        data = data or self.summary()

        print(f"\n{'-' * 72}")
        print(f"  profile — {self.stage}   wall {data['wall_s']:.1f}s")

        vram = []
        if data["peak_proc_vram_mb"]:
            vram.append(f"{data['peak_proc_vram_mb']:.0f} MB process")
        if data["peak_alloc_mb"]:
            vram.append(f"{data['peak_alloc_mb']:.0f} MB torch tensors")
        if vram:
            print("  peak VRAM: " + "  /  ".join(vram))

        if data["spans"]:
            print(f"  {'span':<20} {'n':>5} {'mean':>9} {'p95':>9} {'total':>9} {'peak MB':>9}")
            for name, s in data["spans"].items():
                peak = f"{s['peak_alloc_mb']:.0f}" if "peak_alloc_mb" in s else "-"
                print(f"  {name:<20} {s['count']:>5} {s['mean_s']:>8.3f}s"
                      f" {s['p95_s']:>8.3f}s {s['total_s']:>8.1f}s {peak:>9}")
                if "items_per_s" in s:
                    print(f"    {'':<18} throughput {s['items_per_s']:.2f} items/s")
                if "data_wait_pct" in s:
                    print(f"    {'':<18} data wait  {s['data_wait_s']:.1f}s"
                          f" ({s['data_wait_pct']:.0f}% of span)")
        print(f"{'-' * 72}\n", flush=True)

    # -------------------------------------------------------------------- helpers

    def _merge_json(self, name: str, payload: dict):
        path = os.path.join(self.dir, name)
        current = {}
        if os.path.exists(path):
            try:
                with open(path) as f:
                    current = json.load(f)
            except (json.JSONDecodeError, OSError):
                current = {}
        current.update(payload)
        with open(path, "w") as f:
            json.dump(current, f, indent=2)

    @staticmethod
    def _environment() -> dict:
        env = {
            "hostname":     socket.gethostname(),
            "python":       platform.python_version(),
            "torch":        torch.__version__,
            "cuda":         torch.version.cuda,
            "git_commit":   _git_commit(),
            "cpu_count":    os.cpu_count(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        }

        # _cuda_active, not is_available: get_device_properties forces a lazy init,
        # which would cost a ~400 MB CUDA context on a CPU-only stage.
        if _cuda_active():
            props = torch.cuda.get_device_properties(torch.cuda.current_device())
            env["gpu_name"]     = props.name
            env["gpu_total_mb"] = round(props.total_memory / MB)

        if psutil:
            env["host_ram_total_mb"] = round(psutil.virtual_memory().total / MB)

        return env


def _jsonable(obj):
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)
