"""Resource monitors: server RSS/CPU (psutil), GPU (nvidia-smi polling).

Event-loop lag is NOT sampled here — it must be observed from inside the
server's own loop, which server_wrapper.py does (scaffolding, no server
source change). This module consumes the wrapper's pidfile to find the
server process.
"""

import json
import os
import subprocess
import threading
import time

import psutil

from .metrics import Series


class ResourceMonitor:
    """1 Hz sampler thread for server RSS/CPU and GPU memory/utilisation."""

    def __init__(self, server_pid=None, interval_s=1.0):
        self.interval_s = interval_s
        self.proc = psutil.Process(server_pid) if server_pid else None
        self.rss_mib = Series("rss_mib")
        self.cpu_pct = Series("cpu_pct")
        self.gpu_used_mib = Series("gpu_used_mib")
        self.gpu_util_pct = Series("gpu_util_pct")
        self._stop = threading.Event()
        self._t0 = time.monotonic()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self.nvidia_ok = True

    def _sample_gpu(self):
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5)
            used, util = out.stdout.strip().splitlines()[0].split(",")
            return float(used), float(util)
        except Exception:
            self.nvidia_ok = False
            return None, None

    def _run(self):
        while not self._stop.is_set():
            t = time.monotonic() - self._t0
            if self.proc is not None:
                try:
                    self.rss_mib.add(t, self.proc.memory_info().rss / (1 << 20))
                    self.cpu_pct.add(t, self.proc.cpu_percent(interval=None))
                except psutil.NoSuchProcess:
                    break
            used, util = self._sample_gpu()
            if used is not None:
                self.gpu_used_mib.add(t, used)
                self.gpu_util_pct.add(t, util)
            self._stop.wait(self.interval_s)

    def now(self):
        """Current time on this monitor's axis (seconds since start)."""
        return time.monotonic() - self._t0

    def start(self):
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=5)

    def snapshot(self):
        return {
            "rss_mib": self.rss_mib.bundle(),
            "cpu_pct": self.cpu_pct.bundle(),
            "gpu_used_mib": self.gpu_used_mib.bundle(),
            "gpu_util_pct": self.gpu_util_pct.bundle(),
            "nvidia_smi_available": self.nvidia_ok,
        }


def read_server_info(path):
    """Read the pidfile/loop-lag paths written by server_wrapper.py."""
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_loop_lag_csv(path):
    """Parse the wrapper's loop-lag CSV into a Series."""
    s = Series("event_loop_lag_ms")
    if not path or not os.path.exists(path):
        return s
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                t, lag = line.strip().split(",")
                s.add(float(t), float(lag))
            except ValueError:
                continue
    return s
