"""Background resource logger: process CPU/RSS (psutil) + Jetson tegrastats (GPU load, rail power, temps).

Nothing is estimated. If tegrastats is not present, the GPU/power columns stay empty and
raw_source says so.
"""
import csv
import os
import re
import shutil
import subprocess
import threading
import time

import psutil

RX = {
    "ram_used_MiB": re.compile(r"RAM (\d+)/(\d+)MB"),
    "gpu_load_pct": re.compile(r"GR3D_FREQ (\d+)%"),
    "vdd_in_mW": re.compile(r"VDD_IN (\d+)mW"),
    "vdd_cpu_gpu_cv_mW": re.compile(r"VDD_CPU_GPU_CV (\d+)mW"),
    "vdd_soc_mW": re.compile(r"VDD_SOC (\d+)mW"),
    "cpu_temp_C": re.compile(r"cpu@([\d.]+)C"),
    "tj_temp_C": re.compile(r"tj@([\d.]+)C"),
}


class ResourceMonitor:
    def __init__(self, out_csv, raw_log, interval_s=0.5):
        self.out_csv, self.raw_log, self.dt = out_csv, raw_log, interval_s
        self.phase = "init"
        self.proc = psutil.Process(os.getpid())
        self._stop = threading.Event()
        self._last_teg = {}
        self._teg = None
        self.has_tegrastats = shutil.which("tegrastats") is not None
        self.t0 = time.time()

    def set_phase(self, name):
        self.phase = name

    def _teg_reader(self):
        with open(self.raw_log, "w") as raw:
            for line in self._teg.stdout:
                raw.write(f"{time.time() - self.t0:.3f} {line}")
                vals = {}
                for k, rx in RX.items():
                    m = rx.search(line)
                    if m:
                        vals[k] = float(m.group(1))
                self._last_teg = vals
                if self._stop.is_set():
                    break

    def _loop(self):
        fields = ["t_s", "phase", "proc_cpu_pct", "proc_rss_MiB", "sys_cpu_pct", "sys_mem_used_MiB"] + list(RX)
        self.proc.cpu_percent(None)
        psutil.cpu_percent(None)
        with open(self.out_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(fields)
            while not self._stop.is_set():
                time.sleep(self.dt)
                row = [f"{time.time() - self.t0:.3f}", self.phase, self.proc.cpu_percent(None),
                       self.proc.memory_info().rss / 2**20, psutil.cpu_percent(None),
                       psutil.virtual_memory().used / 2**20]
                row += [self._last_teg.get(k, "") for k in RX]
                w.writerow(row)
                f.flush()

    def start(self):
        if self.has_tegrastats:
            ms = int(self.dt * 1000)
            self._teg = subprocess.Popen(["tegrastats", "--interval", str(ms)], stdout=subprocess.PIPE,
                                         stderr=subprocess.DEVNULL, text=True)
            threading.Thread(target=self._teg_reader, daemon=True).start()
        else:
            with open(self.raw_log, "w") as raw:
                raw.write("tegrastats not found on PATH; GPU/power columns are empty.\n")
        self._th = threading.Thread(target=self._loop, daemon=True)
        self._th.start()
        return self

    def stop(self):
        self._stop.set()
        self._th.join(timeout=2)
        if self._teg:
            self._teg.terminate()
