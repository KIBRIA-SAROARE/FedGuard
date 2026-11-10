"""Shared helpers: config, CSV loading, JSON I/O."""
import json
import os
import platform
import subprocess
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_config(path=None):
    path = path or os.path.join(ROOT, "config.json")
    with open(path) as f:
        return json.load(f)


def load_csv(path, t_min=2.0):
    """Load a 17/19-column dataset CSV into a dict of float arrays. Drops t < t_min (model soft-start)."""
    with open(path) as f:
        header = f.readline().strip().split(",")
    arr = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float64)
    d = {name: arr[:, i] for i, name in enumerate(header)}
    keep = d["t"] >= t_min - 1e-9
    d = {k: v[keep] for k, v in d.items()}
    if "label" not in d:
        d["label"] = np.zeros_like(d["t"])
        d["attack_type"] = np.zeros_like(d["t"])
    return d


def csv_path(cfg, kind, scenario, dc, atk=None):
    base = cfg["dataset_dir"]
    if kind == "clean":
        return os.path.join(base, "clean", f"{scenario}_DC{dc}.csv")
    return os.path.join(base, "attacked", f"{scenario}_DC{dc}_A{atk}.csv")


def attack_windows(cfg, scenario, dc):
    """Rows of attack_log.csv for <scenario>_DC<dc>: list of (type, t0, t1)."""
    out = []
    path = os.path.join(cfg["dataset_dir"], "attack_log.csv")
    key = f"{scenario}_DC{dc}"
    with open(path) as f:
        f.readline()
        for line in f:
            p = line.strip().split(",")
            if p[0] == key:
                out.append((int(p[1]), float(p[2]), float(p[3])))
    return out


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=float)


def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception as e:  # noqa: BLE001
        return f"n/a ({e})"


def system_info():
    import psutil
    info = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hostname": platform.node(),
        "machine": platform.machine(),
        "kernel": platform.release(),
        "os": sh("grep PRETTY_NAME /etc/os-release | cut -d= -f2"),
        "device_model": sh("tr -d '\\0' < /proc/device-tree/model"),
        "l4t_release": sh("head -1 /etc/nv_tegra_release"),
        "nvpmodel": sh("nvpmodel -q 2>/dev/null"),
        "cpu_count_logical": psutil.cpu_count(),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "ram_total_MiB": psutil.virtual_memory().total / 2**20,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas_scipy_sklearn": sh("python3 -c 'import pandas,scipy,sklearn;print(pandas.__version__,scipy.__version__,sklearn.__version__)'"),
        "blas_threads_env": {k: os.environ.get(k) for k in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"]},
    }
    return info
