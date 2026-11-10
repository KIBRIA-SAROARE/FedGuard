#!/usr/bin/env python3
"""Idle baseline so monitoring CPU/RAM/power can be reported as an increment over idle."""
import argparse
import os
import time

from fgpt.resmon import ResourceMonitor

ap = argparse.ArgumentParser()
ap.add_argument("--results", default="results")
ap.add_argument("--seconds", type=float, default=30)
a = ap.parse_args()
os.makedirs(a.results, exist_ok=True)
m = ResourceMonitor(os.path.join(a.results, "resources_idle.csv"), os.path.join(a.results, "tegrastats_idle.log"), 0.5).start()
m.set_phase("idle")
print(f"[idle] recording {a.seconds:.0f} s; leave the board alone ...", flush=True)
time.sleep(a.seconds)
m.stop()
print("[idle] tegrastats available:", m.has_tegrastats)
