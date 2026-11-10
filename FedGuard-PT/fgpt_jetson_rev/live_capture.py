#!/usr/bin/env python3
"""Live monitor display at 100 Hz wall clock, with automatic snapshots for the paper.

Run AFTER run_full_experiment.py, as a separate step: drawing costs CPU, so timing numbers for the
paper come from rt_monitor.py streams, not from this display.

  python3 live_capture.py --fold S02_fault_100ms --test A1
  python3 live_capture.py --fold FULL --test ctrl --scenario S07_atk_false_ups
  MPLBACKEND=Agg python3 live_capture.py ...      # no screen: still paced, files only
Snapshots: results_rt/live/<fold>_<test>_<scenario>_t<sim-time>.{png,pdf} at --snap times
(default 'auto': 3 s after the grid event, 8 s after each attack onset (max 3), and the end).
"""
import argparse
import collections
import os
import time

import matplotlib
matplotlib.use(os.environ.get("MPLBACKEND", "TkAgg"))
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from fgpt import csec_rt, nb, stream  # noqa: E402
from rt_monitor import KEYS, RtScorer, build_test, load_bundle  # noqa: E402

EVENT_T = {"S02_fault_100ms": 60.0, "S03_fault_250ms": 60.0, "S06_multi_event": 60.0, "S11_fault_plus_atk": 60.0,
           "S04_line_trip": 90.0, "S05_load_step": 120.0}

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="data/dataset")
ap.add_argument("--out", default="results_rt")
ap.add_argument("--fold", default="S02_fault_100ms")
ap.add_argument("--test", default="A1")
ap.add_argument("--scenario", default=None, help="defaults to the fold's held-out scenario")
ap.add_argument("--dc", type=int, default=1)
ap.add_argument("--span", type=float, default=30.0)
ap.add_argument("--snap", default="auto", help="'auto' or comma-separated simulation times in s")
ap.add_argument("--font", type=float, default=15)
ap.add_argument("--speed", type=float, default=1.0, help="1 = real time (use 1 for paper snapshots)")
ap.add_argument("--no-hold", action="store_true", help="close the window at the end instead of waiting (batch runs)")
a = ap.parse_args()
scen = a.scenario or a.fold
nb.DATASET_DIR = a.dataset
CAL, p, fusion = load_bundle(os.path.join(a.out, "folds", a.fold), a.dc)
tel, lab, tels, wins = build_test(a.test, scen, a.dc, CAL, a.dataset)
corr, unc, avail, _, _ = csec_rt.csec_with_release(tels, CAL, a.dc)
T = nb.tvec()
N, dt = len(T), nb.CFG["DT"]
if a.snap == "auto":
    snaps = []
    if scen in EVENT_T:
        snaps.append(EVENT_T[scen] + 3.0)
    snaps += [w0 + 8.0 for w0, _ in wins[:3]]
    snaps.append(float(T[-1]))
else:
    snaps = [float(v) for v in a.snap.split(",")]
snaps = sorted(set(round(min(s, float(T[-1])), 2) for s in snaps))
live_dir = os.path.join(a.out, "live")
os.makedirs(live_dir, exist_ok=True)

F = a.font
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Nimbus Roman", "Liberation Serif", "DejaVu Serif"],
                     "mathtext.fontset": "stix", "font.size": F, "axes.labelsize": F + 1, "xtick.labelsize": F - 1,
                     "ytick.labelsize": F - 1, "legend.fontsize": F - 2, "axes.grid": True, "grid.alpha": 0.35,
                     "lines.linewidth": 1.8, "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False})
sf, sc = stream.StreamingFeatures(a.dc, CAL[a.dc]), RtScorer(p, fusion, corr, unc, avail, a.dc)
M = int(a.span / dt)
tt, vv, pp, ff, ll = (collections.deque(maxlen=M) for _ in range(5))
wt, ws, wa, pt, ps = (collections.deque(maxlen=M // nb.CFG["HOP"] + 40) for _ in range(5))
last, lat = None, collections.deque(maxlen=100)

plt.ion()
fig, ax = plt.subplots(4, 1, figsize=(7.16, 7.6), sharex=True, gridspec_kw=dict(height_ratios=[1, 1.2, 1.2, 0.7]))
try:
    fig.canvas.manager.set_window_title(f"FedGuard-PT live monitor DC{a.dc}")
except Exception:  # noqa: BLE001
    pass
lv, = ax[0].plot([], [], c="k"); ax[0].set_ylabel("$V_a$ (pu)")
lp, = ax[1].plot([], [], c="#1f77b4", label="$P_{total}$ reported")
lf, = ax[1].plot([], [], c="#ff7f0e", ls="--", label="$P_{it}+P_{cool}+P_{aux}$")
ax[1].set_ylabel("$P$ (MW)")
lps, = ax[2].step([], [], where="post", c="#9467bd", ls=":", label="provisional")
ls_, = ax[2].step([], [], where="post", c="#2ca02c", label="released (with CSEC)")
ax[2].axhline(fusion["threshold"], c="r", ls="--", lw=1.4, label="threshold")
ax[2].set_ylabel("score"); ax[2].set_yscale("symlog", linthresh=10.0)   # linear in [-10, 10]; physics rule adds +1000
la, = ax[3].step([], [], where="post", c="#1f77b4", label="alarm")
lb, = ax[3].step([], [], where="post", c="#d62728", alpha=0.6, label="attack")
ax[3].set_ylim(-0.15, 1.35); ax[3].set_yticks([0, 1]); ax[3].set_xlabel("time (s)")
banner = fig.suptitle("", fontsize=F - 1, y=0.998)
fig.legend(handles=[lp, lf, lps, ls_, ax[2].lines[-1], la, lb], ncol=3, loc="upper center", bbox_to_anchor=(0.5, 0.965),
           frameon=False, fontsize=F - 3, handlelength=1.8, columnspacing=1.0)
fig.align_ylabels(ax)
fig.tight_layout(rect=(0, 0, 1, 0.875))

si, i, base = 0, 0, time.perf_counter() + 0.2
while i < N:
    due = min(N, int((time.perf_counter() - base) * a.speed / dt) + 1)
    while i < due:
        s0 = time.perf_counter()
        x = sf.step(*(float(tel[c][i]) for c in KEYS))
        out, ran = sc.push(x)
        if ran:
            w = sc.pending[-1] if sc.pending and sc.pending[-1]["target"] == i else None
            if w is None:
                w = next((o for o in out if o["target"] == i), None)
            if w is not None:
                pt.append(T[i]); ps.append(w["prov_score"])
        for o in out:
            wt.append(T[i]); ws.append(o["score"]); wa.append(float(o["alarm"])); last = o
            lat.append((time.perf_counter() - s0) * 1e3)
        tt.append(T[i]); vv.append(tel["Va_pu"][i]); pp.append(tel["P_total_MW"][i])
        ff.append(tel["P_it_MW"][i] + tel["P_cool_MW"][i] + tel["P_aux_MW"][i]); ll.append(float(lab[i] == 1))
        i += 1
        if si < len(snaps) and T[i - 1] >= snaps[si]:
            break
    if not tt:
        time.sleep(0.02)
        continue
    for line, xs, ys in ((lv, tt, vv), (lp, tt, pp), (lf, tt, ff), (lb, tt, ll), (ls_, wt, ws), (la, wt, wa), (lps, pt, ps)):
        line.set_data(xs, ys)
    for k in (0, 1, 2):
        ax[k].relim(); ax[k].autoscale_view(scalex=False)
    ax[3].set_xlim(max(float(tt[0]), float(tt[-1]) - a.span), float(tt[-1]) + 0.3)
    if last is not None:
        st = "ALARM" if last["alarm"] else "normal"
        banner.set_text(f"DC{a.dc} | {scen} | {a.test} | t = {tt[-1]:.2f} s | {st}"
                        + (f" | term {last['dominant']}" if last["alarm"] else ""))
        banner.set_color("#b00000" if last["alarm"] else "black")
    if os.environ.get("MPLBACKEND", "TkAgg") != "Agg":
        plt.pause(0.001)
    if si < len(snaps) and T[i - 1] >= snaps[si]:
        name = f"{a.fold}_{a.test}_{scen}_t{snaps[si]:06.2f}".replace(".", "p")
        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(live_dir, f"{name}.{ext}"), dpi=300 if ext == "png" else None, bbox_inches="tight")
        print("saved live snapshot", name, flush=True)
        si += 1
    if i < N:
        time.sleep(0.05)
plt.ioff()
if os.environ.get("MPLBACKEND", "TkAgg") != "Agg" and not a.no_hold:
    plt.show()          # interactive use: keep the final frame open until the window is closed
plt.close("all")
