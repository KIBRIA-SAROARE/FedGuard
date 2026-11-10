#!/usr/bin/env python3
"""REV: recompute the detection metrics of finished streams with the warm-up exclusion (no re-run needed).

For every <stream>/metrics.json that has a windows.csv next to it, the detection metrics are recomputed from the
released windows, keeping only labelled windows with t_end >= WARMUP_S (default 10 s, nb.CFG["WARMUP_S"]).
Timing, resource, CSEC-traffic and stream-batch parity fields are kept as they are (they describe the whole run).
The original file is kept once as metrics_before_recompute.json; re-running is safe (always starts from that copy).

  python3 recompute_metrics.py --root results_rev                # all streams under the root
  python3 recompute_metrics.py --root results_rev --warmup-s 0   # check: must reproduce the original files
"""
import argparse, glob, json, os, shutil, sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fgpt import nb


def _prf(alarm, y):
    """F1/precision/recall/FPR from the stored alarm flags (exact; the stored scores are rounded)."""
    from sklearn.metrics import f1_score, precision_score, recall_score
    out = dict(f1=float(f1_score(y, alarm, zero_division=0)), precision=float(precision_score(y, alarm, zero_division=0)),
               recall=float(recall_score(y, alarm, zero_division=0)))
    neg = y == 0
    out["fpr"] = float(alarm[neg].mean()) if neg.any() else float("nan")
    out["n_pos"], out["n_neg"] = int((y == 1).sum()), int(neg.sum())
    return out


def _metrics(score, alarm, y, thr):
    m = nb.metrics_from(score, y, thr)     # ROC-AUC, PR-AUC
    m.update(_prf(alarm.astype(int), y))
    return m


def recompute(mdir, warmup):
    mp, wp = os.path.join(mdir, "metrics.json"), os.path.join(mdir, "windows.csv")
    orig = os.path.join(mdir, "metrics_before_recompute.json")
    if not os.path.exists(orig):
        shutil.copy2(mp, orig)
    met = json.load(open(orig))
    w = pd.read_csv(wp)
    lab, ts, te = w["label"].to_numpy(), w["t_start"].to_numpy(), w["t_end"].to_numpy()
    score, alarm = w["score"].to_numpy(float), w["alarm"].to_numpy().astype(bool)
    thr = met["threshold"]
    # t_end is stored with 2 decimals; the warm-up boundary is compared with a 1e-6 tolerance
    valid = (lab >= 0) & (te >= warmup - 1e-6)
    y = (lab[valid] == 1).astype(int)

    met.update(_metrics(score[valid], alarm[valid], y, thr))
    met["windows_valid"], met["windows_pos"], met["warmup_s"] = int(valid.sum()), int(y.sum()), float(warmup)

    rf = w["physics_rule"].to_numpy().astype(bool)
    met["physics_rule"].update(fired_valid_neg=int((rf & valid & (lab == 0)).sum()), fired_pos=int((rf & valid & (lab == 1)).sum()))
    cr = met.get("clipped_release", {})
    if cr.get("windows", 0) == 0:            # no clipped window in the run: excluding them changes nothing
        met["excluding_clipped"] = _metrics(score[valid], alarm[valid], y, thr)
        cr["windows_valid"] = cr["windows_pos"] = 0
    else:                                     # the clipped flags are not stored per window; keep and mark the old value
        met["excluding_clipped"]["note"] = "not recomputed (clipped windows present); includes warm-up windows"
    met["delay_s_notebook"] = nb.detection_delay(score[valid], y, thr, te[valid])
    pa = w["prov_alarm"].to_numpy().astype(bool)
    met["provisional_no_csec"] = _metrics(w["prov_score"].to_numpy(float)[valid], pa[valid], y, thr)
    dw = met.get("disturbance_window_10s")
    if dw:
        t0 = dw["event_time_s"]
        fw = valid & (te >= t0) & (ts <= t0 + 10.0) & (lab == 0)
        dw.update(neg_windows=int(fw.sum()), false_alarms=int(alarm[fw].sum()))
    after = te >= warmup - 1e-6
    a_ = alarm & after
    if a_.any():
        d = w["dominant"].to_numpy()[a_]
        met["dominant_term_share_in_alarms"] = {t: float((d == t).mean()) for t in nb.TERMS}
    else:
        met.pop("dominant_term_share_in_alarms", None)
    # attack_events are unaffected: every attack starts at t >= 20 s (asserted)
    for e in met.get("attack_events", []):
        assert e["t0"] >= warmup, f"attack event at {e['t0']} s inside the warm-up ({mdir})"
    with open(mp, "w") as f:
        json.dump(met, f, indent=2)
    return met


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results_rev")
    ap.add_argument("--warmup-s", type=float, default=nb.CFG["WARMUP_S"])
    a = ap.parse_args()
    dirs = sorted(os.path.dirname(p) for p in glob.glob(os.path.join(a.root, "**", "metrics.json"), recursive=True)
                  if os.path.exists(os.path.join(os.path.dirname(p), "windows.csv")))
    for d in dirs:
        recompute(d, a.warmup_s)
    print(f"recomputed {len(dirs)} streams under {a.root} with warm-up {a.warmup_s:g} s "
          f"(originals kept as metrics_before_recompute.json)")


if __name__ == "__main__":
    main()
