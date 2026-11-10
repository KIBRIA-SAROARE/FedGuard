#!/usr/bin/env python3
"""Offline analysis: contribution of the physics rule on the UPS check.

The detector now applies this rule itself (nb.physics_rule, fusion.json "physics_rule"). This script separates the
two parts from the windows.csv of each stream: "fused" = learned score without the rule (score minus the boost),
"with_rule" = the deployed alarm.

Rule: alarm if the window's |r_ups| term is above RULE_Z (the term is in units of UPS_SCALE = 0.05 pu, exactly 0 when
clean), OR the fused alarm. Computed from the windows.csv written by rt_monitor.py, on the same windows and labels.
  python3 ups_rule_analysis.py --root results_rev [--rule-z 1.0]
"""
import argparse
import csv
import glob
import json
import os

import numpy as np
import sys as _sys
_sys_path = __import__("os").path.dirname(__import__("os").path.abspath(__file__))
__import__("sys").path.insert(0, _sys_path)
from fgpt import nb  # noqa: E402  (REV: warm-up setting)


def metrics(alarm, y):
    tp = int((alarm & (y == 1)).sum()); fp = int((alarm & (y == 0)).sum())
    fn = int((~alarm & (y == 1)).sum()); neg = int((y == 0).sum()); pos = int((y == 1).sum())
    rec = tp / pos if pos else float("nan"); prec = tp / (tp + fp) if tp + fp else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if pos and tp else (0.0 if pos else float("nan"))
    return dict(recall=rec, precision=prec, f1=f1, fpr=fp / neg if neg else float("nan"), pos=pos, neg=neg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results_rev")
    ap.add_argument("--rule-z", type=float, default=1.0)
    a = ap.parse_args()
    rows = []
    pats = [os.path.join(a.root, "streams", "*", "*", "windows.csv"),
            os.path.join(a.root, "streams_attacked", "*", "*", "windows.csv"),
            os.path.join(a.root, "streams_sites", "*", "*", "*", "windows.csv")]
    for p in sorted(sum((glob.glob(x) for x in pats), [])):
        m = json.load(open(os.path.join(os.path.dirname(p), "metrics.json")))
        with open(p) as f:
            R = list(csv.DictReader(f))
        y = np.array([int(r["label"]) for r in R])
        v = nb.eval_mask(y, np.array([float(r["t_end"]) for r in R]) + 1e-6)     # REV: warm-up excluded
        zu = np.array([float(r["|r_ups|"]) for r in R])
        sc = np.array([float(r["score"]) for r in R])
        fired = np.array([r.get("physics_rule", "0") == "1" for r in R])
        al = (sc - 1000.0 * fired) > m["threshold"]           # learned fusion alone
        rule = al | (zu > a.rule_z)
        rows.append(dict(dc=m["dc"], scenario=m["scenario"], test=m["test"], fold=os.path.basename(os.path.dirname(os.path.dirname(p))),
                         fused=metrics(al[v], y[v]), with_rule=metrics(rule[v], y[v]),
                         rule_fired_clean=int(((zu > a.rule_z) & v & (y == 0)).sum())))
    out = os.path.join(a.root, "revision", "ups_rule_analysis.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(dict(rule_z=a.rule_z, streams=rows), open(out, "w"), indent=1, default=float)

    def agg(sel, key):
        S = [r for r in rows if sel(r)]
        if not S:
            return None
        f = lambda k, w: np.nanmean([r[w][k] for r in S])  # noqa: E731
        return dict(n=len(S), recall=(f("recall", "fused"), f("recall", "with_rule")), fpr=(f("fpr", "fused"), f("fpr", "with_rule")),
                    rule_clean_windows=sum(r["rule_fired_clean"] for r in S))
    groups = {"DC1 U1 (5 folds)": lambda r: r["dc"] == 1 and r["test"] == "U1",
              "all sites U1 (30)": lambda r: r["test"] == "U1",
              "DC1 no_attack": lambda r: r["dc"] == 1 and r["test"] == "no_attack",
              "all sites no_attack": lambda r: r["test"] == "no_attack",
              "all sites other LOSO tests": lambda r: r["test"] not in ("U1", "no_attack", "ctrl", "ctrl_hide"),
              "S07 ctrl DC1/DC4": lambda r: r["scenario"] == "S07_atk_false_ups" and r["test"] == "ctrl" and r["dc"] in (1, 4),
              "S09 ctrl DC3": lambda r: r["scenario"] == "S09_atk_sensor_spoof" and r["test"] == "ctrl" and r["dc"] == 3,
              "U2 ctrl_hide": lambda r: r["test"] == "ctrl_hide"}
    print(f"rule: |r_ups| term > {a.rule_z}  (recall fused -> with rule | FPR fused -> with rule | clean windows the rule fired)")
    for name, sel in groups.items():
        g = agg(sel, None)
        if g:
            print(f"{name:28s} n={g['n']:3d}  recall {g['recall'][0]:.3f} -> {g['recall'][1]:.3f}   "
                  f"FPR {g['fpr'][0]:.4f} -> {g['fpr'][1]:.4f}   rule on clean: {g['rule_clean_windows']}")
    print("->", out)


if __name__ == "__main__":
    main()
