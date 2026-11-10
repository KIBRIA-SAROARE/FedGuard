"""Revision checks that need no trained model. Run: python3 tests/check_generators.py [dataset]"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fgpt import nb  # noqa: E402

nb.DATASET_DIR = sys.argv[1] if len(sys.argv) > 1 else "data/dataset"
CAL = {k: nb.calibrate_client(k) for k in range(1, 7)}
T = nb.tvec()
FLOAT = [c for c in nb.TEL_KEYS if c != "ups_state"]
fail = 0


def check(name, ok, info=""):
    global fail
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {info}")
    fail += not ok


for scen in ["S01_normal", "S02_fault_100ms", "S03_fault_250ms", "S05_load_step", "S06_multi_event"]:
    for k in range(1, 7):
        clean = nb.telemetry(scen, k)
        Rc = nb.raw_residuals(clean, k)
        for reg in ("T3a", "T3"):
            tel, lab, log = nb.make_attacked(scen, k, reg, nb.notebook_attack_seed(nb.CFG["ATK_SEED_TEST"], scen, k), CAL)
            pre = T < min(e["t0"] for e in log)
            dev = max(float(np.abs(tel[c] - clean[c])[pre].max()) for c in FLOAT)
            R = nb.raw_residuals(tel, k)
            att = lab == nb.LBL_ATTACK
            inv = max(float(np.abs(R[c] - Rc[c])[att].max()) for c in ("r_bal", "r_aux", "r_qp"))
            dP = float(np.abs(tel["P_total_MW"] - clean["P_total_MW"])[att].max())
            if k in (1, 5) or dev > 1e-9 or inv > 1e-9:
                check(f"{reg} {scen} DC{k}: clean before first attack; invariants kept in attack",
                      dev < 1e-9 and inv < 1e-9 and dP > 1.0, f"max|dev|={dev:.1e} inv={inv:.1e} attack|dP|={dP:.1f} MW")
        tel, lab, log = nb.make_attacked(scen, k, "U1", nb.notebook_attack_seed(nb.CFG["ATK_SEED_TEST"], scen, k), CAL)
        att = lab == nb.LBL_ATTACK
        other = max(float(np.abs(tel[c] - clean[c]).max()) for c in FLOAT)
        R = nb.raw_residuals(tel, k)
        ok = other == 0 and np.all(tel["ups_state"][att] != 0) and np.array_equal(tel["ups_state"][~att], clean["ups_state"][~att])
        if k == 1 or not ok:
            check(f"U1 {scen} DC{k}: only m changes, m!=0 inside", ok,
                  f"r_ups clean max={np.abs(Rc['r_ups'][att]).max():.3f} -> attacked mean={R['r_ups'][att].mean():.3f} (pu of Fit*Pdc)")

for scen, k in [("S07_atk_false_ups", 1), ("S07_atk_false_ups", 4), ("S09_atk_sensor_spoof", 3)]:
    clean = nb.telemetry(scen, k)
    tel = nb.hide_transfer(nb.telemetry(scen, k), scen, k)
    lab = nb.control_labels(scen, k) == nb.LBL_ATTACK
    Rc, R = nb.raw_residuals(clean, k), nb.raw_residuals(tel, k)
    check(f"U2 {scen} DC{k}: true state non-zero, reported 0", (clean["ups_state"][lab] != 0).mean() > 0.95 and np.all(tel["ups_state"][lab] == 0),
          f"true m!=0 share={(clean['ups_state'][lab] != 0).mean():.3f}; |r_ups| true-state max={np.abs(Rc['r_ups'][lab]).max():.2f}, hidden max={np.abs(R['r_ups'][lab]).max():.2f} (pu)")

# ---- REV: UPS check
if not nb.LEGACY_UPS:
    worst = 0.0
    for scen in nb.CFG["ATTACK_FREE"] + ["S08_atk_load_alter", "S10_atk_cooling", "S11_fault_plus_atk"]:
        for k in range(1, 7):
            worst = max(worst, float(np.abs(nb.ups_check(nb.telemetry(scen, k), k)).max()))
    check("UPS check exactly 0 on every record without a UPS-state attack (S01-S06, S08, S10, S11; 6 sites)",
          worst == 0.0, f"max={worst}")
    for scen, k in [("S07_atk_false_ups", 1), ("S07_atk_false_ups", 4), ("S09_atk_sensor_spoof", 3)]:
        r, a_, x_ = nb.ups_check(nb.telemetry(scen, k), k, return_parts=True)
        L = nb.control_labels(scen, k)
        lab, cln = L == nb.LBL_ATTACK, L == nb.LBL_CLEAN      # recovery tail (-1) is excluded from all metrics
        check(f"UPS check flags the {scen} transfer at DC{k} over the whole attack interval, 0 on clean samples",
              x_[lab & (nb.telemetry(scen, k)["ups_state"] != 0)].min() == 1.0 and np.abs(r[cln]).max() == 0.0,
              f"flagged share of attack samples={x_[lab].mean():.4f} (first reported transfer one sample after the label), clean-sample max={np.abs(r[cln]).max():.3f}, "
              f"flag ends at t={nb.tvec()[np.flatnonzero(x_)[-1]]:.2f} s (attack ends {nb.CONTROL_PATH[scen]['t1']} s)")
    for k in range(1, 7):
        tel, lab, log = nb.make_attacked("S02_fault_100ms", k, "U1", nb.notebook_attack_seed(nb.CFG["ATK_SEED_TEST"], "S02_fault_100ms", k), CAL)
        r, a_, x_ = nb.ups_check(tel, k, return_parts=True)
        att = lab == nb.LBL_ATTACK
        check(f"UPS check under U1 at DC{k}: non-zero on every attacked sample, 0 elsewhere",
              (r[att] > 0).all() and np.abs(r[~att]).max() == 0.0,
              f"onset flag share={x_[att].mean():.3f}, median power part={np.median(a_[att]):.3f} pu")

for k in range(1, 7):
    X = nb.features(nb.telemetry("S01_normal", k), k, CAL[k])[0]
    lab = np.zeros(len(X), np.int8)
    old = np.bincount(nb.WinSet(X, lab, k, CAL[k]["reg_edges"]).reg, minlength=3) / 742
    new = np.bincount(nb.winset(X, lab, k, CAL[k]).reg, minlength=3) / 742
    check(f"aux class DC{k} on raw scale near thirds", np.all(np.abs(new - 1 / 3) < 0.08),
          f"old={np.round(old, 3)} new={np.round(new, 3)}")

print("FAILURES:", fail)
sys.exit(1 if fail else 0)
