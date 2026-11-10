#!/usr/bin/env python3
"""Tables and numbers for the revision, from a results folder written by run_full_experiment.py.

  python3 revision_summary.py --root results_rev [--allow-nonrealtime]

Writes <root>/revision/:
  summary.json                    every value below
  Table_rev_gains.{csv,tex}       selected lambda_c, lambda_u, threshold per fold, validation scenario, grid-edge flags
  Table_rev_dc1.{csv,tex}         DC1 detection per regime, mean +/- std over the 5 LOSO folds (real-time streams)
  Table_rev_control.{csv,tex}     control-path and UPS-state-hiding streams (FULL model) at DC1 and the attacked sites
  Table_rev_sites.{csv,tex}       detection per regime at each site DC1..DC6 (DC2..DC6 from fast replay)
  Table_rev_clipped.csv           CSEC releases clipped at the end of the record, per stream group
  Table_rev_sensitivity.{csv,tex} S02 fold under assumed meter error and model mismatch, DC1, plus clean residual floors
Only numbers read from metrics.json / fusion.json / the dataset are written. Nothing is estimated.
"""
import argparse
import csv
import glob
import json
import os
import subprocess
import sys

import numpy as np
_sys_path = __import__("os").path.dirname(__import__("os").path.abspath(__file__))
__import__("sys").path.insert(0, _sys_path)
from fgpt import nb  # noqa: E402  (REV: warm-up setting)

HERE = os.path.dirname(os.path.abspath(__file__))
REG = ["T1", "T2", "T3a", "T3", "U1", "A1", "A2", "A3", "A4", "A5"]
GROUPS = REG + ["A1-A5"]
KEYS = ["roc_auc", "f1", "precision", "recall", "fpr"]


def load(pattern, allow_nonrt):
    out = []
    for p in sorted(glob.glob(pattern)):
        m = json.load(open(p))
        m["_dir"] = os.path.dirname(p)
        m["_rt_ok"] = bool(m.get("realtime")) or allow_nonrt
        out.append(m)
    return out


def ms(v):
    v = np.array([x for x in v if x is not None and np.isfinite(x)], float)
    return (float(v.mean()), float(v.std()), int(len(v))) if len(v) else (float("nan"), float("nan"), 0)


def group_of(test):
    return "A1-A5" if (test.startswith("A") and test[1:].isdigit()) else test


def regime_table(S, extra_key=None):
    """mean/std over folds per regime; A1-A5 pooled over 5 types x folds (as in the paper)."""
    rows = {}
    for g in GROUPS + ["no_attack"]:
        sub = [m for m in S if m["test"] == g or (g == "A1-A5" and group_of(m["test"]) == "A1-A5")]
        if not sub:
            continue
        src = [(m[extra_key] if extra_key else m) for m in sub]
        r = {k: ms([x[k] for x in src]) for k in KEYS}
        ev = [e for m in sub for e in m.get("attack_events", [])]
        r["intervals"] = f"{sum(e['detected'] for e in ev)}/{len(ev)}" if ev else "--"
        dl = [e["delay_s"] for e in ev if e["detected"]]
        dp = [e["provisional_delay_s"] for e in ev if e.get("provisional_detected")]
        r["delay_released_s"], r["delay_provisional_s"] = ms(dl), ms(dp)
        r["streams"] = len(sub)
        rows[g] = r
    return rows


def f3(t, nd=3):
    return "--" if (t is None or not t[2] or not np.isfinite(t[0])) else f"{t[0]:.{nd}f} $\\pm$ {t[1]:.{nd}f}"


def write(od, name, header, rows, tex_cols=None):
    with open(os.path.join(od, f"{name}.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    if tex_cols is not None:
        lines = ["\\begin{tabular}{" + "l" + "c" * (len(tex_cols) - 1) + "}", "\\hline",
                 " & ".join(tex_cols) + " \\\\", "\\hline"]
        for r in rows:
            cells = [str(x) if "$" in str(x) else str(x).replace("_", "\\_").replace("{", "\\{").replace("}", "\\}").replace('"', "")
                     for x in r[:len(tex_cols)]]
            lines.append(" & ".join(cells) + " \\\\")
        lines += ["\\hline", "\\end{tabular}"]
        open(os.path.join(od, f"{name}.tex"), "w").write("\n".join(lines) + "\n")


def residual_floor(dataset, pert):
    """95th percentile of |r_bal| (MW), |r_qp| (Mvar), |r_aux - delta_k| (pu) on attack-free S01..S06, worst site."""
    code = (
        "import sys,json,numpy as np;sys.path.insert(0,%r);from fgpt import nb;nb.DATASET_DIR=%r\n"
        "out={}\n"
        "for c in ('r_bal','r_qp','r_aux'):\n"
        "  out[c]=0.0\n"
        "for k in range(1,7):\n"
        "  cal=nb.calibrate_client(k)\n"
        "  for s in nb.CFG['ATTACK_FREE']:\n"
        "    R=nb.raw_residuals(nb.telemetry(s,k),k)\n"
        "    out['r_bal']=max(out['r_bal'],float(np.quantile(np.abs(R['r_bal']),0.95)))\n"
        "    out['r_qp']=max(out['r_qp'],float(np.quantile(np.abs(R['r_qp']),0.95)))\n"
        "    out['r_aux']=max(out['r_aux'],float(np.quantile(np.abs(R['r_aux']-cal['aux_off']),0.95)))\n"
        "print(json.dumps(out))\n") % (HERE, dataset)
    env = dict(os.environ, FGPT_PERTURB=json.dumps(pert))
    p = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    if p.returncode:
        print(p.stderr)
        return None
    return json.loads(p.stdout.strip().splitlines()[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results_rev")
    ap.add_argument("--dataset", default="data/dataset")
    ap.add_argument("--allow-nonrealtime", action="store_true", help="pipeline check: accept fast-replay DC1 streams")
    a = ap.parse_args()
    od = os.path.join(a.root, "revision")
    os.makedirs(od, exist_ok=True)
    NUM = {}

    # ---- gains per fold (A2)
    rows = []
    for fp in sorted(glob.glob(os.path.join(a.root, "folds", "*", "bundle", "fusion.json"))):
        f = json.load(open(fp))
        fold = os.path.basename(os.path.dirname(os.path.dirname(fp)))
        rows.append([fold, f.get("validation_scenario"), f["kappa"], f["eta"], round(f["threshold"], 4), round(f["val_f1"], 4),
                     f["val_windows"], f["val_pos"], f.get("val_csec_active_windows"),
                     f.get("gain_at_grid_edge", {}).get("kappa"), f.get("gain_at_grid_edge", {}).get("eta"),
                     ";".join(f.get("train_scenarios", []))])
    NUM["gains"] = rows
    if rows:
        write(od, "Table_rev_gains", ["fold", "validation", "lambda_c", "lambda_u", "tau", "val_F1", "val_windows", "val_pos",
                                      "val_csec_windows", "lambda_c_at_edge", "lambda_u_at_edge", "train"], rows,
              ["Fold", "Validation", "$\\lambda_c$", "$\\lambda_u$", "$\\tau^\\star$", "Val. $F_1$"])

    # ---- DC1 main streams (real time)
    S = [m for m in load(os.path.join(a.root, "streams", "*", "*", "metrics.json"), a.allow_nonrealtime) if m["_rt_ok"]]
    L = [m for m in S if os.path.basename(os.path.dirname(m["_dir"])) != "FULL"]
    dc1 = regime_table(L)
    dc1_x = regime_table(L, "excluding_clipped")
    NUM["dc1_by_regime"], NUM["dc1_by_regime_excluding_clipped"] = dc1, dc1_x
    rows = []
    for g in GROUPS + ["no_attack"]:
        if g in dc1:
            r, x = dc1[g], dc1_x.get(g, {})
            rows.append([g, f3(r["roc_auc"]), f3(r["f1"]), f3(r["precision"]), f3(r["recall"]), f3(r["fpr"], 4), r["intervals"],
                         f"{r['delay_provisional_s'][0]:.3f} / {r['delay_released_s'][0]:.3f}" if r["delay_released_s"][2] else "--",
                         f3(x.get("roc_auc")), f3(x.get("fpr"), 4), r["streams"]])
    if rows:
        write(od, "Table_rev_dc1", ["attack", "roc_auc", "f1", "precision", "recall", "fpr", "intervals", "delay_prov/rel_s",
                                    "roc_auc_excl_clipped", "fpr_excl_clipped", "streams"], rows,
              ["Attack", "ROC-AUC", "$F_1$", "Precision", "Recall", "FPR", "Intervals", "Delay (s)"])

    # ---- control path (A6) and UPS-state hiding (A3, U2)
    F = [m for m in S if os.path.basename(os.path.dirname(m["_dir"])) == "FULL"]
    F += [m for m in load(os.path.join(a.root, "streams_attacked", "DC*", "*", "metrics.json"), a.allow_nonrealtime) if m["_rt_ok"]]
    rows = []
    for m in sorted(F, key=lambda m: (m["scenario"], m["test"], m["dc"])):
        ev = m.get("attack_events", [])
        rows.append([m["scenario"], f"DC{m['dc']}", m["test"], round(m["roc_auc"], 4) if np.isfinite(m["roc_auc"]) else "--",
                     round(m["recall"], 3) if m["windows_pos"] else "--", round(m["fpr"], 4), m["windows_pos"],
                     f"{sum(e['detected'] for e in ev)}/{len(ev)}" if ev else "--",
                     next((round(e["delay_s"], 2) for e in ev if e["detected"]), "--"), m["realtime"]])
    NUM["control"] = rows
    if rows:
        write(od, "Table_rev_control", ["scenario", "site", "test", "roc_auc", "recall", "fpr", "pos_windows", "intervals",
                                        "delay_released_s", "realtime"], rows,
              ["Scenario", "Site", "Test", "ROC-AUC", "Recall", "FPR", "Pos.", "Intervals", "Delay (s)"])

    # ---- all sites (A5)
    sites = {1: L}
    for d in sorted(glob.glob(os.path.join(a.root, "streams_sites", "DC*"))):
        k = int(os.path.basename(d)[2:])
        sites[k] = load(os.path.join(d, "*", "*", "metrics.json"), True)
    rows, NUM["sites"] = [], {}
    for k, Sk in sorted(sites.items()):
        t = regime_table(Sk)
        NUM["sites"][f"DC{k}"] = t
        for g in GROUPS + ["no_attack"]:
            if g in t:
                rows.append([f"DC{k}", g, f3(t[g]["roc_auc"]), f3(t[g]["f1"]), f3(t[g]["fpr"], 4), t[g]["intervals"], t[g]["streams"],
                             "real time" if all(m.get("realtime") for m in Sk) else "fast replay"])
    if rows:
        write(od, "Table_rev_sites", ["site", "attack", "roc_auc", "f1", "fpr", "intervals", "streams", "mode"], rows,
              ["Site", "Attack", "ROC-AUC", "$F_1$", "FPR", "Intervals"])

    # ---- clipped releases (A9)
    rows = []
    for name, Sx in [("DC1 LOSO", L), ("control", F)] + [(f"DC{k} LOSO", v) for k, v in sites.items() if k != 1]:
        c = [m.get("clipped_release", {}) for m in Sx]
        rows.append([name, len(Sx), sum(x.get("windows", 0) for x in c), sum(x.get("windows_valid", 0) for x in c),
                     sum(x.get("windows_pos", 0) for x in c), sum(m["windows_valid"] for m in Sx)])
    NUM["clipped"] = rows
    write(od, "Table_rev_clipped", ["group", "streams", "clipped_windows", "clipped_valid", "clipped_pos", "valid_windows"], rows)

    # ---- sensitivity (A14)
    rows, NUM["sensitivity"] = [], {}
    plan = sorted(glob.glob(os.path.join(a.root, "experiment_plan*sens*.json"))) or sorted(glob.glob(os.path.join(a.root, "experiment_plan*all*.json")))
    cases = json.load(open(plan[-1]))["sens_cases"] if plan else {}
    for case in sorted(glob.glob(os.path.join(a.root, "sens", "*"))):
        cname = os.path.basename(case)
        Sc = load(os.path.join(case, "streams", "*", "metrics.json"), True)
        t = regime_table(Sc)
        fl = residual_floor(os.path.abspath(a.dataset), cases.get(cname, {})) if cname in cases else None
        NUM["sensitivity"][cname] = dict(perturb=cases.get(cname), by_regime=t, residual_floor_p95_worst_site=fl)
        for g in ["T1", "T2", "T3a", "T3", "U1", "A1-A5", "no_attack"]:
            if g in t:
                rows.append([cname, json.dumps(cases.get(cname, {})), g,
                             round(t[g]["roc_auc"][0], 4) if t[g]["roc_auc"][2] else "--",
                             round(t[g]["f1"][0], 4) if t[g]["f1"][2] else "--", round(t[g]["fpr"][0], 4),
                             f"{fl['r_bal']:.3g}" if fl else "--", f"{fl['r_qp']:.3g}" if fl else "--", f"{fl['r_aux']:.3g}" if fl else "--"])
    if rows:
        write(od, "Table_rev_sensitivity", ["case", "perturbation", "attack", "roc_auc", "f1", "fpr",
                                            "r_bal_p95_MW", "r_qp_p95_Mvar", "r_aux_p95_pu"], rows,
              ["Case", "Perturbation", "Attack", "ROC-AUC", "$F_1$", "FPR"])

    # ---- checks-only detection baseline (no learning): window score = max of the standardized check terms.
    # Computed from the same windows.csv, labels and valid windows as the detector. ROC-AUC only (no threshold is
    # tuned for it). "bal+aux+qp" uses only the three algebraic checks.
    from sklearn.metrics import roc_auc_score
    CHK = ["|r_bal|", "|r_aux|", "|r_cool|", "|r_ups|", "|r_qp|"]
    ALG = ["|r_bal|", "|r_aux|", "|r_qp|"]
    co = {}
    for k, Sk in sorted(sites.items()):
        for m in Sk:
            wp = os.path.join(m["_dir"], "windows.csv")
            if m["test"] in ("no_attack", "ctrl", "ctrl_hide") or not os.path.exists(wp):
                continue
            with open(wp) as fh:
                R = list(csv.DictReader(fh))
            y = np.array([int(r["label"]) for r in R])
            v = nb.eval_mask(y, np.array([float(r["t_end"]) for r in R]) + 1e-6)   # REV: warm-up excluded
            if len(set(y[v])) < 2:
                continue
            full = np.array([float(r["score"]) for r in R])
            chk = np.array([max(float(r[c]) for c in CHK) for r in R])
            alg = np.array([max(float(r[c]) for c in ALG) for r in R])
            g = group_of(m["test"])
            d = co.setdefault((f"DC{k}", g), dict(full=[], checks=[], alg=[]))
            d["full"].append(roc_auc_score(y[v], full[v])); d["checks"].append(roc_auc_score(y[v], chk[v]))
            d["alg"].append(roc_auc_score(y[v], alg[v]))
    rows = []
    for (site, g), d in sorted(co.items()):
        rows.append([site, g, f3(ms(d["full"])), f3(ms(d["checks"])), f3(ms(d["alg"])), len(d["full"])])
    NUM["checks_only"] = {f"{a}|{b}": {kk: ms(vv) for kk, vv in d.items()} for (a, b), d in co.items()}
    if rows:
        write(od, "Table_rev_checks_only", ["site", "attack", "fedguard_pt_roc_auc", "checks_only_roc_auc",
                                            "bal_aux_qp_roc_auc", "streams"], rows,
              ["Site", "Attack", "FedGuard-PT", "Checks only", "bal+aux+qp"])

    # ---- integrity: streamed vs batch scores over every stream found
    allm = (S + F + [m for kk, v in sites.items() if kk != 1 for m in v]
            + [m for c in glob.glob(os.path.join(a.root, "sens", "*")) for m in load(os.path.join(c, "streams", "*", "metrics.json"), True)])
    allm = list({m["_dir"]: m for m in allm}.values())
    d = [m["streaming_vs_batch_max_abs_diff"] for m in allm if np.isfinite(m.get("streaming_vs_batch_max_abs_diff", np.nan))]
    pr = [m.get("physics_rule", {}) for m in allm]
    NUM["integrity"] = dict(streams=len(allm), stream_vs_batch_max=max(d) if d else None,
                            realtime_streams=sum(bool(m.get("realtime")) for m in allm),
                            physics_rule_streams_enabled=sum(bool(x.get("enabled")) for x in pr),
                            physics_rule_fired_windows=sum(x.get("fired_windows", 0) for x in pr),
                            physics_rule_fired_on_clean_windows=sum(x.get("fired_valid_neg", 0) for x in pr),
                            physics_rule_fired_on_attack_windows=sum(x.get("fired_pos", 0) for x in pr))
    print("integrity:", NUM["integrity"])
    json.dump(NUM, open(os.path.join(od, "summary.json"), "w"), indent=1, default=float)
    print("written:", sorted(os.listdir(od)))


if __name__ == "__main__":
    main()
