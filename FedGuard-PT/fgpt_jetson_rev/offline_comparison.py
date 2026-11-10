#!/usr/bin/env python3
"""REV: offline comparison table (disturbance false-alarm rate, model-exchange cost, parameters) + DC1 detection.

Configurations (all use the same LOSO folds, validation scenario, calibration and test streams):
  pt       FedGuard-PT: the trained folds under --root (not retrained)
  nocsec   FedGuard-PT without CSEC: same TCNs; scoring weights, threshold refit with kappa = eta = 0
  central  centralized training: one TCN trained on the pooled windows of all six sites (same windows/epochs)
  raw      raw-signal variant: TCN input = the three signal channels only (v_z, p_z, q_z); targets, heads, scoring
           terms, CSEC and physics rule unchanged (FGPT_TCN_INPUT=raw)
  fgdc     FedGuard-DC baseline, ported from its notebook (fgpt/fgdc.py; adaptations listed there)
  checks   check-only rule: score = max of the five standardized physical-check terms; threshold = F1-max on the
           validation set (no learning)

Scores are computed offline with the batch path of rt_monitor.py (identical to the streamed released scores; the
pt rows are cross-checked against the stream metrics under --root). Windows ending before WARMUP_S are excluded.

  python3 offline_comparison.py --root results_rev --out results_cmp            # train what is missing, evaluate
  python3 offline_comparison.py --root results_rev --out results_cmp --stage train
  python3 offline_comparison.py --root results_rev --out results_cmp --stage eval
Outputs in --out: comparison_streams.csv (one row per configuration x fold x site x test),
Table_offline_comparison.{csv,tex}, comparison_summary.json, cross_check_pt.json
"""
import argparse
import csv
import glob
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fgpt import common, csec_rt, fgdc, nb  # noqa: E402

FOLDS = ["S02_fault_100ms", "S03_fault_250ms", "S04_line_trip", "S05_load_step", "S06_multi_event"]
PAPER_SET = ["S02_fault_100ms", "S03_fault_250ms", "S05_load_step", "S06_multi_event"]
VARIANTS = ["central", "pt", "nocsec", "raw", "fgdc", "checks"]
NAMES = dict(central="Centralized, pooled data", pt="FedGuard-PT", nocsec="FedGuard-PT, no CSEC",
             raw="Raw-signal variant", fgdc="FedGuard-DC", checks="Check-only rule")
DC1_TESTS = ["A1", "A2", "A3", "A4", "A5", "T1", "T2", "T3a", "T3", "U1"]
HERE = os.path.dirname(os.path.abspath(__file__))


def sh(cmd, env=None, log=None):
    print("$", " ".join(cmd), flush=True)
    e = dict(os.environ, **(env or {}))
    with open(log, "a") if log else open(os.devnull, "w") as lf:
        p = subprocess.run(cmd, env=e, cwd=HERE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        lf.write(p.stdout)
    if p.returncode:
        print(p.stdout[-3000:])
        raise SystemExit(f"failed: {' '.join(cmd)}")
    return p.stdout


# ------------------------------------------------------------------------------------------------- training
def train_tcn_variant(v, fold, a):
    d = os.path.join(a.out, v, "folds", fold)
    if os.path.exists(os.path.join(d, "bundle", "fusion.json")) and (v == "nocsec" or os.path.exists(os.path.join(d, "train_log.json"))):
        print(f"[{v} {fold}] done, skipped")
        return
    os.makedirs(d, exist_ok=True)
    lg = os.path.join(d, "train_console.log")
    if v == "nocsec":
        src = os.path.join(a.root, "folds", fold, "bundle")
        os.makedirs(os.path.join(d, "bundle"), exist_ok=True)
        for f in glob.glob(os.path.join(src, "DC*.npz")) + glob.glob(os.path.join(src, "calib_DC*.json")):
            shutil.copy2(f, os.path.join(d, "bundle"))
        sh([sys.executable, "train_edge.py", "--dataset", a.dataset, "--results", d, "--test-scenario", fold, "--refit-fusion"],
           env=dict(FGPT_KAPPA_GRID="0", FGPT_ETA_GRID="0"), log=lg)
    elif v == "central":
        sh([sys.executable, "train_edge.py", "--dataset", a.dataset, "--results", d, "--test-scenario", fold, "--mode", "central"], log=lg)
    elif v == "raw":
        sh([sys.executable, "train_edge.py", "--dataset", a.dataset, "--results", d, "--test-scenario", fold],
           env=dict(FGPT_TCN_INPUT="raw"), log=lg)


def validation_sets(fold, CAL):
    """Same validation windows as train_edge.py: validation scenario x T1/T2/T3a/T3 at all six sites."""
    val = nb.CFG["VAL_SCENARIO"][fold]
    out = []
    for reg in nb.REGIMES:
        tels, labs = {}, {}
        for k in range(1, 7):
            tels[k], labs[k], _ = nb.make_attacked(val, k, reg, nb.notebook_attack_seed(nb.CFG["ATK_SEED_VAL"], val, k), CAL)
        for k in range(1, 7):
            ws = nb.winset(nb.features(tels[k], k, CAL[k])[0], labs[k], k, CAL[k])
            out.append((k, tels[k], ws, np.flatnonzero(ws.y >= 0)))
    return val, out


def train_fgdc(fold, a, CAL):
    d = os.path.join(a.out, "fgdc", "folds", fold)
    if os.path.exists(os.path.join(d, "fgdc.json")):
        print(f"[fgdc {fold}] done, skipped")
        return
    os.makedirs(d, exist_ok=True)
    val = nb.CFG["VAL_SCENARIO"][fold]
    train = [s for s in nb.CFG["ATTACK_FREE"] if s not in (fold, val)]
    t = nb.tvec()
    scale = {k: float(max(nb.telemetry(s, k)["P_total_MW"].max() for s in train)) for k in range(1, 7)}
    clients = []
    for k in range(1, 7):
        Xs, ys = zip(*(fgdc.train_windows(fgdc.site_series(nb.telemetry(s, k), scale[k]), t) for s in train))
        clients.append(dict(X=np.concatenate(Xs), y=np.concatenate(ys)))
    print(f"[fgdc {fold}] train={train} windows/site={len(clients[0]['y'])}")
    G, st = fgdc.train_fed(clients)
    # calibration: clean windows of the calibration scenario, pooled over sites (all windows, hop TRAIN_HOP)
    cx, cy = zip(*(fgdc.train_windows(fgdc.site_series(nb.telemetry(nb.CFG["CALIB_SCENARIO"], k), scale[k]), t) for k in range(1, 7)))
    cal = fgdc.calibrate(G, np.concatenate(cx), np.concatenate(cy))
    _, vs = validation_sets(fold, CAL)
    zr, zf, yv = [], [], []
    for k, tel, ws, idx in vs:
        X, y = fgdc.windows_at_ends(fgdc.site_series(tel, scale[k]), ws.starts[idx] + nb.CFG["WIN"] - 1)
        r_, f_ = fgdc.components(G, X, y, cal)
        zr.append(r_); zf.append(f_); yv.append((ws.y[idx] == 1).astype(int))
    tun = fgdc.tune_alpha_beta(np.concatenate(zr), np.concatenate(zf), np.concatenate(yv))
    np.savez(os.path.join(d, "fgdc.npz"), **G)
    common.save_json(dict(scale=scale, calib=cal, tuning=tun, stats=st, train_scenarios=train, validation_scenario=val,
                          W=fgdc.W, H=fgdc.H, rounds=fgdc.ROUNDS, epochs=fgdc.EPOCHS, train_hop=fgdc.TRAIN_HOP),
                     os.path.join(d, "fgdc.json"))
    print(f"[fgdc {fold}] {st['wall_s']:.0f} s, params {st['params']}, val F1 {tun['val_f1']:.4f}, "
          f"alpha {tun['alpha']:.2f} beta {tun['beta']:.2f} thr {tun['threshold']:.3f}")


def tune_checks(fold, a, CAL):
    d = os.path.join(a.out, "checks", "folds", fold)
    fp = os.path.join(d, "checks.json")
    if os.path.exists(fp):
        return
    from train_edge import window_terms
    os.makedirs(d, exist_ok=True)
    fusion, models = load_tcn(os.path.join(a.root, "folds", fold))
    ci = [nb.TERMS.index(c) for c in nb.CHECK_TERMS]
    _, vs = validation_sets(fold, CAL)
    s, y = [], []
    for k, tel, ws, idx in vs:
        Z = nb.z_terms(window_terms(models[k], ws, idx), nb.fusion_ms(fusion, k))
        s.append(Z[:, ci].max(1)); y.append((ws.y[idx] == 1).astype(int))
    thr, f1 = nb.best_f1_threshold(np.concatenate(s), np.concatenate(y))
    common.save_json(dict(threshold=thr, val_f1=f1, terms=nb.CHECK_TERMS, source_fold=os.path.join(a.root, "folds", fold)), fp)
    print(f"[checks {fold}] threshold {thr:.3f}, val F1 {f1:.4f}")


# ------------------------------------------------------------------------------------------------- scoring
def load_tcn(fold_dir):
    b = os.path.join(fold_dir, "bundle")
    fusion = common.load_config(os.path.join(b, "fusion.json"))
    return fusion, {k: dict(np.load(os.path.join(b, f"DC{k}.npz"))) for k in range(1, 7)}


def set_variant_globals(fusion):
    nb.LEGACY_UPS = bool(fusion.get("ups", {}).get("legacy", True))
    nb.TCN_INPUT = nb.SIG_IDX if fusion.get("tcn_input") == "raw" else None


def evaluate(a):
    from rt_monitor import build_test
    from train_edge import window_terms
    rows, xcheck = [], []
    outp = os.path.join(a.out, "comparison_streams.csv")
    for fold in a.folds:
        pt_dir = os.path.join(a.root, "folds", fold)
        CAL = {j: nb.cal_from_json(common.load_config(os.path.join(pt_dir, "bundle", f"calib_DC{j}.json"))) for j in range(1, 7)}
        tcn = {}
        for v in ("pt", "nocsec", "central", "raw"):
            if v in a.variants:
                tcn[v] = load_tcn(pt_dir if v == "pt" else os.path.join(a.out, v, "folds", fold))
        if "checks" in a.variants:
            chk = common.load_config(os.path.join(a.out, "checks", "folds", fold, "checks.json"))
            pt_fusion, pt_models = tcn["pt"] if "pt" in tcn else load_tcn(pt_dir)
        if "fgdc" in a.variants:
            fj = common.load_config(os.path.join(a.out, "fgdc", "folds", fold, "fgdc.json"))
            G = dict(np.load(os.path.join(a.out, "fgdc", "folds", fold, "fgdc.npz")))
        jobs = [(k, "no_attack") for k in range(1, 7)] + ([(1, t) for t in DC1_TESTS] if a.detection else [])
        T = nb.tvec()
        for k, test in jobs:
            t0 = time.perf_counter()
            nb.LEGACY_UPS = False
            tel, lab, tels, _ = build_test(test, fold, k, CAL, a.dataset)
            corr, unc, _, _, _, clipped = csec_rt.csec_with_release(tels, CAL, k, 0, return_clipped=True)
            ws = nb.winset(nb.features(tel, k, CAL[k])[0], lab, k, CAL[k])
            ends = ws.starts + nb.CFG["WIN"] - 1
            ywin = ws.y
            valid = nb.eval_mask(ywin, T[ends])
            y = (ywin[valid] == 1).astype(int)
            cw, uw = nb.csec_window(corr, ws.starts), nb.csec_window(unc, ws.starts)
            allidx = np.arange(len(ws.starts))
            scores = {}
            for v, (fusion, models) in tcn.items():
                set_variant_globals(fusion)
                Z = nb.z_terms(window_terms(models[k], ws, allidx), nb.fusion_ms(fusion, k))
                scores[v] = (nb.fused_score(fusion["coef"], fusion["intercept"], Z, cw, uw, fusion["kappa"], fusion["eta"],
                                            fusion.get("physics_rule")), fusion["threshold"])
            nb.TCN_INPUT = None
            if "checks" in a.variants:
                set_variant_globals(pt_fusion)
                Z = nb.z_terms(window_terms(pt_models[k], ws, allidx), nb.fusion_ms(pt_fusion, k))
                ci = [nb.TERMS.index(c) for c in nb.CHECK_TERMS]
                scores["checks"] = (Z[:, ci].max(1), chk["threshold"])
            if "fgdc" in a.variants:
                X, yy = fgdc.windows_at_ends(fgdc.site_series(tel, fj["scale"][str(k)]), ends)
                zr, zf = fgdc.components(G, X, yy, fj["calib"])
                tu = fj["tuning"]
                scores["fgdc"] = (tu["alpha"] * zr + tu["beta"] * zf, tu["threshold"])
            ev = {"S02_fault_100ms": 60.0, "S03_fault_250ms": 60.0, "S06_multi_event": 60.0,
                  "S04_line_trip": 90.0, "S05_load_step": 120.0}[fold]
            dwin = valid & (T[ends] >= ev) & (T[ws.starts] <= ev + 10.0) & (ywin == 0)
            for v in a.variants:
                if v not in scores:
                    continue
                s, thr = scores[v]
                m = nb.metrics_from(s[valid], y, thr)
                al = s > thr
                rows.append(dict(config=v, fold=fold, dc=k, test=test, roc_auc=m["roc_auc"], f1=m["f1"], recall=m["recall"],
                                 precision=m["precision"], fpr=m["fpr"], n_neg=m["n_neg"], n_pos=m["n_pos"],
                                 dist10_false_alarms=int(al[dwin].sum()), dist10_neg=int(dwin.sum()), threshold=thr))
            if "pt" in scores:
                xcheck.append(dict(fold=fold, dc=k, test=test, scores=scores["pt"][0], ws=ws))
            print(f"[eval {fold} DC{k} {test}] {time.perf_counter() - t0:.1f} s  " +
                  "  ".join(f"{v}:FPR={r['fpr']:.4f}" for v in a.variants for r in rows[-len(scores):] if r['config'] == v), flush=True)
    with open(outp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    cross_check(a, xcheck)
    summarize(a, rows)


def cross_check(a, xcheck):
    """pt rows vs the stream metrics under --root (same fold, site, test)."""
    out = []
    for x in xcheck:
        sub = "streams" if x["dc"] == 1 else os.path.join("streams_sites", f"DC{x['dc']}")
        wp = os.path.join(a.root, sub, x["fold"], f"{x['test']}_{x['fold']}", "windows.csv")
        if not os.path.exists(wp):
            continue
        with open(wp) as f:
            R = list(csv.DictReader(f))
        s_stream = np.array([float(r["score"]) for r in R])
        n = len(s_stream)
        out.append(dict(fold=x["fold"], dc=x["dc"], test=x["test"], windows_stream=n, windows_batch=int(len(x["scores"])),
                        max_abs_diff=float(np.abs(x["scores"][:n] - s_stream).max())))
    md = max((o["max_abs_diff"] for o in out), default=float("nan"))
    common.save_json(dict(streams=len(out), max_abs_diff=md, rows=out), os.path.join(a.out, "cross_check_pt.json"))
    print(f"[cross-check] FedGuard-PT batch vs streamed scores over {len(out)} streams: max |diff| {md:.2e} "
          f"(stream scores are stored with 5 decimals)")


# ------------------------------------------------------------------------------------------------- summary
def cost(v, a):
    """(KiB per round for all six sites, parameters) from the training logs; None where not applicable."""
    if v in ("pt", "nocsec", "raw", "central"):
        lg = os.path.join(a.root if v in ("pt", "nocsec") else os.path.join(a.out, v), "folds", a.folds[0], "train_log.json")
        d = json.load(open(lg))
        return (None if v == "central" else d["KiB_per_round_all_clients"]), d["params_total"]
    if v == "fgdc":
        st = json.load(open(os.path.join(a.out, "fgdc", "folds", a.folds[0], "fgdc.json")))["stats"]
        return st["KiB_per_round_all_clients"], st["params"]
    return None, None


def summarize(a, rows):
    S = dict(warmup_s=nb.CFG["WARMUP_S"], folds=a.folds, paper_set=PAPER_SET, configs={})
    tab = []
    for v in a.variants:
        R = [r for r in rows if r["config"] == v]
        na = lambda fs: np.array([r["fpr"] for r in R if r["test"] == "no_attack" and r["fold"] in fs])  # noqa: E731
        fp, fa = na(PAPER_SET), na(a.folds)
        d10 = [r for r in R if r["test"] == "no_attack" and r["fold"] in PAPER_SET]
        det = [r for r in R if r["test"] != "no_attack"]
        kib, npar = cost(v, a)
        c = dict(name=NAMES[v], fpr_paper_mean=float(fp.mean()) if len(fp) else None, fpr_paper_std=float(fp.std()) if len(fp) else None,
                 fpr_paper_n=int(len(fp)), fpr_all_folds_mean=float(fa.mean()) if len(fa) else None,
                 fpr_all_folds_std=float(fa.std()) if len(fa) else None, fpr_all_folds_n=int(len(fa)),
                 dist10_false_alarms=int(sum(r["dist10_false_alarms"] for r in d10)), dist10_neg=int(sum(r["dist10_neg"] for r in d10)),
                 KiB_per_round=kib, params=npar)
        if det:
            for g, tests in (("A1-A5", ["A1", "A2", "A3", "A4", "A5"]), ("T1-T2", ["T1", "T2"]), ("T3a", ["T3a"]), ("T3", ["T3"]), ("U1", ["U1"])):
                G_ = [r for r in det if r["test"] in tests and r["fold"] in PAPER_SET]
                c[f"DC1_{g}"] = dict(roc_auc=float(np.nanmean([r["roc_auc"] for r in G_])), f1=float(np.mean([r["f1"] for r in G_])),
                                     recall=float(np.mean([r["recall"] for r in G_])), fpr=float(np.mean([r["fpr"] for r in G_])), n=len(G_))
        S["configs"][v] = c
        tab.append(c)
    common.save_json(S, os.path.join(a.out, "comparison_summary.json"))
    hdr = ["Configuration", "FPR mean", "FPR std", "n", "KiB/round", "Parameters", "FPR all folds", "dist10 FA/neg"]
    dets = [k for k in ("DC1_A1-A5", "DC1_T1-T2", "DC1_T3a", "DC1_T3", "DC1_U1") if k in tab[0]]
    with open(os.path.join(a.out, "Table_offline_comparison.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(hdr + [f"{k} ROC-AUC" for k in dets] + [f"{k} F1" for k in dets])
        for c in tab:
            w.writerow([c["name"], f"{c['fpr_paper_mean']:.4f}", f"{c['fpr_paper_std']:.4f}", c["fpr_paper_n"],
                        "--" if c["KiB_per_round"] is None else f"{c['KiB_per_round']:.2f}", "--" if c["params"] is None else c["params"],
                        f"{c['fpr_all_folds_mean']:.4f}+-{c['fpr_all_folds_std']:.4f}", f"{c['dist10_false_alarms']}/{c['dist10_neg']}"]
                       + [f"{c[k]['roc_auc']:.3f}" for k in dets] + [f"{c[k]['f1']:.3f}" for k in dets])
    L = [r"\begin{tabular}{lccc}", r"\hline", r"Configuration & FPR (mean $\pm$ std) & KiB/round & Parameters\\", r"\hline"]
    for v, c in zip(a.variants, tab):
        nm = r"\textbf{FedGuard-PT}" if v == "pt" else (r"FedGuard-DC~\cite{Saroare2026FedGuardDC}" if v == "fgdc" else c["name"])
        f_ = f"{c['fpr_paper_mean']:.4f}\\pm{c['fpr_paper_std']:.4f}"
        f_ = rf"$\mathbf{{{f_}}}$" if v == "pt" else f"${f_}$"
        kib = "--" if c["KiB_per_round"] is None else f"{c['KiB_per_round']:.2f}"
        par = "--" if c["params"] is None else f"{c['params']:,}".replace(",", r"\,")
        L.append(f"{nm}\n& {f_} & {kib} & {par}\\\\")
    L += [r"\hline", r"\end{tabular}"]
    open(os.path.join(a.out, "Table_offline_comparison.tex"), "w").write("\n".join(L) + "\n")
    print("\n" + "\n".join(f"{c['name']:<28} FPR {100 * c['fpr_paper_mean']:6.2f} +- {100 * c['fpr_paper_std']:5.2f} % (n={c['fpr_paper_n']})"
                           f"  KiB/round {c['KiB_per_round'] if c['KiB_per_round'] is None else round(c['KiB_per_round'], 2)}"
                           f"  params {c['params']}  10-s post-event FA {c['dist10_false_alarms']}/{c['dist10_neg']}"
                           + ("".join(f"  {k[4:]} AUC {c[k]['roc_auc']:.3f}" for k in dets)) for c in tab))
    print(f"-> {a.out}/Table_offline_comparison.tex, .csv, comparison_summary.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results_rev", help="finished FedGuard-PT run (folds/, streams/, streams_sites/)")
    ap.add_argument("--out", default="results_cmp")
    ap.add_argument("--dataset", default="data/dataset")
    ap.add_argument("--stage", default="all", choices=["all", "train", "eval"])
    ap.add_argument("--folds", default=",".join(FOLDS))
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--no-detection", dest="detection", action="store_false", help="skip the DC1 attacked tests")
    a = ap.parse_args()
    a.folds, a.variants = a.folds.split(","), a.variants.split(",")
    a.root, a.out, a.dataset = (os.path.abspath(x) for x in (a.root, a.out, a.dataset))
    nb.DATASET_DIR = a.dataset
    os.makedirs(a.out, exist_ok=True)
    if a.stage in ("all", "train"):
        for fold in a.folds:
            CAL = {j: nb.cal_from_json(common.load_config(os.path.join(a.root, "folds", fold, "bundle", f"calib_DC{j}.json")))
                   for j in range(1, 7)}
            nb.LEGACY_UPS = False
            for v in ("nocsec", "central", "raw"):
                if v in a.variants:
                    train_tcn_variant(v, fold, a)
            if "checks" in a.variants:
                tune_checks(fold, a, CAL)
            if "fgdc" in a.variants:
                train_fgdc(fold, a, CAL)
    if a.stage in ("all", "eval"):
        evaluate(a)


if __name__ == "__main__":
    main()
