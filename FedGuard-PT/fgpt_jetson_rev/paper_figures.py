#!/usr/bin/env python3
"""IEEE Transactions figures and tables from the board's real-time experiment (results_rt/).

Only streams with metrics.json "realtime": true are used. Figures are 3.5 in (single column) or
7.16 in (double column) wide, large type, TrueType fonts, PDF + 600 dpi PNG.
Output: results_rt/paper/  (Fig*.pdf/png, Table*.csv/.tex, numbers.json with every plotted value)
"""
import argparse
import csv
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import sys as _sys
_sys_path = __import__("os").path.dirname(__import__("os").path.abspath(__file__))
__import__("sys").path.insert(0, _sys_path)
from fgpt import nb  # noqa: E402  (REV: warm-up setting)

W1, W2 = 3.5, 7.16
C = dict(blue="#1f77b4", orange="#ff7f0e", green="#2ca02c", red="#d62728", purple="#9467bd", grey="#7f7f7f", brown="#8c564b")
REG_ORDER = ["T1", "T2", "T3a", "T3", "U1", "A"]
REG_LABEL = {"T1": "T1", "T2": "T2", "T3a": "T3a", "T3": "T3", "U1": "U1", "A": "A1–A5", "ctrl_hide": "U2\nS07"}
SCEN_SHORT = {"S02_fault_100ms": "S02\nfault", "S03_fault_250ms": "S03\nfault", "S04_line_trip": "S04\nline trip",
              "S05_load_step": "S05\nload step", "S06_multi_event": "S06\nmulti", "S07_atk_false_ups": "S07",
              "S08_atk_load_alter": "S08", "S09_atk_sensor_spoof": "S09", "S10_atk_cooling": "S10", "S11_fault_plus_atk": "S11"}
NUM = {}


def style(fs):
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "Nimbus Roman", "Liberation Serif", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": fs, "axes.labelsize": fs + 1, "axes.titlesize": fs + 1,
        "xtick.labelsize": fs, "ytick.labelsize": fs, "legend.fontsize": fs - 1, "legend.frameon": False,
        "lines.linewidth": 1.8, "axes.linewidth": 1.0, "axes.grid": True, "grid.alpha": 0.35, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.dpi": 600, "savefig.bbox": "tight", "savefig.pad_inches": 0.03})


def save(fig, od, name, warn):
    if warn:
        fig.text(0.5, 0.5, "NOT REAL-TIME", fontsize=40, color="red", alpha=0.25, ha="center", va="center", rotation=30)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(od, f"{name}.{ext}"))
    plt.close(fig)
    print("saved", name)


def rcsv(path):
    with open(path) as f:
        r = csv.reader(f); h = next(r); rows = list(r)
    out = {}
    for j, c in enumerate(h):
        col = [x[j] for x in rows]
        try:
            out[c] = np.array([float(v) if v != "" else np.nan for v in col])
        except ValueError:
            out[c] = np.array(col)
    return out


def load_streams(root, allow_nonrt):
    S = []
    for mp in sorted(glob.glob(os.path.join(root, "streams", "*", "*", "metrics.json"))):
        m = json.load(open(mp))
        if not m.get("realtime") and not allow_nonrt:
            print("skip (not real-time):", mp)
            continue
        m["_dir"] = os.path.dirname(mp)
        m["_fold"] = os.path.basename(os.path.dirname(m["_dir"]))
        m["_group"] = "A" if (m["test"].startswith("A") and m["test"][1:].isdigit()) else m["test"]
        S.append(m)
    return S


def panel(ax, tag):
    ax.text(0.0, 1.02, tag, transform=ax.transAxes, ha="left", va="bottom", fontweight="bold")


# ------------------------------------------------------------------------------------------ Fig: timeline
def fig_timeline(m, od, warn, xlim=None, suffix=""):
    sig = np.load(os.path.join(m["_dir"], "signals.npz"))
    w = rcsv(os.path.join(m["_dir"], "windows.csv"))
    t, lab = sig["t"], sig["label"]
    fig, ax = plt.subplots(4, 1, figsize=(W2, 6.6), sharex=True, gridspec_kw=dict(height_ratios=[1, 1.15, 1.15, 0.65]))
    ax[0].plot(t, sig["Va"], c="k", lw=1.3); ax[0].set_ylabel("$V_a$ (pu)")
    ax[1].plot(t, sig["P_total"], c=C["blue"], lw=1.3, label="$P_{total}$ reported")
    ax[1].plot(t, sig["P_fac"], c=C["orange"], lw=1.3, ls="--", label="$P_{it}+P_{cool}+P_{aux}$")
    ax[1].set_ylabel("$P$ (MW)")
    tt = w["t_end"] + 0.01
    order = np.argsort(w["t_release"])
    ax[2].step(tt, w["prov_score"], where="post", c=C["purple"], ls=":", lw=1.4, label="at window end (no CSEC)")
    ax[2].step(w["t_release"][order], w["score"][order], where="post", c=C["green"], lw=1.6, label="released (CSEC)")
    ax[2].axhline(m["threshold"], c=C["red"], ls="--", lw=1.3, label="threshold")
    ax[2].set_ylabel("score")
    ax[2].set_yscale("symlog", linthresh=2.0)
    ax[3].fill_between(t, 0, (lab == 1).astype(float), step="post", color=C["red"], alpha=0.35, lw=0, label="attack")
    ax[3].step(w["t_release"][order], 1.15 * w["alarm"][order], where="post", c=C["blue"], lw=1.6, label="alarm")
    ax[3].set_ylim(-0.1, 1.35); ax[3].set_yticks([0, 1]); ax[3].set_xlabel("time (s)")
    for a, tag in zip(ax, "abcd"):
        on = np.flatnonzero(np.diff(np.r_[0, (lab == 1).astype(int), 0]))
        for s0, s1 in zip(on[::2], on[1::2]):
            a.axvspan(t[s0], t[s1 - 1], color=C["red"], alpha=0.08, lw=0)
        panel(a, f"({tag})")
    # REV: full-record figures start after the evaluation warm-up (nb.CFG["WARMUP_S"])
    ax[0].set_xlim(*(xlim or (nb.CFG["WARMUP_S"], float(t[-1]))))
    hs, ls = [], []
    for a in ax:
        h, l = a.get_legend_handles_labels(); hs += h; ls += l
    fig.legend(hs, ls, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.0), fontsize=plt.rcParams["font.size"] - 1,
               handlelength=1.8, columnspacing=1.0)
    fig.align_ylabels(ax)
    fig.tight_layout(h_pad=0.4, rect=(0, 0, 1, 0.925))
    save(fig, od, f"Fig_rt_timeline_{m['_fold']}_{m['test']}_{m['scenario']}{suffix}", warn)


# ------------------------------------------------------------------------------------------ Fig: detection
def agg(values):
    v = np.array([x for x in values if x is not None and np.isfinite(x)], float)
    return (float(v.mean()), float(v.std()), int(len(v))) if len(v) else (np.nan, np.nan, 0)


def fig_detection(S, od, warn):
    L = [m for m in S if m["_fold"] != "FULL"]
    rows = {}
    for g in REG_ORDER:
        sub = [m for m in L if m["_group"] == g]
        rows[g] = {k: agg([m[k] for m in sub]) for k in ("roc_auc", "f1", "precision", "recall", "fpr")}
    na = [m for m in L if m["test"] == "no_attack"]
    rows["no_attack"] = {"fpr": agg([m["fpr"] for m in na])}
    NUM["detection_by_regime"] = rows
    groups = [g for g in REG_ORDER if rows[g]["roc_auc"][2]]
    if not groups:
        return rows
    fig, ax = plt.subplots(1, 3, figsize=(W2, 3.2))
    x = np.arange(len(groups))
    for a, key, lab, col in ((ax[0], "roc_auc", "ROC-AUC", C["blue"]), (ax[1], "f1", "F1", C["green"])):
        mu = [rows[g][key][0] for g in groups]; sd = [rows[g][key][1] for g in groups]
        a.bar(x, mu, yerr=sd, color=col, capsize=4, width=0.65, error_kw=dict(lw=1.2))
        a.set_xticks(x); a.set_xticklabels([REG_LABEL[g] for g in groups], rotation=45, ha="right", rotation_mode="anchor")
        a.set_ylim(0, 1.05); a.set_ylabel(lab)
    fg = groups + (["no_attack"] if rows["no_attack"]["fpr"][2] else [])
    mu = [rows[g]["fpr"][0] for g in fg]; sd = [rows[g]["fpr"][1] for g in fg]
    ax[2].bar(np.arange(len(fg)), mu, yerr=sd, color=[C["red"]] * len(groups) + [C["grey"]] * (len(fg) - len(groups)), capsize=4, width=0.65)
    ax[2].set_xticks(np.arange(len(fg))); ax[2].set_xticklabels([REG_LABEL.get(g, "none") for g in fg], rotation=45, ha="right",
                                                             rotation_mode="anchor"); ax[2].set_ylabel("FPR")
    for a, tag in zip(ax, "abc"):
        panel(a, f"({tag})")
    fig.tight_layout(w_pad=0.8)
    save(fig, od, "Fig_rt_detection_by_regime", warn)
    return rows


def fig_disturbance(S, od, warn):
    na = sorted([m for m in S if m["test"] == "no_attack"], key=lambda m: m["scenario"])
    if not na:
        return
    fig, ax = plt.subplots(figsize=(W1, 2.8))
    x = np.arange(len(na))
    fpr = [m["fpr"] for m in na]
    ax.bar(x, fpr, color=C["grey"], width=0.65)
    for i, m in enumerate(na):
        d = m.get("disturbance_window_10s")
        if d:
            ax.text(i, fpr[i], f"{d['false_alarms']}/{d['neg_windows']}", ha="center", va="bottom", fontsize=plt.rcParams["font.size"] - 1)
    ax.set_xticks(x); ax.set_xticklabels([SCEN_SHORT[m["scenario"]] for m in na]); ax.set_ylabel("false-positive rate")
    ax.set_ylim(0, max(max(fpr) * 1.35, 0.02))
    NUM["disturbance_fpr"] = {m["scenario"]: dict(fpr=m["fpr"], windows=m["windows_valid"], event_window=m.get("disturbance_window_10s")) for m in na}
    fig.tight_layout()
    save(fig, od, "Fig_rt_disturbance_false_alarms", warn)


def fig_delay(S, od, warn):
    data, labels = [], []
    for g in REG_ORDER + ["ctrl", "ctrl_hide"]:
        fin = [e["delay_s"] for m in S if m["_group"] == g for e in m["attack_events"] if e["detected"]]
        prv = [e["provisional_delay_s"] for m in S if m["_group"] == g for e in m["attack_events"] if e.get("provisional_detected")]
        n = sum(len(m["attack_events"]) for m in S if m["_group"] == g)
        if n:
            data.append((fin, prv)); labels.append((REG_LABEL.get(g, "Ctrl\nS07"), n, len(fin), len(prv)))
    if not data:
        return
    NUM["detection_delay_s"] = {l[0].replace("\n", " "): dict(events=l[1], detected=l[2], provisional_detected=l[3],
                                                            final=agg(d[0]), provisional=agg(d[1]))
                                for l, d in zip(labels, data)}
    fig, ax = plt.subplots(figsize=(W1, 3.1))
    pos = np.arange(len(data))
    b1 = ax.boxplot([d[1] if d[1] else [np.nan] for d in data], positions=pos - 0.18, widths=0.3, patch_artist=True, showfliers=True)
    b2 = ax.boxplot([d[0] if d[0] else [np.nan] for d in data], positions=pos + 0.18, widths=0.3, patch_artist=True, showfliers=True)
    for b, col in ((b1, C["purple"]), (b2, C["green"])):
        for p_ in b["boxes"]:
            p_.set_facecolor(col); p_.set_alpha(0.55)
        for med in b["medians"]:
            med.set_color("k")
    ax.set_xticks(pos); ax.set_xticklabels([l[0].replace("\n", " ") for l in labels], rotation=45, ha="right", rotation_mode="anchor")
    ax.set_ylabel("detection delay (s)")
    ax.legend([b1["boxes"][0], b2["boxes"][0]], ["window end", "released (CSEC)"], loc="upper left")
    fig.tight_layout()
    save(fig, od, "Fig_rt_detection_delay", warn)


def fig_control(S, od, warn):
    F = sorted([m for m in S if m["_fold"] == "FULL"], key=lambda m: (m["scenario"], m["test"]))
    if not F:
        return
    NUM["control_path"] = {f"{m['test']}_{m['scenario']}": dict(auc=m["roc_auc"], recall=m["recall"], fpr=m["fpr"], pos=m["windows_pos"],
                                                               valid=m["windows_valid"], events=m["attack_events"]) for m in F}
    fig, ax = plt.subplots(figsize=(W1, 2.8))
    x = np.arange(len(F))
    ax.bar(x - 0.2, [m["recall"] if m["windows_pos"] else 0 for m in F], 0.38, color=C["green"], label="recall")
    ax.bar(x + 0.2, [m["fpr"] for m in F], 0.38, color=C["red"], label="FPR")
    for i, m in enumerate(F):
        if not m["windows_pos"]:
            ax.text(i - 0.2, 0.02, "n/a", ha="center", va="bottom", rotation=90, fontsize=plt.rcParams["font.size"] - 2)
    ax.set_xticks(x); ax.set_xticklabels([SCEN_SHORT[m["scenario"]] + ("\nU2" if m["test"] == "ctrl_hide" else "") for m in F]); ax.set_ylim(0, 1.3)
    ax.set_ylabel("rate"); ax.legend(loc="upper right", ncol=2)
    fig.tight_layout()
    save(fig, od, "Fig_rt_control_path", warn)


# ------------------------------------------------------------------------------------------ Fig: timing
def fig_latency(S, od, warn):
    norm, tcn, late, dec, wait, miss, tot = [], [], [], [], [], 0, 0
    for m in S:
        z = np.load(os.path.join(m["_dir"], "sample_timing.npz"))
        norm.append(z["proc_ms"][~z["tcn"]]); tcn.append(z["proc_ms"][z["tcn"]]); late.append(z["lateness_ms"])
        miss += int(z["missed"].sum()); tot += len(z["missed"])
        w = rcsv(os.path.join(m["_dir"], "windows.csv"))
        dec.append(w["decision_ms_after_target"]); wait.append(w["csec_wait_ms"])
    norm, tcn, late, dec, wait = (np.concatenate(v) for v in (norm, tcn, late, dec, wait))
    q = lambda v: dict(mean=float(v.mean()), p50=float(np.percentile(v, 50)), p99=float(np.percentile(v, 99)),  # noqa: E731
                       p999=float(np.percentile(v, 99.9)), max=float(v.max()), n=int(len(v)))
    compute_only = dec[wait == 0]
    NUM["timing_ms"] = dict(sample_features=q(norm), sample_with_tcn_window=q(tcn), start_lateness=q(late),
                            decision_after_target_no_csec_wait=q(compute_only) if len(compute_only) else None,
                            csec_wait_nonzero=q(wait[wait > 0]) if (wait > 0).any() else None,
                            windows_waiting_for_csec=int((wait > 0).sum()), windows=int(len(wait)),
                            deadline_misses=miss, samples=tot, streams=len(S))
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.9))
    for v, lab, col in ((norm, "feature update", C["blue"]), (tcn, "feature + TCN + fusion", C["orange"]),
                        (late[late > 0], "start lateness (>0)", C["grey"])):
        if len(v):
            s_ = np.sort(v); ax[0].plot(s_, np.arange(1, len(s_) + 1) / len(s_), c=col, label=lab)
    ax[0].axvline(10.0, c=C["red"], ls="--", lw=1.3, label="10 ms period")
    ax[0].set_xscale("log"); ax[0].set_xlabel("time per sample (ms)"); ax[0].set_ylabel("CDF"); ax[0].legend(loc="center right", fontsize=plt.rcParams["font.size"] - 2)
    if (wait > 0).any():
        ws = np.sort(wait[wait > 0] / 1e3)
        ax[1].plot(ws, np.arange(1, len(ws) + 1) / len(ws), c=C["purple"])
        ax[1].text(0.97, 0.05, f"{len(ws)} of {len(wait)} windows wait", transform=ax[1].transAxes, ha="right", va="bottom")
    ax[1].set_xlabel("CSEC release wait (s)"); ax[1].set_ylabel("CDF")
    for a, tag in zip(ax, "ab"):
        panel(a, f"({tag})")
    fig.tight_layout(w_pad=1.0)
    save(fig, od, "Fig_rt_latency", warn)


# ------------------------------------------------------------------------------------------ Fig: resources
def res_stats(files, phase_prefix=None):
    vals = {}
    for f in files:
        r = rcsv(f)
        m = np.ones(len(r["t_s"]), bool) if phase_prefix is None else np.char.startswith(r["phase"].astype(str), phase_prefix)
        for k in ("proc_cpu_pct", "sys_cpu_pct", "proc_rss_MiB", "vdd_in_mW", "vdd_cpu_gpu_cv_mW", "vdd_soc_mW", "gpu_load_pct", "tj_temp_C", "ram_used_MiB"):
            v = r[k][m]
            vals.setdefault(k, []).append(v[np.isfinite(v)])
    keys = ("proc_cpu_pct", "sys_cpu_pct", "proc_rss_MiB", "vdd_in_mW", "vdd_cpu_gpu_cv_mW", "vdd_soc_mW", "gpu_load_pct",
            "tj_temp_C", "ram_used_MiB")
    return {k: (np.concatenate(vals[k]) if vals.get(k) else np.array([])) for k in keys}   # empty if no file (e.g. no idle run)


def fig_resources(S, root, od, warn):
    idle = res_stats(glob.glob(os.path.join(root, "idle_*", "resources_idle.csv")))
    train = res_stats(glob.glob(os.path.join(root, "folds", "*", "resources_train.csv")), "train:")
    mon = res_stats([os.path.join(m["_dir"], "resources.csv") for m in S], "stream")
    phases = [("idle", idle), ("training\n(6 sites)", train), ("real-time\nmonitor", mon)]
    summ = {}
    for name, d in phases:
        summ[name.replace("\n", " ")] = {k: (dict(mean=float(v.mean()), std=float(v.std()), max=float(v.max()), n=int(len(v))) if len(v) else None)
                                         for k, v in d.items()}
    NUM["resources"] = summ
    panels = [("vdd_in_mW", "board input power (W)", 1e-3), ("proc_cpu_pct", "process CPU (% of 1 core)", 1),
              ("proc_rss_MiB", "process RSS (MiB)", 1), ("tj_temp_C", "$T_j$ (°C)", 1)]
    panels = [p_ for p_ in panels if any(len(d[p_[0]]) for _, d in phases)]
    fig, ax = plt.subplots(1, len(panels), figsize=(W2, 2.9))
    ax = np.atleast_1d(ax)
    for a, (k, lab, sc), tag in zip(ax, panels, "abcd"):
        mu = [d[k].mean() * sc if len(d[k]) else np.nan for _, d in phases]
        sd = [d[k].std() * sc if len(d[k]) else 0 for _, d in phases]
        lo = [min(m_, d_) if np.isfinite(m_) else 0 for m_, d_ in zip(mu, sd)]
        a.bar(np.arange(3), mu, yerr=[lo, sd], capsize=4, color=[C["grey"], C["orange"], C["blue"]], width=0.65)
        a.set_xticks(np.arange(3)); a.set_xticklabels([p_[0] for p_ in phases], fontsize=plt.rcParams["font.size"] - 2)
        a.set_ylabel(lab); panel(a, f"({tag})")
    fig.tight_layout(w_pad=0.6)
    save(fig, od, "Fig_rt_resources", warn)

    rep = next((m for m in S if m["test"] == "A1" and m["scenario"] == "S02_fault_100ms"), S[0] if S else None)
    if rep is None:
        return
    r = rcsv(os.path.join(rep["_dir"], "resources.csv"))
    ks = [(k, lab) for k, lab in (("proc_cpu_pct", "CPU (%)"), ("vdd_in_mW", "VDD_IN (W)"), ("tj_temp_C", "$T_j$ (°C)")) if np.isfinite(r[k]).any()]
    fig, ax = plt.subplots(len(ks), 1, figsize=(W1, 1.35 * len(ks) + 0.6), sharex=True, squeeze=False)
    ax = ax[:, 0]
    st = np.char.startswith(r["phase"].astype(str), "stream")
    t0 = r["t_s"][st][0] if st.any() else 0
    for a, (k, lab) in zip(ax, ks):
        v = r[k] * (1e-3 if k == "vdd_in_mW" else 1)
        a.plot(r["t_s"] - t0, v, c="k", lw=1.2); a.set_ylabel(lab)
        if len(idle[k]):
            a.axhline(idle[k].mean() * (1e-3 if k == "vdd_in_mW" else 1), c=C["grey"], ls="--", lw=1.2)
    ax[-1].set_xlabel("time since stream start (s)")
    fig.align_ylabels(ax)
    fig.tight_layout(h_pad=0.3)
    save(fig, od, f"Fig_rt_resources_timeseries_{rep['test']}_{rep['scenario']}", warn)


def fig_training(root, od, warn):
    folds = sorted(glob.glob(os.path.join(root, "folds", "*", "train_log.json")))
    if not folds:
        return
    L = {os.path.basename(os.path.dirname(f)): json.load(open(f)) for f in folds}
    NUM["training"] = {f: dict(wall_s=l["train_stats"]["wall_s"], fusion_s=l["timing"]["fusion_fit_s"],
                               DC1_local_s_per_round=l["DC1_local_train_s_per_round"], peak_rss_MiB=l["peak_rss_MiB"],
                               uplink_B=l["uplink_bytes_per_client_per_round"], downlink_B=l["downlink_bytes_per_client_per_round"],
                               KiB_round=l["KiB_per_round_all_clients"], MiB_total=l["MiB_to_convergence"],
                               params=l["params_total"], shared=l["params_shared"]) for f, l in L.items()}
    names = list(L)
    fig, ax = plt.subplots(1, 3, figsize=(W2, 2.9))
    per = [[r["DC1_local_s"] for r in L[n]["train_stats"]["round_log"]] for n in names]
    ax[0].boxplot(per, patch_artist=True, boxprops=dict(facecolor=C["orange"], alpha=0.6), medianprops=dict(color="k"))
    ax[0].set_xticks(np.arange(1, len(names) + 1)); ax[0].set_xticklabels([SCEN_SHORT.get(n, n).split("\n")[0] for n in names])
    ax[0].set_ylabel("DC1 local training\nper round (s)")
    ax[1].plot(np.arange(1, len(L[names[0]]["train_stats"]["convergence"]) + 1), L[names[0]]["train_stats"]["convergence"], "o-", ms=3, c=C["blue"])
    ax[1].set_xlabel("round"); ax[1].set_ylabel("mean loss")
    l0 = L[names[0]]
    ax[2].bar([0, 1], [l0["uplink_bytes_per_client_per_round"], l0["downlink_bytes_per_client_per_round"]], color=[C["green"], C["grey"]], width=0.6)
    ax[2].set_xticks([0, 1]); ax[2].set_xticklabels(["uplink\n(int8)", "downlink\n(fp32)"]); ax[2].set_ylabel("bytes per site per round")
    for i, v in enumerate([l0["uplink_bytes_per_client_per_round"], l0["downlink_bytes_per_client_per_round"]]):
        ax[2].text(i, v, f"{v:,.0f}", ha="center", va="bottom")
    ax[2].set_ylim(0, l0["downlink_bytes_per_client_per_round"] * 1.2)
    for a, tag in zip(ax, "abc"):
        panel(a, f"({tag})")
    fig.tight_layout(w_pad=0.8)
    save(fig, od, "Fig_rt_training_communication", warn)


# ------------------------------------------------------------------------------------------ tables
def tables(S, od):
    rows = []
    for m in sorted(S, key=lambda m: (m["_fold"], m["test"])):
        tm = m["timing_ms"]
        rows.append(dict(fold=m["_fold"], scenario=m["scenario"], test=m["test"], auc=m["roc_auc"], f1=m["f1"], precision=m["precision"],
                         recall=m["recall"], fpr=m["fpr"], pos=m["windows_pos"], valid=m["windows_valid"],
                         events=len(m["attack_events"]), detected=sum(e["detected"] for e in m["attack_events"]),
                         feat_ms_mean=tm["sample_proc_mean"], window_ms_mean=tm["tcn_sample_proc_mean"], window_ms_p99=tm["tcn_sample_proc_p99"],
                         misses=tm["deadline_misses"], csec_wait_max_ms=tm["csec_wait_max"], peak_rss_MiB=m["peak_rss_MiB"],
                         stream_vs_batch=m["streaming_vs_batch_max_abs_diff"], realtime=m["realtime"]))
    if rows:
        with open(os.path.join(od, "Table_rt_all_streams.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    det = NUM.get("detection_by_regime", {})
    fmt = lambda t: "--" if not t or not t[2] or not np.isfinite(t[0]) else f"{t[0]:.3f} $\\pm$ {t[1]:.3f}"  # noqa: E731
    lines = ["\\begin{tabular}{lccccc}", "\\hline", "Attack & ROC-AUC & F1 & Precision & Recall & FPR \\\\", "\\hline"]
    for g in REG_ORDER:
        if g in det and det[g]["roc_auc"][2]:
            d = det[g]
            lines.append(f"{REG_LABEL[g].replace(chr(10), ' ')} & {fmt(d['roc_auc'])} & {fmt(d['f1'])} & {fmt(d['precision'])} & {fmt(d['recall'])} & {fmt(d['fpr'])} \\\\")
    if "no_attack" in det and det["no_attack"]["fpr"][2]:
        lines.append(f"No attack & -- & -- & -- & -- & {fmt(det['no_attack']['fpr'])} \\\\")
    lines += ["\\hline", "\\end{tabular}"]
    open(os.path.join(od, "Table_rt_detection.tex"), "w").write("\n".join(lines) + "\n")
    t = NUM.get("timing_ms", {}); r = NUM.get("resources", {})
    g = lambda d, k, s=1: "--" if not d or not d.get(k) else f"{d[k]['mean'] * s:.2f}"  # noqa: E731
    lines = ["\\begin{tabular}{lc}", "\\hline", "Quantity (measured on the board) & Value \\\\", "\\hline"]
    if t:
        lines += [f"Feature update per sample, mean / p99 (ms) & {t['sample_features']['mean']:.3f} / {t['sample_features']['p99']:.3f} \\\\",
                  f"Sample with TCN + fusion, mean / p99 (ms) & {t['sample_with_tcn_window']['mean']:.3f} / {t['sample_with_tcn_window']['p99']:.3f} \\\\",
                  f"Deadline misses (10 ms) & {t['deadline_misses']} / {t['samples']} \\\\",
                  f"Windows waiting for CSEC & {t['windows_waiting_for_csec']} / {t['windows']} \\\\"]
    for ph in ("idle", "training (6 sites)", "real-time monitor"):
        if ph in r:
            lines.append(f"{ph}: VDD\\_IN (W) / CPU (\\%) / RSS (MiB) & {g(r[ph], 'vdd_in_mW', 1e-3)} / {g(r[ph], 'proc_cpu_pct')} / {g(r[ph], 'proc_rss_MiB')} \\\\")
    lines += ["\\hline", "\\end{tabular}"]
    open(os.path.join(od, "Table_rt_edge_resources.tex"), "w").write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results_rt")
    ap.add_argument("--font", type=float, default=12)
    ap.add_argument("--allow-nonrealtime", action="store_true", help="pipeline check only; figures get a watermark")
    a = ap.parse_args()
    style(a.font)
    od = os.path.join(a.root, "paper")
    os.makedirs(od, exist_ok=True)
    S = load_streams(a.root, a.allow_nonrealtime)
    warn = any(not m["realtime"] for m in S)
    print(f"{len(S)} streams")
    for m in S:
        key = (m["_fold"], m["test"])
        if key in {("S02_fault_100ms", "A1"), ("S02_fault_100ms", "no_attack"), ("S02_fault_100ms", "T3"), ("FULL", "ctrl")}:
            if m["test"] == "ctrl" and m["scenario"] != "S07_atk_false_ups":
                continue
            fig_timeline(m, od, warn)
            ev = {"S02_fault_100ms": (40, 120), "S07_atk_false_ups": (120, 210)}.get(m["scenario"])
            if ev:
                fig_timeline(m, od, warn, xlim=ev, suffix=f"_zoom_{ev[0]}_{ev[1]}s")
    fig_detection(S, od, warn)
    fig_disturbance(S, od, warn)
    fig_delay(S, od, warn)
    fig_control(S, od, warn)
    if S:
        fig_latency(S, od, warn)
        fig_resources(S, a.root, od, warn)
    fig_training(a.root, od, warn)
    tables(S, od)
    plans = sorted(glob.glob(os.path.join(a.root, "experiment_plan*.json")))
    NUM["system"] = json.load(open(plans[0]))["system"] if plans else None
    NUM["streams_used"] = [f"{m['_fold']}/{m['test']}_{m['scenario']}" for m in S]
    json.dump(NUM, open(os.path.join(od, "numbers.json"), "w"), indent=1, default=float)
    print("numbers ->", os.path.join(od, "numbers.json"))


if __name__ == "__main__":
    main()
