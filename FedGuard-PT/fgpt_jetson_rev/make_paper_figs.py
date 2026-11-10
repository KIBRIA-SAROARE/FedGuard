#!/usr/bin/env python3
"""REV: the three result figures of the manuscript, in the manuscript's format (IEEE two-column, figure* width).

  figs/fig_tl_S02.pdf      (a) S02 no attack, (b) S02 A1                      -- Fig. timeline
  figs/fig_tl_S07_T3.pdf   (a) S07 false UPS (FULL model), (b) S02 T3         -- Fig. tl2
  figs/fig_rt_timing2.pdf  (a) per-sample processing, (b) CSEC release wait   -- Fig. timing (all DC1 streams)

Inputs: results_rev/streams/<fold>/<test>_<scenario>/{signals.npz, windows.csv, sample_timing.npz}.
Timelines start after the evaluation warm-up (nb.CFG["WARMUP_S"]); timing statistics use every sample.
  python3 make_paper_figs.py --root results_rev --out figs
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import matplotlib
import matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fgpt import nb  # noqa: E402

W2 = 7.16                       # IEEE double-column text width (in)
C = dict(P="#1f6fb4", fac="#e3650c", prov="#d46ab0", rel="#119e7d", thr="k", shade="0.82",
         b1="#1f6fb4", b2="#e3650c", b3="#119e7d")


def style():
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "Nimbus Roman", "TeX Gyre Termes", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5, "legend.frameon": False,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "axes.grid": True,
        "grid.color": "0.88", "grid.linewidth": 0.5, "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def rcsv(p):
    import pandas as pd
    return pd.read_csv(p)


def stream_dir(root, fold, test, scen):
    d = os.path.join(root, "streams", fold, f"{test}_{scen}")
    assert os.path.isdir(d), d
    return d


def spans(t, lab):
    on = np.flatnonzero(np.diff(np.r_[0, (lab == 1).astype(int), 0]))
    return [(t[a], t[b - 1]) for a, b in zip(on[::2], on[1::2])]


def timeline(axc, d, title, x0):
    sig = np.load(os.path.join(d, "signals.npz"))
    w = rcsv(os.path.join(d, "windows.csv"))
    thr = json.load(open(os.path.join(d, "metrics.json")))["threshold"] if os.path.exists(os.path.join(d, "metrics.json")) else None
    t, lab = sig["t"], sig["label"]
    k = t >= x0
    rel = w.sort_values("t_release")
    a0, a1, a2, a3 = axc
    a0.plot(t[k], sig["Va"][k], c="k", lw=0.7)
    a1.plot(t[k], sig["P_total"][k], c=C["P"], lw=0.7)
    a1.plot(t[k], sig["P_fac"][k], c=C["fac"], lw=0.7, alpha=0.9)
    we = w[w.t_end >= x0]
    a2.step(we.t_end, we.prov_score, where="post", c=C["prov"], ls=":", lw=0.9)
    rr = rel[rel.t_release >= x0]
    a2.step(rr.t_release, rr.score, where="post", c=C["rel"], lw=0.9)
    if thr is None:
        thr = float(np.nan)
    a2.axhline(thr, c=C["thr"], ls="--", lw=0.8)
    a2.set_yscale("symlog", linthresh=10.0)
    a2.set_ylim(-15, 3000)
    a2.set_yticks([-10, 0, 10, 1000]); a2.set_yticklabels(["$-10$", "0", "10", "$10^3$"])
    a2.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    a3.step(rr.t_release, rr.alarm, where="post", c="k", lw=0.8)
    a3.set_ylim(-0.15, 1.25); a3.set_yticks([0, 1]); a3.set_yticklabels(["off", "on"])
    for a in axc:
        for s0, s1 in spans(t, lab):
            a.axvspan(s0, s1, color=C["shade"], alpha=0.7, lw=0, zorder=0)
        a.set_xlim(x0, 300.0)
        a.set_xticks([50, 100, 150, 200, 250, 300])
    a0.set_title(title, loc="left", pad=2)
    a3.set_xlabel("time (s)")
    return thr


def fig_timelines(root, out, name, left, right):
    x0 = nb.CFG["WARMUP_S"]
    fig, ax = plt.subplots(4, 2, figsize=(W2, 2.78), sharex="col",
                           gridspec_kw=dict(height_ratios=[1, 1.1, 1.1, 0.6], hspace=0.12, wspace=0.16))
    info = {}
    for j, (fold, test, scen, title) in enumerate((left, right)):
        d = stream_dir(root, fold, test, scen)
        info[title] = timeline(ax[:, j], d, title, x0)
    ax[0, 0].set_ylabel("$V_a$ (pu)"); ax[1, 0].set_ylabel("$P$ (MW)")
    ax[2, 0].set_ylabel("score"); ax[3, 0].set_ylabel("alarm")
    fig.align_ylabels(ax[:, 0])
    hs = [Line2D([], [], c=C["P"], lw=1.2), Line2D([], [], c=C["fac"], lw=1.2),
          Line2D([], [], c=C["prov"], ls=":", lw=1.2), Line2D([], [], c=C["rel"], lw=1.2),
          Line2D([], [], c="k", ls="--", lw=1.0), Patch(color=C["shade"], alpha=0.7)]
    ls = ["reported $P$", r"$P_{\mathrm{it}}+P_{\mathrm{cool}}+P_{\mathrm{aux}}$", r"provisional score $\phi^0$",
          r"released score $\phi$", r"threshold $\tau^\star$", "attack interval"]
    fig.legend(hs, ls, loc="upper center", ncol=6, bbox_to_anchor=(0.5, 1.0), handlelength=1.8, columnspacing=1.1)
    fig.subplots_adjust(left=0.078, right=0.985, top=0.865, bottom=0.135)
    fig.savefig(os.path.join(out, name + ".pdf")); fig.savefig(os.path.join(out, name + ".png"), dpi=300)
    plt.close(fig)
    print("saved", name, {k: round(v, 3) for k, v in info.items()})


def fig_timing(root, out, name="fig_rt_timing2"):
    dirs = sorted(glob.glob(os.path.join(root, "streams", "*", "*", "sample_timing.npz")))
    norm, tcn, late, wait, nwin = [], [], [], [], 0
    for p in dirs:
        z = np.load(p)
        norm.append(z["proc_ms"][~z["tcn"]]); tcn.append(z["proc_ms"][z["tcn"]]); late.append(z["lateness_ms"])
        w = rcsv(os.path.join(os.path.dirname(p), "windows.csv"))
        wait.append(w["csec_wait_ms"].to_numpy()); nwin += len(w)
    norm, tcn, late, wait = (np.concatenate(v) for v in (norm, tcn, late, wait))
    ns = len(norm) + len(tcn)
    st = lambda v: [v.mean(), np.percentile(v, 50), np.percentile(v, 99), np.percentile(v, 99.9), v.max()]  # noqa: E731
    S = dict(norm=st(norm), tcn=st(tcn), late=st(late))
    wv = wait[wait > 0] / 1e3
    Wt = st(wv)
    cats = ["mean", "median", "99th", "99.9th", "max"]
    fig, ax = plt.subplots(1, 2, figsize=(W2, 1.85), gridspec_kw=dict(width_ratios=[2.6, 1], wspace=0.22))
    x = np.arange(5); bw = 0.26
    for i, (key, lab, col) in enumerate((("norm", "channel update only", C["b1"]), ("tcn", "with TCN and fusion", C["b2"]),
                                         ("late", "start lateness", C["b3"]))):
        v = S[key]
        ax[0].bar(x + (i - 1) * bw, v, bw, color=col, label=lab, zorder=3)
        for xi, vi in zip(x + (i - 1) * bw, v):
            ax[0].text(xi, vi * 1.15, f"{vi:.2f}" if vi < 10 else f"{vi:.1f}", ha="center", va="bottom", fontsize=6.5, zorder=5,
                       bbox=dict(boxstyle="square,pad=0.05", fc="white", ec="none"))
    ax[0].axhline(10.0, c="k", ls="--", lw=0.8, zorder=2)
    ax[0].text(-0.45, 11.5, "10-ms sample period", fontsize=7, va="bottom")
    ax[0].set_yscale("log"); ax[0].set_ylim(0.03, 400)
    ax[0].set_xticks(x); ax[0].set_xticklabels(cats); ax[0].set_ylabel("time per sample (ms)")
    ax[0].legend(loc="upper left", ncol=3, bbox_to_anchor=(0.0, 1.0), handlelength=1.2, columnspacing=1.0)
    ax[0].set_title(f"(a) per-sample processing over {ns:,} samples", loc="left", pad=2)
    ax[1].bar(x, Wt, 0.6, color=C["b3"], zorder=3)
    for xi, vi in zip(x, Wt):
        ax[1].text(xi, vi * 1.15, f"{vi:.2f}" if vi < 10 else f"{vi:.1f}", ha="center", va="bottom", fontsize=6.5)
    ax[1].set_yscale("log"); ax[1].set_ylim(0.5, 150)
    ax[1].set_xticks(x); ax[1].set_xticklabels(cats); ax[1].set_ylabel("release wait (s)")
    ax[1].set_title("(b) CSEC release wait", loc="left", pad=2)
    ax[1].text(0.03, 0.96, f"{len(wv):,} of {nwin:,}\nwindows wait", transform=ax[1].transAxes, ha="left", va="top", fontsize=7)
    for a in ax:
        a.grid(axis="x", visible=False)
    fig.subplots_adjust(left=0.07, right=0.995, top=0.88, bottom=0.14)
    fig.savefig(os.path.join(out, name + ".pdf")); fig.savefig(os.path.join(out, name + ".png"), dpi=300)
    plt.close(fig)
    res = dict(streams=len(dirs), samples=ns, windows=nwin, waiting=int(len(wv)),
               channel_ms=S["norm"], tcn_ms=S["tcn"], lateness_ms=S["late"], wait_s=Wt)
    print("saved", name, json.dumps({k: (np.round(v, 3).tolist() if isinstance(v, list) else v) for k, v in res.items()}))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results_rev")
    ap.add_argument("--out", default="figs")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    style()
    s02 = "S02_fault_100ms"
    fig_timelines(a.root, a.out, "fig_tl_S02",
                  (s02, "no_attack", s02, "(a) S02, 100-ms fault at 60 s, no attack"),
                  (s02, "A1", s02, "(b) S02, attack A1 (scaling)"))
    fig_timelines(a.root, a.out, "fig_tl_S07_T3",
                  ("FULL", "ctrl", "S07_atk_false_ups", "(a) S07, false UPS transfer at 140–170 s (FULL model)"),
                  (s02, "T3", s02, "(b) S02, coordinated attack T3"))
    fig_timing(a.root, a.out)


if __name__ == "__main__":
    main()
