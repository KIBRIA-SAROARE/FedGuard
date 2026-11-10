#!/usr/bin/env python3
"""Real-time FedGuard-PT monitor for one site: one test stream, paced at 100 Hz wall clock.

Per sample (every 10 ms): wait for the sample's scheduled arrival -> update the 10 features.
Per window (every 40 samples): TCN errors + term z-scores when the forecast target arrives.
The fused score is released when the window's CSEC values are causally known (fgpt/csec_rt.py).

Tests:  no_attack | T1 | T2 | T3a | T3 | U1 (false UPS-state report) | A1..A5 (repository attacked file)
        | ctrl (control-path scenario) | ctrl_hide (control-path transfer with the UPS state reported as 0, U2)
Outputs in --out: metrics.json, windows.csv, sample_timing.npz, signals.npz, resources.csv, tegrastats.log
"""
import argparse
import collections
import csv
import os
import resource
import time

import numpy as np

from fgpt import common, csec_rt, nb, stream, tcn_np
from fgpt.resmon import ResourceMonitor

KEYS = ["Va_pu", "P_meas_MW", "P_total_MW", "Q_total_Mvar", "P_it_MW", "P_cool_MW", "P_aux_MW", "ups_state", "P_ai_pu"]


def load_bundle(fold_dir, k):
    b = os.path.join(fold_dir, "bundle")
    fusion = common.load_config(os.path.join(b, "fusion.json"))
    nb.LEGACY_UPS = bool(fusion.get("ups", {}).get("legacy", True))   # REV: use the UPS check the bundle was trained with
    CAL = {j: nb.cal_from_json(common.load_config(os.path.join(b, f"calib_DC{j}.json"))) for j in range(1, 7)}
    p = dict(np.load(os.path.join(b, f"DC{k}.npz")))
    return CAL, p, fusion


def attack_log_rows(dataset, key, atype):
    out = []
    with open(os.path.join(dataset, "attack_log.csv")) as f:
        f.readline()
        for line in f:
            q = line.strip().split(",")
            if q[0] == key and int(q[1]) == atype:
                out.append((float(q[2]), float(q[3])))
    return sorted(out)


def build_test(test, scen, k, CAL, dataset):
    """-> site telemetry, per-sample labels (1 attack, 0 clean, -1 recovery), all-site telemetry, attack windows."""
    if test.startswith("A") and test[1:].isdigit():
        a = int(test[1:])
        rec, extra = nb.load_record(os.path.join(dataset, "attacked", f"{scen}_DC{k}_A{a}.csv"))
        tel = nb.telemetry_from_record(rec)
        tels = {j: nb.telemetry(scen, j) for j in range(1, 7)}
        tels[k] = tel
        return tel, extra["label"].astype(np.int8), tels, attack_log_rows(dataset, f"{scen}_DC{k}", a)
    if test in ("no_attack", "ctrl", "ctrl_hide"):
        tels = {j: nb.telemetry(scen, j) for j in range(1, 7)}
        lab = nb.control_labels(scen, k) if test != "no_attack" else np.zeros(len(nb.tvec()), np.int8)
        spec = nb.CONTROL_PATH.get(scen)
        wins = [(spec["t0"], spec["t1"])] if (test != "no_attack" and spec and k in spec["targets"]) else []
        if test == "ctrl_hide":
            tels[k] = nb.hide_transfer(tels[k], scen, k)
        return tels[k], lab, tels, wins
    if test == "U1":       # state spoof at the monitored site only; peers keep the clean scenario
        tels = {j: nb.telemetry(scen, j) for j in range(1, 7)}
        tels[k], lab, log = nb.make_attacked(scen, k, "U1", nb.notebook_attack_seed(nb.CFG["ATK_SEED_TEST"], scen, k), CAL)
        return tels[k], lab, tels, [(e["t0"], e["t1"]) for e in log]
    tels, labs, logs = {}, {}, {}
    for j in range(1, 7):
        tels[j], labs[j], logs[j] = nb.make_attacked(scen, j, test, nb.notebook_attack_seed(nb.CFG["ATK_SEED_TEST"], scen, j), CAL)
    return tels[k], labs[k], tels, [(e["t0"], e["t1"]) for e in logs[k]]


class RtScorer:
    """Window terms at target arrival; fused score at CSEC release."""

    def __init__(self, p, fusion, corr, unc, avail, k=1):
        self.p, self.corr, self.unc, self.avail = p, corr, unc, avail
        self.W, self.H = nb.CFG["WIN"], nb.CFG["HOP"]
        self.buf = collections.deque(maxlen=self.W + 1)
        self.coef = np.asarray(fusion["coef"], np.float64)
        self.b = float(fusion["intercept"])
        self.ms = nb.fusion_ms(fusion, k)
        self.kappa, self.eta, self.thr = fusion["kappa"], fusion["eta"], fusion["threshold"]
        self.rule = fusion.get("physics_rule")      # REV: physics layer (absent in older bundles -> off)
        self.pending = []
        self.n = 0

    def push(self, x):
        """Call once per sample. Returns (list of released window dicts, tcn_ran)."""
        self.buf.append(x)
        i = self.n
        self.n += 1
        ran = False
        s = i - self.W
        if s >= 0 and s % self.H == 0:
            arr = np.asarray(self.buf)
            win, nxt = arr[:self.W], arr[self.W]
            res = np.abs(win[:, nb.RESID_IDX]).astype(np.float64).sum(0) / self.W
            e_rec, e_fc = tcn_np.errors(self.p, nb.tcn_in(win[None]), nxt[None, nb.SIG_IDX], nxt[None, nb.RESID_IDX])
            Z = nb.z_terms(np.r_[res, e_rec, e_fc].astype(np.float32)[None], self.ms)[0]
            rel = max(i, int(self.avail[s:s + self.W].max()))
            fired = bool(nb.physics_rule(Z, self.rule)[0])
            boost = self.rule["boost"] if fired else 0.0
            prov = float(Z @ self.coef + self.b) + boost   # provisional: CSEC terms not yet known (c = u = 0)
            self.pending.append(dict(start=s, target=i, release=rel, Z=Z, prov_score=prov, prov_alarm=prov > self.thr,
                                     rule=fired, boost=boost))
            ran = True
        out = []
        keep = []
        for w in self.pending:
            if w["release"] <= i:
                s = w["start"]
                c = float(self.corr[s:s + self.W].astype(np.float64).sum() / self.W)
                u = float(self.unc[s:s + self.W].astype(np.float64).sum() / self.W)
                w["score"] = float(w["Z"] @ self.coef + self.b - self.kappa * c + self.eta * u) + w["boost"]
                w["alarm"] = w["score"] > self.thr
                dom, cs, _ = nb.explain(w["Z"][None], self.coef)
                w.update(dominant="UPS rule" if w["rule"] else dom[0], domain=cs[0], csec_corr=c, csec_unc=u)
                out.append(w)
            else:
                keep.append(w)
        self.pending = keep
        return out, ran


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data/dataset")
    ap.add_argument("--fold-dir", required=True, help="folder containing bundle/ from train_edge.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dc", type=int, default=1)
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--no-realtime", action="store_true", help="replay as fast as possible (not for the paper)")
    ap.add_argument("--net-delay-ms", type=float, default=0.0, help="assumed CSEC descriptor transport delay")
    ap.add_argument("--resource-interval", type=float, default=0.5)
    a = ap.parse_args()
    nb.DATASET_DIR = a.dataset
    os.makedirs(a.out, exist_ok=True)
    k, dt = a.dc, nb.CFG["DT"]
    CAL, p, fusion = load_bundle(a.fold_dir, k)

    t0 = time.perf_counter()
    tel, lab, tels, wins = build_test(a.test, a.scenario, k, CAL, a.dataset)
    corr, unc, avail, clog, n_events, clipped = csec_rt.csec_with_release(tels, CAL, k, int(round(a.net_delay_ms / 1e3 / dt)),
                                                                           return_clipped=True)
    t_prep = time.perf_counter() - t0
    T = nb.tvec()
    N = len(T)
    cols = [tel[c] for c in KEYS]

    mon = ResourceMonitor(os.path.join(a.out, "resources.csv"), os.path.join(a.out, "tegrastats.log"), a.resource_interval).start()
    mon.set_phase("warmup")
    sf = stream.StreamingFeatures(k, CAL[k])
    sc = RtScorer(p, fusion, corr, unc, avail, k)
    for _ in range(20):                                  # warm the code paths, then reset state
        tcn_np.errors(p, np.zeros((1, nb.CFG["WIN"], nb.N_CH), np.float32), np.zeros((1, 3), np.float32), np.zeros((1, 7), np.float32))
    time.sleep(2.0)
    mon.set_phase("stream")

    lateness = np.zeros(N, np.float32)     # start of processing - scheduled arrival (ms)
    proc = np.zeros(N, np.float32)         # processing time of this sample incl. any window work (ms)
    tcn_flag = np.zeros(N, bool)
    missed = np.zeros(N, bool)
    released = []
    base = time.perf_counter() + 0.1
    for i in range(N):
        sched = base + i * dt
        if not a.no_realtime:
            d = sched - time.perf_counter()
            if d > 0:
                time.sleep(d)
        st = time.perf_counter()
        x = sf.step(*(float(c[i]) for c in cols))
        out, ran = sc.push(x)
        en = time.perf_counter()
        for w in out:
            w["emit_wall_ms_after_target"] = (en - (base + w["target"] * dt)) * 1e3
            w["release_sim_s"] = float(T[w["release"]])
            released.append(w)
        lateness[i] = (st - sched) * 1e3
        proc[i] = (en - st) * 1e3
        tcn_flag[i] = ran
        missed[i] = en > sched + dt
    wall = time.perf_counter() - base
    mon.set_phase("post")
    mon.stop()

    released.sort(key=lambda w: w["start"])
    starts = np.array([w["start"] for w in released])
    score = np.array([w["score"] for w in released])
    alarm = np.array([w["alarm"] for w in released])
    ywin = nb.window_labels(lab, starts)
    valid = nb.eval_mask(ywin, T[starts + nb.CFG["WIN"] - 1])    # REV: labelled windows after the warm-up
    y = (ywin[valid] == 1).astype(int)
    thr = fusion["threshold"]

    # offline notebook pipeline on the same record (not timed): must reproduce the released scores
    from train_edge import window_terms
    X = nb.features(tel, k, CAL[k])[0]
    ws = nb.winset(X, lab, k, CAL[k])
    Zb = nb.z_terms(window_terms(p, ws, np.arange(len(ws.starts))), sc.ms)
    sb = nb.fused_score(fusion["coef"], fusion["intercept"], Zb, nb.csec_window(corr, ws.starts),
                        nb.csec_window(unc, ws.starts), fusion["kappa"], fusion["eta"], fusion.get("physics_rule"))
    diff = float(np.abs(sb[:len(score)] - score).max()) if len(sb) >= len(score) else float("nan")

    t_end = T[starts + nb.CFG["WIN"] - 1]
    met = dict(dc=k, scenario=a.scenario, test=a.test, fold_dir=os.path.abspath(a.fold_dir),
               validation_scenario=fusion.get("validation_scenario"), perturb=nb.PERTURB,
               realtime=not a.no_realtime, net_delay_ms=a.net_delay_ms, samples=N, windows=int(len(released)),
               windows_valid=int(valid.sum()), windows_pos=int(y.sum()), threshold=thr,
               streaming_vs_batch_max_abs_diff=diff, prep_s=t_prep, wall_s=wall, warmup_s=nb.CFG["WARMUP_S"])
    met.update(nb.metrics_from(score[valid], y, thr))
    rf = np.array([w["rule"] for w in released])
    met["physics_rule"] = dict(enabled=bool((fusion.get("physics_rule") or {}).get("enabled")), fired_windows=int(rf.sum()),
                               fired_valid_neg=int((rf & valid & (ywin == 0)).sum()), fired_pos=int((rf & valid & (ywin == 1)).sum()))
    # REV (A9): windows whose CSEC release was clipped to the end of the record
    W_ = nb.CFG["WIN"]
    clip_w = np.array([bool(clipped[st_:st_ + W_].any()) for st_ in starts])
    met["clipped_release"] = dict(windows=int(clip_w.sum()), windows_valid=int((clip_w & valid).sum()),
                                  windows_pos=int((clip_w & valid & (ywin == 1)).sum()))
    keep_ = valid & ~clip_w
    met["excluding_clipped"] = nb.metrics_from(score[keep_], (ywin[keep_] == 1).astype(int), thr)
    met["delay_s_notebook"] = nb.detection_delay(score[valid], y, thr, t_end[valid])
    ev = []
    rel_sim = np.array([w["release_sim_s"] for w in released])
    t_start = T[starts]
    for (w0, w1) in wins:
        m = alarm & (t_end >= w0) & (t_start <= w1)
        idx = np.flatnonzero(m)
        ev.append(dict(t0=w0, t1=w1, detected=bool(len(idx)),
                       delay_s=float(rel_sim[idx].min() - w0) if len(idx) else None))
    prov_alarm = np.array([w["prov_alarm"] for w in released])
    tgt_sim = T[np.array([w["target"] for w in released])]
    for e_, (w0, w1) in zip(ev, wins):
        m = prov_alarm & (t_end >= w0) & (t_start <= w1)
        idx = np.flatnonzero(m)
        e_["provisional_detected"] = bool(len(idx))
        e_["provisional_delay_s"] = float(tgt_sim[idx].min() - w0) if len(idx) else None
    met["attack_events"] = ev
    prov_score = np.array([w["prov_score"] for w in released])
    met["provisional_no_csec"] = nb.metrics_from(prov_score[valid], y, thr)
    fault_t = {"S02_fault_100ms": 60.0, "S03_fault_250ms": 60.0, "S06_multi_event": 60.0, "S11_fault_plus_atk": 60.0,
               "S04_line_trip": 90.0, "S05_load_step": 120.0}.get(a.scenario)
    if fault_t is not None:
        fw = valid & (t_end >= fault_t) & (t_start <= fault_t + 10.0) & (ywin == 0)
        met["disturbance_window_10s"] = dict(event_time_s=fault_t, neg_windows=int(fw.sum()), false_alarms=int(alarm[fw].sum()))
    if alarm.any():
        d = np.array([w["dominant"] for w in released])[alarm]
        met["dominant_term_share_in_alarms"] = {t: float((d == t).mean()) for t in nb.TERMS}
    lat_w = np.array([w["emit_wall_ms_after_target"] for w in released])
    wait = (np.array([w["release"] for w in released]) - np.array([w["target"] for w in released])) * dt * 1e3
    met["timing_ms"] = dict(
        sample_proc_mean=float(proc[~tcn_flag].mean()), sample_proc_p99=float(np.percentile(proc[~tcn_flag], 99)),
        sample_proc_max=float(proc[~tcn_flag].max()),
        tcn_sample_proc_mean=float(proc[tcn_flag].mean()), tcn_sample_proc_p99=float(np.percentile(proc[tcn_flag], 99)),
        tcn_sample_proc_max=float(proc[tcn_flag].max()),
        lateness_mean=float(lateness.mean()), lateness_p99=float(np.percentile(lateness, 99)), lateness_max=float(lateness.max()),
        deadline_misses=int(missed.sum()), deadline_ms=dt * 1e3,
        decision_after_target_mean=float(lat_w.mean()), decision_after_target_p99=float(np.percentile(lat_w, 99)),
        decision_after_target_max=float(lat_w.max()),
        csec_wait_windows=int((wait > 0).sum()), csec_wait_max=float(wait.max()), csec_wait_mean_when_waiting=float(wait[wait > 0].mean()) if (wait > 0).any() else 0.0,
        decision_period_ms=nb.CFG["HOP"] * dt * 1e3, window_ms=nb.CFG["WIN"] * dt * 1e3)
    met["csec"] = dict(site_events=n_events, uplink_bytes=n_events * nb.DESCRIPTOR_BYTES,
                       fleet_events=int(len(clog)), fleet_bytes=int(len(clog)) * nb.DESCRIPTOR_BYTES)
    met["peak_rss_MiB"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    common.save_json(met, os.path.join(a.out, "metrics.json"))

    with open(os.path.join(a.out, "windows.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["t_start", "t_end", "t_release", "label", "score", "alarm", "dominant", "domain", "csec_corr",
                     "csec_unc", "decision_ms_after_target", "csec_wait_ms", "prov_score", "prov_alarm", "physics_rule"] + nb.TERMS)
        for w, yl, wt in zip(released, ywin, wait):
            wr.writerow([f"{T[w['start']]:.2f}", f"{T[w['start'] + nb.CFG['WIN'] - 1]:.2f}", f"{w['release_sim_s']:.2f}", int(yl),
                         f"{w['score']:.5f}", int(w["alarm"]), w["dominant"], w["domain"], f"{w['csec_corr']:.4f}",
                         f"{w['csec_unc']:.4f}", f"{w['emit_wall_ms_after_target']:.3f}", f"{wt:.0f}", f"{w['prov_score']:.5f}", int(w["prov_alarm"]), int(w["rule"])] + [f"{v:.4g}" for v in w["Z"]])
    np.savez_compressed(os.path.join(a.out, "sample_timing.npz"), proc_ms=proc, lateness_ms=lateness, tcn=tcn_flag, missed=missed)
    np.savez_compressed(os.path.join(a.out, "signals.npz"), t=T, label=lab, Va=tel["Va_pu"], P_total=tel["P_total_MW"],
                        P_fac=tel["P_it_MW"] + tel["P_cool_MW"] + tel["P_aux_MW"], P_it=tel["P_it_MW"], P_cool=tel["P_cool_MW"],
                        Q=tel["Q_total_Mvar"], corr=corr, unc=unc, avail=avail)
    tm = met["timing_ms"]
    print(f"[{a.test} {a.scenario} DC{k}] AUC={met['roc_auc']:.4f} F1={met['f1']:.4f} FPR={met['fpr']:.4f} | "
          f"sample {tm['sample_proc_mean']:.3f} ms, window sample {tm['tcn_sample_proc_mean']:.3f} ms (p99 {tm['tcn_sample_proc_p99']:.3f}), "
          f"misses {tm['deadline_misses']}/{N}, lateness p99 {tm['lateness_p99']:.3f} ms | wall {wall:.1f} s | stream-batch {diff:.1e}")


if __name__ == "__main__":
    main()
