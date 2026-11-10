#!/usr/bin/env python3
"""Reproduce the notebook's FedGuard-PT training for the LOSO fold that holds out a test scenario
(default S02_fault_100ms), on the Jetson, and export the deployment bundle for one site.

Notebook run_fold() steps, for method "FedGuard-PT":
  1. per-client calibration on S01 (all six clients)                 section 6.1
  2. clean training windows from the attack-free scenarios minus test   section 9
  3. partial federation, 25 rounds, int8 + error feedback, trimmed mean section 11
     (six clients emulated in one process; each client's local time and uplink are measured)
  4. term statistics on S01, fusion on the S01 validation mixture (T1/T2/T3a/T3, seed 7)
     with CSEC gain (kappa, eta) and threshold by grid search          section 13

REVISION changes (each can be switched off for an ablation with the flags below):
  - inner validation: the fold's validation scenario nb.CFG["VAL_SCENARIO"][fold] (a disturbance
    scenario, never the held-out one) is removed from training and used for fusion, gains and threshold;
  - term statistics per site with the site's own model (was: DC1's model for all six sites, pooled);
  - auxiliary power class on the raw P/s_k scale (nb.winset);
  - widened CSEC grids nb.CFG["KAPPA_GRID"], nb.CFG["ETA_GRID"].
Outputs: results/bundle/  (DC<k>.npz, calib_DC<k>.json, fusion.json), results/train_log.json
"""
import argparse
import json
import os
import resource
import time

import numpy as np

from fgpt import common, nb, tcn_np
from fgpt.resmon import ResourceMonitor


def window_terms(p, ws, idx, bs=1024):
    W = nb.CFG["WIN"]
    st = ws.starts[idx]
    ab = np.abs(ws.X[:, nb.RESID_IDX]).astype(np.float64)
    c = np.vstack([np.zeros((1, ab.shape[1])), np.cumsum(ab, 0)])
    res = (c[st + W] - c[st]) / W
    er, ef = [], []
    for s in range(0, len(idx), bs):
        xb, yfc, yres, _ = ws.arrays(idx[s:s + bs])
        a, b = tcn_np.errors(p, xb, yfc, yres)
        er.append(a); ef.append(b)
    return np.column_stack([res, np.concatenate(er), np.concatenate(ef)]).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data/dataset")
    ap.add_argument("--results", default="results")
    ap.add_argument("--test-scenario", default="S02_fault_100ms")
    ap.add_argument("--mode", default="partial", choices=["partial", "local", "central"],
                    help="partial = FedGuard-PT (notebook default); local = Local-only baseline")
    ap.add_argument("--resource-interval", type=float, default=0.5)
    ap.add_argument("--legacy-validation", action="store_true", help="ablation: validate on S01 (original)")
    ap.add_argument("--legacy-aux", action="store_true", help="ablation: original scale-mismatched auxiliary class")
    ap.add_argument("--legacy-pooled-ms", action="store_true", help="ablation: original pooled term statistics with DC1's model")
    ap.add_argument("--refit-fusion", action="store_true",
                    help="reuse the fold's trained models (bundle/DC*.npz); refit scoring weights, gains and threshold only")
    ap.add_argument("--legacy-ups", action="store_true",
                    help="ablation: original UPS check 1{m!=0}P_it with S01 statistics (also set by FGPT_LEGACY_UPS=1)")
    ap.add_argument("--legacy-fusion", action="store_true",
                    help="ablation: unconstrained sklearn fusion (also set by FGPT_LEGACY_FUSION=1)")
    ap.add_argument("--no-ups-rule", action="store_true", help="ablation: no physics rule on the UPS check")
    ap.add_argument("--legacy-grid", action="store_true", help="ablation: original CSEC grids {0,1,2,4,8} x {0,0.5,1,2,4}")
    args = ap.parse_args()
    nb.DATASET_DIR = args.dataset
    if args.legacy_ups:
        nb.LEGACY_UPS = True
    if args.legacy_fusion:
        nb.FUSION_NONNEG = False
    if args.legacy_grid:
        nb.CFG["KAPPA_GRID"], nb.CFG["ETA_GRID"] = [0.0, 1.0, 2.0, 4.0, 8.0], [0.0, 0.5, 1.0, 2.0, 4.0]
    out = os.path.join(args.results, "bundle")
    os.makedirs(out, exist_ok=True)
    mon = ResourceMonitor(os.path.join(args.results, "resources_train.csv"),
                          os.path.join(args.results, "tegrastats_train.log"), args.resource_interval).start()
    T = {}
    clients = list(range(1, 7))
    val_scen = nb.CFG["CALIB_SCENARIO"] if args.legacy_validation else nb.CFG["VAL_SCENARIO"][args.test_scenario]
    if args.test_scenario != "FULL" and val_scen == args.test_scenario:
        raise SystemExit("validation scenario must differ from the held-out scenario")
    # "FULL" = all attack-free scenarios except the validation one (control-path S07-S11 tests)
    train_scens = [s for s in nb.CFG["ATTACK_FREE"] if s != args.test_scenario
                   and (args.legacy_validation or s != val_scen)]
    print(f"[fold] {args.test_scenario}  train={train_scens}  validation={val_scen}  mode={args.mode}")

    mk = (lambda X, lab, k, cal: nb.WinSet(X, lab, k, cal["reg_edges"])) if args.legacy_aux else nb.winset

    # REV: evidence for the UPS gate G: longest grid-power transient after a genuine reported transfer
    # in this fold's training and validation scenarios (never the held-out one)
    longest, n_xfer, xfer_onset_margin = 0.0, 0, []
    for s_ in sorted(set(train_scens + [val_scen])):
        for k in clients:
            tel_ = nb.telemetry(s_, k)
            m_ = tel_["ups_state"] != 0
            for i0 in np.flatnonzero(np.diff(m_.astype(int)) == 1) + 1:
                i1 = i0
                while i1 < len(m_) and m_[i1]:
                    i1 += 1
                nz = np.flatnonzero(tel_["P_it_MW"][i0:i1] > 0)
                longest = max(longest, (nz[-1] + 1) * nb.CFG["DT"] if len(nz) else 0.0)
                n_xfer += 1
                nw = int(round(nb.UPS_VWIN / nb.CFG["DT"]))
                xfer_onset_margin.append(float(tel_["Va_pu"][max(0, i0 - nw):i0 + 1].min()) - nb.VTRIP[k - 1])
    ups_evidence = dict(transfers=n_xfer, longest_transient_s=longest, gate_s=nb.UPS_G,
                        max_onset_margin_pu=max(xfer_onset_margin) if xfer_onset_margin else None, dv_pu=nb.UPS_DV)
    print(f"[ups] {n_xfer} genuine transfers in training+validation; longest transient {longest:.2f} s (G = {nb.UPS_G} s); "
          f"highest onset voltage above Vtrip {ups_evidence['max_onset_margin_pu']} pu (dV = {nb.UPS_DV})")
    if not nb.LEGACY_UPS and n_xfer and (longest >= nb.UPS_G or ups_evidence["max_onset_margin_pu"] >= nb.UPS_DV):
        raise SystemExit("UPS check design values do not cover the genuine transfers in this fold; revise UPS_G / UPS_DV")

    mon.set_phase("calibration")
    t0 = time.perf_counter()
    CAL = {k: nb.calibrate_client(k) for k in clients}
    T["calibration_6_clients_s"] = time.perf_counter() - t0
    for k in clients:
        common.save_json(nb.cal_to_json(CAL[k]), os.path.join(out, f"calib_DC{k}.json"))
    print(f"[calib] {T['calibration_6_clients_s']:.1f} s  s_k(DC1)={CAL[1]['s_k']:.2f} MW")

    mon.set_phase("features")
    t0 = time.perf_counter()
    train_sets = {k: [] for k in clients}
    for s in train_scens:
        for k in clients:
            X = nb.features(nb.telemetry(s, k), k, CAL[k])[0]
            train_sets[k].append(mk(X, nb.control_labels(s, k), k, CAL[k]))
    T["train_features_s"] = time.perf_counter() - t0
    aux_share = np.bincount(np.concatenate([w.reg for w in train_sets[1]]), minlength=3)
    print(f"[features] {T['train_features_s']:.1f} s  DC1 auxiliary class share {np.round(aux_share / aux_share.sum(), 3)}")

    if args.refit_fusion:       # keep the trained TCNs of this fold; redo only scoring fit, gains and threshold
        models = {k: dict(np.load(os.path.join(out, f"DC{k}.npz"))) for k in clients}
        print("[train] skipped (--refit-fusion): loaded the fold's trained models")
    else:
        mon.set_phase("federated_training")
        models, st = tcn_np.federated_train(train_sets, args.mode, nb.CFG["SEED"], on_phase=mon.set_phase)
        print(f"[train] wall {st['wall_s']:.1f} s, local time per client "
              f"{ {k: round(v, 1) for k, v in st['local_time_per_client_s'].items()} }")

    mon.set_phase("fusion_fit")
    t0 = time.perf_counter()
    cal_sets = {k: mk(nb.features(nb.telemetry(nb.CFG["CALIB_SCENARIO"], k), k, CAL[k])[0],
                      nb.control_labels(nb.CFG["CALIB_SCENARIO"], k), k, CAL[k]) for k in clients}

    def _ms(Tc):
        mu = np.median(Tc, 0)
        sd = 1.4826 * np.median(np.abs(Tc - mu), 0)
        sd = np.where(sd <= 0, np.std(Tc, 0) + 1e-9, sd)
        return mu.astype(np.float32), sd.astype(np.float32)

    i_ups = nb.TERMS.index("|r_ups|")

    def _fix_ups(ms_):
        """REV: the UPS-check term is exactly 0 on clean calibration windows, so its statistical scale is 0;
        it is already in units of UPS_SCALE, so keep it as is (median 0, scale 1)."""
        if nb.LEGACY_UPS:
            return ms_
        mu_, sd_ = ms_[0].copy(), ms_[1].copy()
        mu_[i_ups], sd_[i_ups] = 0.0, 1.0
        return mu_, sd_

    if args.legacy_pooled_ms:       # original: DC1's model for every site, statistics pooled
        Tc = np.concatenate([window_terms(models[clients[0]], ws, np.flatnonzero(ws.y == 0)) for ws in cal_sets.values()])
        ms_pool = _ms(Tc)
        MS = {k: ms_pool for k in clients}
    else:                           # revision: each site's own model and own clean calibration windows
        MS = {k: _ms(window_terms(models[k], cal_sets[k], np.flatnonzero(cal_sets[k].y == 0))) for k in clients}
    MS = {k: _fix_ups(v) for k, v in MS.items()}
    Zv, yv, cv, uv, kv = [], [], [], [], []
    for reg in nb.REGIMES:
        tels, labs = {}, {}
        for k in clients:
            tels[k], labs[k], _ = nb.make_attacked(val_scen, k, reg, nb.notebook_attack_seed(nb.CFG["ATK_SEED_VAL"], val_scen, k), CAL)
        corr, unc, _ = nb.csec_signals(tels, CAL)
        for k in clients:
            ws = mk(nb.features(tels[k], k, CAL[k])[0], labs[k], k, CAL[k])
            idx = np.flatnonzero(ws.y >= 0)
            Zv.append(nb.z_terms(window_terms(models[k], ws, idx), MS[k]))
            yv.append((ws.y[idx] == 1).astype(int))
            cv.append(nb.csec_window(corr[k], ws.starts[idx]))
            uv.append(nb.csec_window(unc[k], ws.starts[idx]))
            kv.append(np.full(len(idx), k))
    Zv = np.concatenate(Zv)
    yv, cv, uv = np.concatenate(yv), np.concatenate(cv), np.concatenate(uv)
    lr = nb.fit_fusion(Zv, yv)
    coef, b0 = lr.coef_.ravel(), float(lr.intercept_[0])
    rule = nb.default_rule(enabled=not (args.no_ups_rule or nb.LEGACY_UPS))
    kap, eta, f1v, thr = nb.tune_csec(coef, b0, Zv, yv, cv, uv, rule)
    rule_val = dict(fired_windows=int(nb.physics_rule(Zv, rule).sum()),
                    fired_on_negatives=int((nb.physics_rule(Zv, rule) & (yv == 0)).sum()))
    print(f"[rule] physics rule {'on' if rule['enabled'] else 'off'}: fired on {rule_val['fired_windows']} validation windows, "
          f"{rule_val['fired_on_negatives']} of them negative")
    T["fusion_fit_s"] = time.perf_counter() - t0
    at_edge = dict(kappa=kap == max(nb.CFG["KAPPA_GRID"]), eta=eta == max(nb.CFG["ETA_GRID"]))
    fusion = dict(coef=coef.tolist(), intercept=b0, ms_mu=MS[1][0].tolist(), ms_sd=MS[1][1].tolist(),
                  ms_by_dc={str(k): dict(mu=MS[k][0].tolist(), sd=MS[k][1].tolist()) for k in clients},
                  kappa=kap, eta=eta, threshold=thr, val_f1=f1v, val_windows=int(len(yv)),
                  val_pos=int(yv.sum()), val_csec_active_windows=int(((cv > 0) | (uv > 0)).sum()),
                  terms=nb.TERMS, test_scenario=args.test_scenario, mode=args.mode,
                  validation_scenario=val_scen, train_scenarios=train_scens,
                  kappa_grid=nb.CFG["KAPPA_GRID"], eta_grid=nb.CFG["ETA_GRID"], gain_at_grid_edge=at_edge,
                  aux_class_share_DC1=(aux_share / aux_share.sum()).tolist(),
                  legacy=dict(validation=args.legacy_validation, aux=args.legacy_aux, pooled_ms=args.legacy_pooled_ms,
                              ups=nb.LEGACY_UPS, fusion=not nb.FUSION_NONNEG,
                              grid=args.legacy_grid, t3=os.environ.get("FGPT_LEGACY_T3") == "1"),
                  perturb=nb.PERTURB, k_v=nb.CFG["K_V"], ups_evidence=ups_evidence,
                  tcn_input="raw" if nb.TCN_INPUT is not None else "all",
                  ups=dict(legacy=nb.LEGACY_UPS, G=nb.UPS_G, dV=nb.UPS_DV, vwin=nb.UPS_VWIN, scale=nb.UPS_SCALE, vtrip=nb.VTRIP),
                  fusion_nonneg=nb.FUSION_NONNEG, check_terms=nb.CHECK_TERMS, bound_terms=nb.BOUND_TERMS,
                  physics_rule=rule, physics_rule_validation=rule_val, grid_f1=list(nb.LAST_GRID))
    common.save_json(fusion, os.path.join(out, "fusion.json"))
    print(f"[fusion] {T['fusion_fit_s']:.1f} s  val={val_scen} windows={len(yv)} pos={int(yv.sum())}  F1={f1v:.4f}  "
          f"kappa={kap} eta={eta} thr={thr:.4f}  edge={at_edge}")

    if args.refit_fusion:
        mon.stop()
        common.save_json(dict(T, refit_fusion=True), os.path.join(args.results, "refit_log.json"))
        return
    for k in clients:
        np.savez(os.path.join(out, f"DC{k}.npz"), **models[k])
    msg = st.pop("uplink_message_example", None)
    if msg:
        with open(os.path.join(out, "uplink_message_example.bin"), "wb") as f:
            f.write(msg)
    mon.stop()
    p1 = models[1]
    skeys = tcn_np.shared_keys(p1)
    rounds = max(st["rounds"], 1)
    log = dict(
        system=common.system_info(), timing=T, train_stats=st,
        params_total=tcn_np.n_params(p1), params_shared=int(sum(p1[n].size for n in skeys)),
        npz_bytes_DC1=os.path.getsize(os.path.join(out, "DC1.npz")),
        uplink_bytes_per_client_per_round=(st["bytes_up"] / rounds / len(clients)) if st["bytes_up"] else 0,
        uplink_serialised_bytes=len(msg) if msg else 0,
        downlink_bytes_per_client_per_round=(st["bytes_down"] / rounds / len(clients)) if st["bytes_down"] else 0,
        KiB_per_round_all_clients=(st["bytes_up"] + st["bytes_down"]) / rounds / 1024,
        MiB_to_convergence=(st["bytes_up"] + st["bytes_down"]) / 2**20,
        DC1_local_train_s_per_round=st["local_time_per_client_s"][1] / rounds,
        peak_rss_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        descriptor_bytes=nb.DESCRIPTOR_BYTES, validation_scenario=val_scen, train_scenarios=train_scens,
        perturb=nb.PERTURB)
    common.save_json(log, os.path.join(args.results, "train_log.json"))
    print(json.dumps({k: v for k, v in log.items() if k not in ("system", "train_stats", "timing")}, indent=1, default=float))


if __name__ == "__main__":
    main()
