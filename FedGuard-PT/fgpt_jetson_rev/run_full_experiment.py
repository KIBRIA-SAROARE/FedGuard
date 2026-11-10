#!/usr/bin/env python3
"""Full on-board experiment for the revision (resumable). Every stage runs in its own process; a finished
stage is skipped on restart (delete its folder to redo it). Run inside tmux: the full plan takes many hours.

Plans (--plan, comma-separated list allowed, "all" = every plan in this order):
  main      idle baseline; 5 LOSO folds (S02..S06 held out) + FULL fold trained on the board;
            DC1 real-time streams (100 Hz wall clock):
              LOSO fold <S>: no_attack, T1, T2, T3a, T3, U1, A1..A5 on <S>          (55 streams)
              FULL fold    : ctrl S07..S11, ctrl_hide S07                           (6 streams)
            latency sweep, idle baseline                                   -> <out>/streams/
  attacked  FULL-model control-path streams at the attacked sites, real time:
              S07@DC4, S08@DC2, S08@DC5, S09@DC3, S10@DC6, S11@DC6, ctrl_hide S07@DC4, S09@DC3
                                                                           -> <out>/streams_attacked/
  sites     the 11 LOSO tests at DC2..DC6 for all five LOSO folds (275 streams), fast replay
            (--no-realtime: identical scores, no pacing; detection metrics only)
                                                                           -> <out>/streams_sites/
  sens      sensitivity to meter error and model mismatch (ASSUMED levels, see fgpt/nb.py):
            the S02 fold retrained per case with FGPT_PERTURB, then the 11 DC1 tests on S02, fast replay
                                                                           -> <out>/sens/<case>/
  quick     short pipeline check (S02 fold + FULL fold, 6 streams, real time)
"""
import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
LOSO = ["S02_fault_100ms", "S03_fault_250ms", "S04_line_trip", "S05_load_step", "S06_multi_event"]
CTRL = ["S07_atk_false_ups", "S08_atk_load_alter", "S09_atk_sensor_spoof", "S10_atk_cooling", "S11_fault_plus_atk"]
LOSO_TESTS = ["no_attack", "T1", "T2", "T3a", "T3", "U1", "A1", "A2", "A3", "A4", "A5"]
ATTACKED = [("ctrl", "S07_atk_false_ups", 4), ("ctrl", "S08_atk_load_alter", 2), ("ctrl", "S08_atk_load_alter", 5),
            ("ctrl", "S09_atk_sensor_spoof", 3), ("ctrl", "S10_atk_cooling", 6), ("ctrl", "S11_fault_plus_atk", 6),
            ("ctrl_hide", "S07_atk_false_ups", 4), ("ctrl_hide", "S09_atk_sensor_spoof", 3)]
# ASSUMED perturbation cases (A14). "base" reuses the main S02 fold.
SENS_CASES = {"base": {}, "noise1": {"noise": 1}, "noise3": {"noise": 3}, "kv04": {"k_v": 0.4}, "kv06": {"k_v": 0.6},
              "fvfd09": {"fvfd_scale": 0.9}, "tau12": {"tau_scale": 1.2}}
SENS_FOLD = "S02_fault_100ms"
STREAM_S = 298.0   # 29,801 samples at 10 ms
ORDER = ["main", "attacked", "sites", "sens"]


def log(msg, fh):
    line = f"{dt.datetime.now():%Y-%m-%d %H:%M:%S}  {msg}"
    print(line, flush=True)
    fh.write(line + "\n")
    fh.flush()


def run(cmd, fh, env):
    log("RUN " + " ".join(cmd), fh)
    t0 = time.time()
    p = subprocess.run(cmd, cwd=HERE, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    fh.write(p.stdout)
    fh.flush()
    print(p.stdout[-1500:], flush=True)
    log(f"exit={p.returncode} after {time.time() - t0:.1f} s", fh)
    if p.returncode != 0:
        raise SystemExit(f"stage failed: {' '.join(cmd)}")


class Runner:
    def __init__(self, a, fh, env):
        self.a, self.fh, self.env = a, fh, env
        self.out = os.path.abspath(a.out)

    def train(self, fold_dir, fold, env=None):
        if os.path.exists(os.path.join(fold_dir, "train_log.json")):
            log(f"skip training {fold_dir} (done)", self.fh)
            return
        run([sys.executable, "train_edge.py", "--dataset", self.a.dataset, "--results", fold_dir, "--test-scenario", fold],
            self.fh, env or self.env)
        time.sleep(self.a.cooldown_s)

    def stream(self, fold_dir, out_dir, test, scen, dc, realtime, env=None):
        if os.path.exists(os.path.join(out_dir, "metrics.json")):
            log(f"skip stream {out_dir} (done)", self.fh)
            return
        rt = realtime and not self.a.no_realtime
        run([sys.executable, "rt_monitor.py", "--dataset", self.a.dataset, "--fold-dir", fold_dir, "--out", out_dir,
             "--dc", str(dc), "--scenario", scen, "--test", test, "--net-delay-ms", str(self.a.net_delay_ms)]
            + ([] if rt else ["--no-realtime"]), self.fh, env or self.env)
        if rt:
            time.sleep(self.a.cooldown_s)

    def idle(self, name):
        d = os.path.join(self.out, name)
        if not os.path.exists(os.path.join(d, "resources_idle.csv")):
            run([sys.executable, "measure_idle.py", "--results", d, "--seconds", str(self.a.idle_s)], self.fh, self.env)

    def folds(self, names):
        for f in names:
            self.train(os.path.join(self.out, "folds", f), f)

    def plan_main(self):
        self.idle("idle_start")
        self.folds(LOSO + ["FULL"])
        jobs = [(f, t, f) for f in LOSO for t in LOSO_TESTS] + [("FULL", "ctrl", s) for s in CTRL] + [("FULL", "ctrl_hide", CTRL[0])]
        for i, (f, t, s) in enumerate(jobs, 1):
            log(f"main stream {i}/{len(jobs)}: fold={f} test={t} scenario={s}", self.fh)
            self.stream(os.path.join(self.out, "folds", f), os.path.join(self.out, "streams", f, f"{t}_{s}"), t, s, 1, True)
        self.idle("idle_end")
        lat = os.path.join(self.out, "folds", LOSO[0], "bench_latency.csv")
        if not os.path.exists(lat):
            run([sys.executable, "bench_latency.py", "--results", os.path.join(self.out, "folds", LOSO[0]), "--dc", "1"],
                self.fh, {k: v for k, v in self.env.items() if k not in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")})

    def plan_attacked(self):
        self.folds(["FULL"])
        for i, (t, s, k) in enumerate(ATTACKED, 1):
            log(f"attacked-site stream {i}/{len(ATTACKED)}: {t} {s} DC{k}", self.fh)
            self.stream(os.path.join(self.out, "folds", "FULL"), os.path.join(self.out, "streams_attacked", f"DC{k}", f"{t}_{s}"),
                        t, s, k, True)

    def plan_sites(self):
        self.folds(LOSO)
        jobs = [(k, f, t) for k in range(2, 7) for f in LOSO for t in LOSO_TESTS]
        for i, (k, f, t) in enumerate(jobs, 1):
            log(f"site stream {i}/{len(jobs)}: DC{k} fold={f} test={t}", self.fh)
            self.stream(os.path.join(self.out, "folds", f), os.path.join(self.out, "streams_sites", f"DC{k}", f, f"{t}_{f}"),
                        t, f, k, False)

    def plan_sens(self):
        self.folds([SENS_FOLD])
        for case, pert in SENS_CASES.items():
            env = dict(self.env, FGPT_PERTURB=json.dumps(pert))
            if case == "base":
                fd = os.path.join(self.out, "folds", SENS_FOLD)
            else:
                fd = os.path.join(self.out, "sens", case, "folds", SENS_FOLD)
                self.train(fd, SENS_FOLD, env)
            for t in LOSO_TESTS:
                log(f"sensitivity {case}: {t}", self.fh)
                self.stream(fd, os.path.join(self.out, "sens", case, "streams", f"{t}_{SENS_FOLD}"), t, SENS_FOLD, 1, False, env)

    def plan_quick(self):
        self.folds([LOSO[0], "FULL"])
        jobs = [(LOSO[0], t, LOSO[0]) for t in ["no_attack", "T1", "T3", "U1", "A1"]] + [("FULL", "ctrl", CTRL[0])]
        for f, t, s in jobs:
            self.stream(os.path.join(self.out, "folds", f), os.path.join(self.out, "streams", f, f"{t}_{s}"), t, s, 1, True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default="main", help="main | attacked | sites | sens | quick | all, or a comma list")
    ap.add_argument("--dataset", default="data/dataset")
    ap.add_argument("--out", default="results_rev")
    ap.add_argument("--idle-s", type=float, default=60)
    ap.add_argument("--cooldown-s", type=float, default=20, help="idle gap after each training and real-time stream")
    ap.add_argument("--net-delay-ms", type=float, default=0.0)
    ap.add_argument("--no-realtime", action="store_true", help="pipeline check only; paper_figures.py rejects these streams")
    a = ap.parse_args()
    plans = ORDER if a.plan == "all" else [p.strip() for p in a.plan.split(",")]
    for p in plans:
        if p not in ORDER + ["quick"]:
            raise SystemExit(f"unknown plan {p}")
    out = os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", FGPT_PERTURB="{}")
    fh = open(os.path.join(out, "experiment_log.txt"), "a")
    sys.path.insert(0, HERE)
    from fgpt import common
    common.save_json(dict(plans=plans, loso_tests=LOSO_TESTS, attacked=ATTACKED, sens_cases=SENS_CASES,
                          net_delay_ms=a.net_delay_ms, cooldown_s=a.cooldown_s, idle_s=a.idle_s,
                          no_realtime=a.no_realtime, system=common.system_info(),
                          jetson_clocks=common.sh("jetson_clocks --show 2>&1 | head -20")),
                     os.path.join(out, f"experiment_plan_{'_'.join(plans)}.json"))
    log(f"plans={plans}  out={out}", fh)
    R = Runner(a, fh, env)
    for p in plans:
        log(f"=== plan {p} ===", fh)
        getattr(R, f"plan_{p}")()
    log("all requested plans finished", fh)


if __name__ == "__main__":
    main()
