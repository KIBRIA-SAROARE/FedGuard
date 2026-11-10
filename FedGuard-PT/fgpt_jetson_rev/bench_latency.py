#!/usr/bin/env python3
"""Per-window inference latency of the deployed DC model: NumPy port (threads sweep) and, if PyTorch
is importable on the board, the notebook's PyTorch module with the same weights (the paper's 2.91 ms
figure was PyTorch 2.14 on the cloud CPU container that ran the notebook). Output: results/bench_latency.csv
"""
import argparse
import csv
import json
import os
import subprocess
import sys

CHILD = r"""
import json, sys, time, numpy as np
sys.path.insert(0, sys.argv[1])
from fgpt import tcn_np
p = dict(np.load(sys.argv[2])); n = int(sys.argv[3]); B = int(sys.argv[4])
x = np.random.default_rng(0).normal(size=(B, 128, 10)).astype(np.float32)
for _ in range(50): tcn_np.forward(p, x)
ts = np.empty(n)
for i in range(n):
    a = time.perf_counter_ns(); tcn_np.forward(p, x); ts[i] = (time.perf_counter_ns() - a) / 1e6
print(json.dumps(dict(mean=ts.mean(), p50=np.percentile(ts, 50), p99=np.percentile(ts, 99))))
"""

TORCH = r"""
import json, sys, time, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
torch.set_num_threads(int(sys.argv[4]))
class B_(nn.Module):
    def __init__(s, c, k, d):
        super().__init__(); s.pad=(k-1)*d
        s.dw1=nn.Conv1d(c,c,k,dilation=d,groups=c,bias=False); s.pw1=nn.Conv1d(c,c,1)
        s.dw2=nn.Conv1d(c,c,k,dilation=d,groups=c,bias=False); s.pw2=nn.Conv1d(c,c,1)
    def forward(s, x):
        h=F.relu(s.pw1(s.dw1(F.pad(x,(s.pad,0))))); h=s.pw2(s.dw2(F.pad(h,(s.pad,0)))); return F.relu(x+h)
class M(nn.Module):
    def __init__(s):
        super().__init__(); s.inp=nn.Conv1d(10,16,1); s.blocks=nn.ModuleList([B_(16,3,d) for d in [1,2,4,8,16]])
        s.head_fc=nn.Linear(16,3); s.head_res=nn.Linear(16,7); s.head_reg=nn.Linear(16,3)
    def forward(s, x):
        h=s.inp(x)
        for b in s.blocks: h=b(h)
        z=h[:,:,-1]; return s.head_fc(z), s.head_res(z), s.head_reg(z)
m=M(); sd=m.state_dict(); p=dict(np.load(sys.argv[2]))
for k in sd: sd[k].copy_(torch.from_numpy(p[k]).reshape(sd[k].shape))
m.eval(); x=torch.randn(1,10,128); n=int(sys.argv[3])
with torch.no_grad():
    for _ in range(20): m(x)
    ts=np.empty(n)
    for i in range(n):
        a=time.perf_counter_ns(); m(x); ts[i]=(time.perf_counter_ns()-a)/1e6
print(json.dumps(dict(mean=ts.mean(), p50=np.percentile(ts,50), p99=np.percentile(ts,99))))
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--dc", type=int, default=1)
    ap.add_argument("--threads", default="1,2,4,6")
    ap.add_argument("--iters", type=int, default=1000)
    a = ap.parse_args()
    root = os.path.dirname(os.path.abspath(__file__))
    model = os.path.join(a.results, "bundle", f"DC{a.dc}.npz")
    rows = []
    for th in [int(v) for v in a.threads.split(",")]:
        env = dict(os.environ, OMP_NUM_THREADS=str(th), OPENBLAS_NUM_THREADS=str(th))
        for backend, code, extra in (("numpy", CHILD, ["1"]), ("torch", TORCH, [str(th)])):
            r = subprocess.run([sys.executable, "-c", code, root, model, str(a.iters)] + extra,
                               env=env, capture_output=True, text=True)
            if r.returncode != 0:
                if backend == "torch" and th == 1:
                    print("PyTorch not usable on this board; skipping torch backend")
                continue
            d = json.loads(r.stdout.strip().splitlines()[-1])
            d.update(backend=backend, threads=th, batch=1)
            rows.append(d)
            print(f"{backend:5s} threads={th}  mean={d['mean']:.3f} ms  p50={d['p50']:.3f}  p99={d['p99']:.3f}")
    with open(os.path.join(a.results, "bench_latency.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["backend", "threads", "batch", "mean", "p50", "p99"])
        w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    main()
