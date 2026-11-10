"""NumPy port of the notebook's FedGuardPT module (section 10) and federated trainer (section 11).

Architecture, loss, optimiser and communication accounting follow the notebook:
  inp   : Conv1d(C_in, 16, 1)                                        shared
  block : pad -> dw1(k=3,d) -> pw1 -> ReLU -> pad -> dw2 -> pw2 ; out = ReLU(x + h)   shared
  heads : Linear(16,3) forecast, Linear(16,7) residual, Linear(16,3) regime            local
  loss  = MSE_fc + 0.5 MSE_res + 0.1 CE_reg ; Adam(lr=2e-3) ; 25 rounds x 1 local epoch
Tensors are (B, L, C). Parameter names match the PyTorch state_dict.
"""
import struct
import time

import numpy as np

from . import nb

C, K, DIL = nb.CFG["TCN_C"], nb.CFG["TCN_K"], nb.CFG["TCN_DIL"]


def shared_keys(p):
    return [n for n in p if n.startswith("inp.") or n.startswith("blocks.")]


def init_params(rng, c_in=None, n_fc=3, n_res=7, n_reg=3):
    """PyTorch default init: U(-1/sqrt(fan_in), 1/sqrt(fan_in)) for weights and biases."""
    if c_in is None:            # REV: input width follows nb.TCN_INPUT (raw-signal variant uses 3 channels)
        c_in = nb.N_CH if nb.TCN_INPUT is None else len(nb.TCN_INPUT)
    def u(shape, fan_in):
        b = 1.0 / np.sqrt(fan_in)
        return rng.uniform(-b, b, shape).astype(np.float32)
    p = {"inp.weight": u((C, c_in), c_in), "inp.bias": u((C,), c_in)}
    for i, _ in enumerate(DIL):
        for j in (1, 2):
            p[f"blocks.{i}.dw{j}.weight"] = u((C, K), K)
            p[f"blocks.{i}.pw{j}.weight"] = u((C, C), C)
            p[f"blocks.{i}.pw{j}.bias"] = u((C,), C)
    for h, n in (("head_fc", n_fc), ("head_res", n_res), ("head_reg", n_reg)):
        p[f"{h}.weight"] = u((n, C), C)
        p[f"{h}.bias"] = u((n,), C)
    return p


def n_params(p):
    return int(sum(v.size for v in p.values()))


def _dw(x, w, d):
    B, L, Ch = x.shape
    xp = np.concatenate([np.zeros((B, (K - 1) * d, Ch), x.dtype), x], axis=1)
    y = np.zeros_like(x)
    for j in range(K):
        y += xp[:, j * d:j * d + L, :] * w[:, j]
    return y, xp


def _dw_back(dy, xp, w, d):
    L = dy.shape[1]
    dw = np.empty_like(w)
    dxp = np.zeros_like(xp)
    for j in range(K):
        dw[:, j] = np.einsum("blc,blc->c", dy, xp[:, j * d:j * d + L, :])
        dxp[:, j * d:j * d + L, :] += dy * w[:, j]
    return dxp[:, (K - 1) * d:, :], dw


def forward(p, x, cache=False):
    """x: (B, L, C_in) -> (fc, res, reg_logits) from the last time step."""
    cc = [x]
    h = x @ p["inp.weight"].T + p["inp.bias"]
    for i, d in enumerate(DIL):
        pre = f"blocks.{i}."
        u1, xp1 = _dw(h, p[pre + "dw1.weight"], d)
        v1 = u1 @ p[pre + "pw1.weight"].T + p[pre + "pw1.bias"]
        a1 = np.maximum(v1, 0)
        u2, xp2 = _dw(a1, p[pre + "dw2.weight"], d)
        v2 = u2 @ p[pre + "pw2.weight"].T + p[pre + "pw2.bias"]
        s = h + v2
        out = np.maximum(s, 0)
        if cache:
            cc.append((xp1, u1, v1, xp2, u2, s))
        h = out
    z = h[:, -1, :]
    fc = z @ p["head_fc.weight"].T + p["head_fc.bias"]
    res = z @ p["head_res.weight"].T + p["head_res.bias"]
    reg = z @ p["head_reg.weight"].T + p["head_reg.bias"]
    if cache:
        return fc, res, reg, (cc, h, z)
    return fc, res, reg


def loss_grads(p, x, yfc, yres, yreg):
    fc, res, reg, (cc, h, z) = forward(p, x, cache=True)
    B = x.shape[0]
    lam, mu = nb.CFG["LAMBDA_REC"], nb.CFG["MU_REG"]
    efc, eres = fc - yfc, res - yres
    lg = reg - reg.max(1, keepdims=True)
    pr = np.exp(lg) / np.exp(lg).sum(1, keepdims=True)
    ce = -np.log(pr[np.arange(B), yreg] + 1e-30).mean()
    loss = float((efc ** 2).mean() + lam * (eres ** 2).mean() + mu * ce)
    g = {}
    dfc = 2 * efc / efc.size
    dres = lam * 2 * eres / eres.size
    oh = np.zeros_like(pr)
    oh[np.arange(B), yreg] = 1
    dreg = mu * (pr - oh) / B
    dz = np.zeros_like(z)
    for hname, dh in (("head_fc", dfc), ("head_res", dres), ("head_reg", dreg)):
        g[f"{hname}.weight"] = (dh.T @ z).astype(np.float32)
        g[f"{hname}.bias"] = dh.sum(0).astype(np.float32)
        dz += dh @ p[f"{hname}.weight"]
    dh = np.zeros_like(h)
    dh[:, -1, :] = dz
    for i in reversed(range(len(DIL))):
        d = DIL[i]
        pre = f"blocks.{i}."
        xp1, u1, v1, xp2, u2, s = cc[i + 1]
        ds = dh * (s > 0)
        dv2 = ds
        g[pre + "pw2.weight"] = np.einsum("blo,blc->oc", dv2, u2)
        g[pre + "pw2.bias"] = dv2.sum((0, 1))
        du2 = dv2 @ p[pre + "pw2.weight"]
        da1, g[pre + "dw2.weight"] = _dw_back(du2, xp2, p[pre + "dw2.weight"], d)
        dv1 = da1 * (v1 > 0)
        g[pre + "pw1.weight"] = np.einsum("blo,blc->oc", dv1, u1)
        g[pre + "pw1.bias"] = dv1.sum((0, 1))
        du1 = dv1 @ p[pre + "pw1.weight"]
        dhin, g[pre + "dw1.weight"] = _dw_back(du1, xp1, p[pre + "dw1.weight"], d)
        dh = dhin + ds
    g["inp.weight"] = np.einsum("blo,blc->oc", dh, cc[0])
    g["inp.bias"] = dh.sum((0, 1))
    return loss, g


def errors(p, x, yfc, yres):
    """(e_rec, e_fc) per window, as FedGuardPT.errors()."""
    fc, res, _ = forward(p, x)
    return ((res - yres) ** 2).mean(1), ((fc - yfc) ** 2).mean(1)


class Adam:
    """torch.optim.Adam defaults (betas 0.9/0.999, eps 1e-8, no weight decay)."""

    def __init__(self, p, lr, state=None):
        self.lr = lr
        if state is None:
            self.t = 0
            self.m = {k: np.zeros_like(v) for k, v in p.items()}
            self.v = {k: np.zeros_like(v) for k, v in p.items()}
        else:
            self.t, self.m, self.v = state

    def step(self, p, g):
        self.t += 1
        b1, b2 = 0.9, 0.999
        c1, c2 = 1 - b1 ** self.t, 1 - b2 ** self.t
        for k in p:
            self.m[k] = b1 * self.m[k] + (1 - b1) * g[k]
            self.v[k] = b2 * self.v[k] + (1 - b2) * g[k] * g[k]
            denom = np.sqrt(self.v[k] / c2) + 1e-8
            p[k] -= (self.lr / c1 * self.m[k] / denom).astype(np.float32)

    def state(self):
        return (self.t, self.m, self.v)


def local_train(p, ws, idx, epochs, lr, rng, opt_state=None):
    opt = Adam(p, lr, opt_state)
    n = len(idx)
    last = 0.0
    for _ in range(epochs):
        perm = rng.permutation(n)
        for s in range(0, n, nb.CFG["BATCH"]):
            b = idx[perm[s:s + nb.CFG["BATCH"]]]
            xb, yfc, yres, yreg = ws.arrays(b)
            loss, g = loss_grads(p, xb, yfc, yres, yreg)
            opt.step(p, g)
            last = loss
    return last, opt.state()


def quantize_int8(delta):
    """Notebook accounting (numel + 4 B per tensor) plus the real serialised message."""
    deq, nbytes, msg = {}, 0, bytearray()
    for n, v in delta.items():
        a = float(np.abs(v).max())
        s = a / 127.0 if a > 0 else 1.0
        q = np.clip(np.round(v / s), -127, 127).astype(np.int8)
        deq[n] = q.astype(np.float32) * s
        nbytes += q.size + 4
        msg += struct.pack("<f", s) + q.tobytes()   # tensor order and shapes are fixed by the architecture
    return deq, nbytes, bytes(msg)


def dp_clip(delta, clip, sigma, rng):
    nrm = float(np.sqrt(sum(float((v.astype(np.float64) ** 2).sum()) for v in delta.values()))) + 1e-12
    f = min(1.0, clip / nrm)
    return {n: (v * f + (rng.standard_normal(v.shape).astype(np.float32) * sigma * clip if sigma > 0 else 0.0))
            .astype(np.float32) for n, v in delta.items()}


def trimmed_mean(stack, trim):
    if stack.shape[0] > 2 * trim > 0:
        s = np.sort(stack, axis=0)
        return s[trim:stack.shape[0] - trim].mean(0)
    return stack.mean(0)


def clean_train_index(ws, rng, cap):
    idx = np.flatnonzero(ws.y == 0)
    if len(idx) > cap:
        idx = np.sort(rng.choice(idx, cap, replace=False))
    return idx


def federated_train(train_sets, mode, seed, log=print, on_phase=None):
    """mode 'partial' (FedGuard-PT) or 'local' (Local-only). Returns ({dc: params}, stats)."""
    rounds = nb.CFG["ROUNDS"]
    rng = np.random.default_rng(seed)
    init_rng = np.random.default_rng(seed + 123)
    train_rng = np.random.default_rng(seed + 7)
    clients = sorted(train_sets)
    packs = {}
    for k in clients:
        sets = train_sets[k]
        merged = nb.Merged(sets) if len(sets) > 1 else sets[0]
        packs[k] = (merged, clean_train_index(merged, rng, nb.CFG["MAX_TRAIN_WIN_PER_CLIENT"]))
    stats = dict(mode=mode, rounds=0, bytes_up=0, bytes_down=0, local_time_s=0.0,
                 local_time_per_client_s={k: 0.0 for k in clients}, convergence=[], round_log=[])
    t_start = time.perf_counter()
    if mode == "central":
        # REV (offline comparison): one model trained on the pooled windows of all sites (raw data leave the sites,
        # no model exchange). Same capped clean windows, epochs, optimizer and seed stream as the federated mode.
        m = init_params(init_rng)
        merged = nb.Merged([packs[k][0] for k in clients])
        off, idx = 0, []
        for k in clients:
            idx.append(packs[k][1] + off)
            off += len(packs[k][0].starts)
        idx = np.concatenate(idx)
        t0 = time.perf_counter()
        opt = None
        for r in range(rounds):
            l, opt = local_train(m, merged, idx, 1, nb.CFG["LR"], train_rng, opt)
            stats["convergence"].append(float(l))
            log(f"  [central] round {r + 1:2d}/{rounds} loss={l:.4f}")
        stats["local_time_s"] = time.perf_counter() - t0
        stats["rounds"] = rounds
        stats["wall_s"] = time.perf_counter() - t_start
        return {k: {n: v.copy() for n, v in m.items()} for k in clients}, stats
    if mode == "local":
        models = {}
        for k in clients:
            m = init_params(init_rng)
            t0 = time.perf_counter()
            for r in range(rounds):
                l, _ = local_train(m, packs[k][0], packs[k][1], 1, nb.CFG["LR"], train_rng)
                log(f"  [local DC{k}] round {r + 1:2d}/{rounds} loss={l:.4f}")
            dt = time.perf_counter() - t0
            stats["local_time_s"] += dt
            stats["local_time_per_client_s"][k] = dt
            models[k] = m
        stats["rounds"] = rounds
        stats["wall_s"] = time.perf_counter() - t_start
        return models, stats

    g = init_params(init_rng)
    skeys = shared_keys(g)
    locals_ = {k: init_params(init_rng) for k in clients}
    ef = {k: {n: np.zeros_like(g[n]) for n in skeys} for k in clients}
    opt_state = {k: None for k in clients}
    dprng = np.random.default_rng(seed + 1)
    msg_example = None
    for r in range(rounds):
        deltas, losses, rl = [], [], dict(round=r + 1)
        for k in clients:
            for n in skeys:
                locals_[k][n] = g[n].copy()
            if on_phase:
                on_phase(f"train:DC{k}")
            t0 = time.perf_counter()
            l, opt_state[k] = local_train(locals_[k], packs[k][0], packs[k][1], 1, nb.CFG["LR"], train_rng, opt_state[k])
            dt = time.perf_counter() - t0
            stats["local_time_s"] += dt
            stats["local_time_per_client_s"][k] += dt
            rl[f"DC{k}_local_s"] = dt
            losses.append(l)
            t0 = time.perf_counter()
            d = {n: (locals_[k][n] - g[n]) + ef[k][n] for n in skeys}
            d = dp_clip(d, nb.CFG["DP_CLIP"], nb.CFG["DP_SIGMA"], dprng)
            deq, nbytes, msg = quantize_int8(d)
            for n in skeys:
                ef[k][n] = d[n] - deq[n]
            rl[f"DC{k}_compress_s"] = time.perf_counter() - t0
            rl[f"DC{k}_uplink_B"] = nbytes
            rl[f"DC{k}_uplink_serialised_B"] = len(msg)
            msg_example = msg
            stats["bytes_up"] += nbytes
            deltas.append(deq)
        t0 = time.perf_counter()
        for n in skeys:
            g[n] = g[n] + trimmed_mean(np.stack([d[n] for d in deltas]), nb.CFG["TRIM"]).astype(np.float32)
        rl["aggregate_s"] = time.perf_counter() - t0
        stats["bytes_down"] += sum(g[n].size * 4 for n in skeys) * len(clients)
        stats["convergence"].append(float(np.mean(losses)))
        rl["mean_loss"] = float(np.mean(losses))
        stats["round_log"].append(rl)
        stats["rounds"] = r + 1
        log(f"  round {r + 1:2d}/{rounds}  mean loss={np.mean(losses):.4f}  "
            f"local s/client={np.mean([rl[f'DC{k}_local_s'] for k in clients]):.2f}")
    for k in clients:
        for n in skeys:
            locals_[k][n] = g[n].copy()
    stats["wall_s"] = time.perf_counter() - t_start
    stats["shared_params"] = int(sum(g[n].size for n in skeys))
    stats["uplink_message_example"] = msg_example
    return locals_, stats
