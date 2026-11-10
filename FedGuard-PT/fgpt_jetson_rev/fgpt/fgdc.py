"""FedGuard-DC baseline, ported from the FedGuard-DC notebook (NumPy) for the offline comparison.

Kept from the notebook (same code path): dual-head MLP (encoder 2 x ReLU -> forecast head + reconstruction head),
loss MSE_fc + LAM * MSE_rec, Adam, local epochs, variability-weighted FedAvg (weight n * sqrt(loss)), update clipping,
z-scored score alpha * z_rec + beta * z_fc, alpha/beta grid and F1-max quantile threshold (tune_threshold),
features [Va, Vb, Vc, P / local max P], window W = 40 steps, horizon H = 25 steps, HID 256, LAT 128, 150 rounds.

Adapted to this dataset (stated in the paper):
  - records are already 10 ms (the notebook downsampled 1 ms to 10 ms), so no further downsampling; W and H keep the
    notebook's time scales (0.40 s window, 0.25 s horizon);
  - positive-sequence phasor telemetry: Vb = Vc = Va;
  - the power channel is the reported total power P_total; its local scale is the maximum over the site's training
    records (kept at the site, as in the notebook);
  - training windows: attack-free training scenarios of the fold, every TRAIN_HOP-th window (compute budget);
  - calibration statistics: clean windows of the calibration scenario (S01), pooled over the six sites;
  - alpha, beta and the threshold are tuned on the same validation set as FedGuard-PT (the notebook tuned them on its
    test windows);
  - scoring is causal at every FedGuard-PT window end e: input steps e-H-W+1 .. e-H, forecast target P(e).
"""
import time

import numpy as np
from sklearn.metrics import f1_score

W, H = 40, 25
ROUNDS, EPOCHS = 150, 2
LR, HID, LAT, LAM = 2e-3, 256, 128, 0.5
DP_CLIP, DP_SIGMA = 5.0, 0.0
SEED = 6
TRAIN_HOP = 10
SETTLE_S = 5.0


# ------------------------------------------------------------------ notebook network (verbatim logic)
def init_params(in_dim, hid=HID, lat=LAT, seed=0):
    rng = np.random.default_rng(seed)
    g = lambda a, b: (rng.standard_normal((a, b)) * np.sqrt(2.0 / a)).astype(np.float32)  # noqa: E731
    return {'W1': g(in_dim, hid), 'b1': np.zeros(hid, np.float32), 'W2': g(hid, lat), 'b2': np.zeros(lat, np.float32),
            'Wf': g(lat, 1), 'bf': np.zeros(1, np.float32), 'Wd': g(lat, in_dim), 'bd': np.zeros(in_dim, np.float32)}


def n_params(P):
    return int(sum(v.size for v in P.values()))


def forward(P, X):
    z1 = X @ P['W1'] + P['b1']; a1 = np.maximum(0, z1)
    z2 = a1 @ P['W2'] + P['b2']; lat = np.maximum(0, z2)
    return (lat @ P['Wf'] + P['bf'])[:, 0], lat @ P['Wd'] + P['bd'], (X, z1, a1, z2, lat)


def loss_and_grads(P, X, y, lam=LAM):
    B = X.shape[0]
    yhat, xhat, (Xin, z1, a1, z2, lat) = forward(P, X)
    rf, rd = yhat - y, xhat - Xin
    loss = (rf ** 2).mean() + lam * (rd ** 2).mean()
    g = {}
    dyhat = (2.0 / B) * rf
    g['Wf'] = lat.T @ dyhat[:, None]; g['bf'] = dyhat.sum()[None]
    dxhat = (2.0 * lam / B) * rd
    g['Wd'] = lat.T @ dxhat; g['bd'] = dxhat.sum(0)
    dlat = dyhat[:, None] @ P['Wf'].T + dxhat @ P['Wd'].T
    dz2 = dlat * (z2 > 0)
    g['W2'] = a1.T @ dz2; g['b2'] = dz2.sum(0)
    dz1 = (dz2 @ P['W2'].T) * (z1 > 0)
    g['W1'] = Xin.T @ dz1; g['b1'] = dz1.sum(0)
    return float(loss), g


def local_train(P, X, y, epochs=EPOCHS, bs=256, lr=LR, lam=LAM, seed=0):
    P = {k: v.copy() for k, v in P.items()}
    m = {k: np.zeros_like(v) for k, v in P.items()}; v_ = {k: np.zeros_like(v) for k, v in P.items()}; t = 0
    rng = np.random.default_rng(seed); n = X.shape[0]; last = 0.0
    for _ in range(epochs):
        perm = rng.permutation(n)
        for s in range(0, n, bs):
            b = perm[s:s + bs]; t += 1
            last, g = loss_and_grads(P, X[b], y[b], lam)
            for k in P:
                m[k] = 0.9 * m[k] + 0.1 * g[k]
                v_[k] = 0.999 * v_[k] + 0.001 * g[k] ** 2
                P[k] -= lr * (m[k] / (1 - 0.9 ** t)) / (np.sqrt(v_[k] / (1 - 0.999 ** t)) + 1e-8)
    return P, last


def dp_protect(update, clip=DP_CLIP, sigma=DP_SIGMA, seed=0):
    rng = np.random.default_rng(seed)
    norm = np.linalg.norm(np.concatenate([u.ravel() for u in update.values()])) + 1e-12
    f = min(1.0, clip / norm)
    return {k: u * f + (rng.normal(0, sigma * clip, u.shape) if sigma > 0 else 0.0) for k, u in update.items()}


def train_fed(clients, rounds=None, seed0=SEED, log=print):
    """clients: list of dict(X=(n, in_dim), y=(n,)). Variability-weighted FedAvg as in the notebook."""
    rounds = ROUNDS if rounds is None else rounds
    in_dim = clients[0]["X"].shape[1]
    G = init_params(in_dim, seed=seed0)
    t0 = time.perf_counter()
    for r in range(rounds):
        ups, ws, ls = [], [], []
        for ci, c in enumerate(clients):
            P, l = local_train(G, c["X"], c["y"], seed=r)
            ups.append(dp_protect({k: P[k] - G[k] for k in G}, seed=r * 10 + ci))
            ws.append(len(c["y"]) * np.sqrt(l)); ls.append(l)
        w = np.asarray(ws, float); w = w / w.sum()
        G = {k: G[k] + np.tensordot(w, np.stack([u[k] for u in ups]), axes=(0, 0)).astype(np.float32) for k in G}
        if (r + 1) % 25 == 0 or r == 0:
            log(f"  [FedGuard-DC] round {r + 1:3d}/{rounds} mean local loss={np.mean(ls):.5f}")
    n = n_params(G)
    stats = dict(rounds=rounds, wall_s=time.perf_counter() - t0, params=n,
                 bytes_up_per_client_per_round=4 * n, bytes_down_per_client_per_round=4 * n,
                 KiB_per_round_all_clients=2 * 4 * n * len(clients) / 1024)
    return G, stats


# ------------------------------------------------------------------ data adaptation
def site_series(tel, scale):
    """(T, 4) features [Va, Vb, Vc, P/scale]; Vb = Vc = Va (positive-sequence telemetry)."""
    va = tel["Va_pu"].astype(np.float32)
    return np.column_stack([va, va, va, (tel["P_total_MW"] / scale).astype(np.float32)])


def train_windows(F, t, hop=TRAIN_HOP):
    i0 = int(np.searchsorted(t, SETTLE_S))
    idx = np.arange(i0, len(F) - W - H + 1, hop)
    X = np.stack([F[i:i + W].ravel() for i in idx])
    return X.astype(np.float32), F[idx + W + H - 1, 3].astype(np.float32)


def windows_at_ends(F, ends):
    """Causal windows scored at sample index e: input e-H-W+1 .. e-H, target P(e)."""
    i = np.asarray(ends) - W - H + 1
    assert (i >= 0).all()
    X = np.stack([F[j:j + W].ravel() for j in i])
    return X.astype(np.float32), F[np.asarray(ends), 3].astype(np.float32)


def calibrate(P, X, y):
    yhat, xhat, _ = forward(P, X)
    rec, fres = ((xhat - X) ** 2).mean(1), (yhat - y) ** 2
    return dict(mu_r=float(rec.mean()), sd_r=float(rec.std() + 1e-9), mu_f=float(fres.mean()), sd_f=float(fres.std() + 1e-9))


def components(P, X, y, cal):
    yhat, xhat, _ = forward(P, X)
    return (((xhat - X) ** 2).mean(1) - cal["mu_r"]) / cal["sd_r"], ((yhat - y) ** 2 - cal["mu_f"]) / cal["sd_f"]


def tune_threshold(scores, labels):
    qs = np.quantile(scores, np.linspace(0.50, 0.999, 200))
    f1s = [f1_score(labels, (scores > t).astype(int), zero_division=0) for t in qs]
    j = int(np.argmax(f1s))
    return float(qs[j]), float(f1s[j])


def tune_alpha_beta(zr, zf, y):
    best = (-1.0, 1.0, 1.0, 0.0)
    for a in np.linspace(0.1, 5.0, 20):
        for b in np.linspace(0.0, 5.0, 20):
            thr, f1 = tune_threshold(a * zr + b * zf, y)
            if f1 > best[0]:
                best = (f1, float(a), float(b), thr)
    return dict(val_f1=best[0], alpha=best[1], beta=best[2], threshold=best[3])
