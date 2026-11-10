"""Sample-by-sample edge versions of nb.features() and the notebook's per-window score.

The monitor sees one telemetry sample every 10 ms. Every HOP samples a 1.28 s window closes
(its forecast target is the sample that just arrived) and a score is produced, which is exactly
the window grid window_starts() uses offline. run_monitor.py checks streaming == batch.
"""
import collections

import numpy as np

from . import nb, tcn_np


class StreamingFeatures:
    def __init__(self, k, cal):
        self.k, self.cal, self.p = k, cal, nb.DC_PARAMS[k]
        self.mu = np.array([cal["stats"][c][0] for c in nb.FEAT_CH])
        self.sd = np.array([cal["stats"][c][1] for c in nb.FEAT_CH])
        W = nb.CFG["CSEC_EVWIN"]
        self.bV = collections.deque(maxlen=W)
        self.bP = collections.deque(maxlen=W)
        self.prev = None  # (Pcf, ref) of the previous sample
        self.a = nb.CFG["DT"] / self.p["tau_cl"]
        # REV: UPS-check state (same rule as nb.ups_check)
        self.nw = int(round(nb.UPS_VWIN / nb.CFG["DT"]))
        self.G = int(round(nb.UPS_G / nb.CFG["DT"]))
        self.bVx = collections.deque(maxlen=self.nw + 1)
        self.m_prev = None
        self.i = 0
        self.last_change = -10 ** 9
        self.xflag = 0.0
        self.vlim = nb.VTRIP[k - 1] + nb.UPS_DV

    def step(self, Va, P_meas, P_total, Q_total, P_it, P_cool, P_aux, ups, P_ai):
        p, cal = self.p, self.cal
        s = cal["s_k"]
        r_bal = P_total - (P_it + P_cool + P_aux)
        Vdn = np.sqrt(max(P_aux / (p["Faux"] * p["Pdc"]), 0.0))
        r_aux = Va - Vdn
        den = max((1.0 - p["Fvfd"]) * Vdn ** 2 + p["Fvfd"], 1e-6)
        Pcf = P_cool / den
        ref = p["Fcool"] * p["Pdc"] * (nb.COOL_A + nb.COOL_B * P_ai)
        r_cool = 0.0 if self.prev is None else Pcf - self.prev[0] - (self.prev[1] - self.prev[0]) * self.a
        self.prev = (Pcf, ref)
        m = ups != 0
        self.bVx.append(Va)
        if nb.LEGACY_UPS:
            r_ups = (P_it if m else 0.0) / s
        else:
            if self.m_prev is not None and m != self.m_prev:
                self.last_change = self.i
                if m:
                    self.xflag = 1.0 if min(self.bVx) > self.vlim else 0.0
            if not m:
                self.xflag = 0.0
            a_ = P_it / (p["Fit"] * p["Pdc"]) if (m and self.i - self.last_change >= self.G) else 0.0
            r_ups = max(a_, self.xflag)
        self.m_prev = m
        self.i += 1
        Qcool = nb.TAN_COOL * ((1.0 - p["Fvfd"]) * Pcf * Vdn ** 2 + nb.CFG["K_V"] * p["Fvfd"] * Pcf)
        Qaux = p["Faux"] * p["Pdc"] * nb.TAN_AUX * Vdn ** 2
        if ups != 0:
            Qit = 0.0
        elif p["Qmode"] == 0:
            Qit = P_it * nb.TAN_IT
        elif p["Qmode"] == 1:
            Qit = -p["Qset"] * p["Fit"] * p["Pdc"]
        else:
            lim = nb.QLIM_PU * p["Fit"] * p["Pdc"]
            Qit = float(np.clip(-nb.KQ_DROOP * (Vdn - 1.0) * p["Fit"] * p["Pdc"], -lim, lim))
        r_qp = Q_total - (Qit + Qcool + Qaux)
        self.bV.append(Va)
        self.bP.append(P_total / s)
        pz = P_total / s
        sc = np.array([r_bal / s, r_aux - cal["aux_off"], r_cool / s, r_ups, r_qp / s,
                       (max(self.bV) - min(self.bV)) - cal["S_k"] * (max(self.bP) - min(self.bP)),
                       float(np.min(np.abs(pz - cal["reg_centroids"]))),
                       (Va - cal["v_mu"]) / cal["v_sd"], pz, Q_total / s])
        return np.clip((sc - self.mu) / self.sd, -nb.Z_CLIP, nb.Z_CLIP).astype(np.float32)


class WindowScorer:
    """Ring buffer of the last WIN+1 feature rows -> window score every HOP samples."""

    def __init__(self, params, fusion, k=1):
        self.p = params
        self.W, self.H = nb.CFG["WIN"], nb.CFG["HOP"]
        self.buf = collections.deque(maxlen=self.W + 1)
        self.coef = np.asarray(fusion["coef"], np.float64)
        self.b = float(fusion["intercept"])
        self.ms = nb.fusion_ms(fusion, k)
        self.kappa, self.eta, self.thr = fusion["kappa"], fusion["eta"], fusion["threshold"]
        self.n = 0

    def push(self, x, csec_c=None, csec_u=None):
        """Returns None or a dict for the window [n-WIN-1, n-1) whose target is x."""
        self.buf.append(x)
        self.n += 1
        s = self.n - self.W - 1          # start index of the window whose target just arrived
        if s < 0 or s % self.H:
            return None
        arr = np.asarray(self.buf)
        win, nxt = arr[:self.W], arr[self.W]
        ab = np.abs(win[:, nb.RESID_IDX]).astype(np.float64)
        res_terms = ab.sum(0) / self.W
        e_rec, e_fc = tcn_np.errors(self.p, win[None], nxt[None, nb.SIG_IDX], nxt[None, nb.RESID_IDX])
        T = np.r_[res_terms, e_rec, e_fc].astype(np.float32)
        Z = nb.z_terms(T[None], self.ms)
        c = u = 0.0
        if csec_c is not None:
            c = float(csec_c[s:s + self.W].astype(np.float64).sum() / self.W)
            u = float(csec_u[s:s + self.W].astype(np.float64).sum() / self.W)
        score = float(Z[0] @ self.coef + self.b - self.kappa * c + self.eta * u)
        dom, cs, _ = nb.explain(Z, self.coef)
        return dict(start=s, score=score, alarm=score > self.thr, dominant=dom[0], domain=cs[0],
                    csec_corr=c, csec_unc=u, terms=Z[0])
