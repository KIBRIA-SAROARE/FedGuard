"""Port of FedGuard_PT_Phase2_Final.ipynb, sections 1, 4-9, 12-13 (config, physics constants,
invariant residuals, calibration, threat regimes, windowing, CSEC, fusion).

Functions keep the notebook's names and arithmetic. Differences are marked `# EDGE:`.
The notebook loads all 66 records into memory; here a record is loaded on demand from
<dataset>/clean/<scenario>_DC<k>.csv so the board only holds what it needs.
"""
import math
import os

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, average_precision_score

# ----------------------------------------------------------------------------- section 1 (subset)
CFG = {
    "SEED": 2026, "FS": 100.0, "DT": 0.01, "T_DROP": 2.0, "T_END": 300.0, "N_DC": 6,
    "CALIB_SCENARIO": "S01_normal",
    "WIN": 128, "HOP": 40, "LABEL_FRAC": 0.5, "MAX_TRAIN_WIN_PER_CLIENT": 2000,
    "TCN_C": 16, "TCN_K": 3, "TCN_DIL": [1, 2, 4, 8, 16],
    "ROUNDS": 25, "LOCAL_EPOCHS": 1, "BATCH": 256, "LR": 2e-3, "LAMBDA_REC": 0.5, "MU_REG": 0.1,
    "DP_CLIP": 5.0, "DP_SIGMA": 0.0, "TRIM": 1, "QUANT_BITS": 8,
    "ATK_SEED_TEST": 2026, "ATK_SEED_VAL": 7, "ATK_NWIN": (2, 4), "ATK_DUR": (8.0, 40.0), "ATK_TMIN": 20.0,
    "ATK_MAG": {1: (0.12, 0.35), 2: (0.10, 0.35), 3: (15.0, 60.0), 4: 0.04, 5: (0.03, 0.07)},
    "CSEC_EVWIN": 100, "CSEC_DTALIGN": 2.0, "CSEC_RHOMIN": 0.30, "CSEC_QTRIG": 0.9995,
    "ATTACK_FREE": ["S01_normal", "S02_fault_100ms", "S03_fault_250ms",
                    "S04_line_trip", "S05_load_step", "S06_multi_event"],
    "RECOVERY_S": 30.0,
    # REV: inner validation scenario per fold (never the held-out scenario; removed from that fold's training)
    "VAL_SCENARIO": {"S02_fault_100ms": "S03_fault_250ms", "S03_fault_250ms": "S02_fault_100ms",
                     "S04_line_trip": "S05_load_step", "S05_load_step": "S04_line_trip",
                     "S06_multi_event": "S02_fault_100ms", "FULL": "S06_multi_event"},
    # REV: CSEC gain grids widened (previous optima sat at the upper bounds 8 and 4)
    "KAPPA_GRID": [0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0],
    "ETA_GRID": [0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0, 1024.0],
    "K_V": 0.5,
    # REV: evaluation ignores windows ending in the first WARMUP_S s of each record (monitor start-up: filters and
    # window history initialise). Applies to all reported detection metrics and figures; training is unchanged.
    "WARMUP_S": 10.0,          # drive reactive fraction k_v (facility-model assumption, DC-CLM v1 code: Q_vfd = 0.5*...)
}
# The notebook indexes scenarios alphabetically over all 11; the attack seed uses that index.
SCENARIOS = ["S01_normal", "S02_fault_100ms", "S03_fault_250ms", "S04_line_trip", "S05_load_step",
             "S06_multi_event", "S07_atk_false_ups", "S08_atk_load_alter", "S09_atk_sensor_spoof",
             "S10_atk_cooling", "S11_fault_plus_atk"]

# ----------------------------------------------------------------------------- section 4
_CFG_ROWS = [
    ("DC1", 4, 350, 0.60, 0.30, 0.10, 0.40, 1, 1.75, 0, 0.00),
    ("DC2", 23, 200, 0.62, 0.28, 0.10, 0.55, 2, 0.70, 1, 0.05),
    ("DC3", 18, 150, 0.58, 0.32, 0.10, 0.30, 4, 0.60, 0, 0.00),
    ("DC4", 16, 300, 0.61, 0.29, 0.10, 0.45, 1, 1.20, 2, 0.00),
    ("DC5", 20, 450, 0.63, 0.27, 0.10, 0.60, 4, 2.60, 1, 0.08),
    ("DC6", 8, 250, 0.57, 0.33, 0.10, 0.35, 2, 0.55, 0, 0.00),
]
_TAUCL = [45.0, 30.0, 60.0, 45.0, 25.0, 55.0]
PF_IT, PF_COOL, PF_AUX = 0.95, 0.85, 0.98
TAN_IT, TAN_COOL, TAN_AUX = (math.tan(math.acos(PF_IT)), math.tan(math.acos(PF_COOL)), math.tan(math.acos(PF_AUX)))
KQ_DROOP, QLIM_PU = 4.0, 0.30
# REV (UPS check). Site UPS settings copied from dc_params.m (Vtrip_v, Vret_v, Tsoft_v, Pchg_v; DCP{k}(16:17)).
VTRIP = [0.90, 0.88, 0.92, 0.90, 0.87, 0.91]       # dc_params.m Vtrip_v: UPS transfer threshold (pu)
VRET = [0.95, 0.94, 0.96, 0.95, 0.93, 0.95]        # dc_params.m Vret_v
ETA_RECT, ETA_INV = 0.985, 0.975                   # dc_params.m DCP(16), DCP(17) [model assumptions there]
UPS_G = 2.0           # s; power part of the UPS check ignored for G after any reported change of m (design value)
UPS_DV = 0.02         # pu; a reported transfer onset with min V over the last UPS_VWIN s above Vtrip+UPS_DV is unexplained
UPS_VWIN = 0.2        # s
UPS_SCALE = 0.05      # pu of F_it*P_dc; fixed physical scale of the UPS check (design value, replaces S01 statistics)
LEGACY_UPS = os.environ.get("FGPT_LEGACY_UPS") == "1"
# grid override for edge probes only, e.g. FGPT_KAPPA_GRID="0,1,2,4,8,16,32,64,128,256" (probe_grid_edge.sh)
for _g, _e in (("KAPPA_GRID", "FGPT_KAPPA_GRID"), ("ETA_GRID", "FGPT_ETA_GRID")):
    if os.environ.get(_e):
        CFG[_g] = [float(v) for v in os.environ[_e].split(",")]   # ablation: original r_ups = 1{m!=0} P_it, statistical scale
COOL_A, COOL_B = 0.40, 0.60
DC_PARAMS = {}
_P0 = __import__("json").loads(os.environ.get("FGPT_PERTURB", "{}") or "{}")   # REV: detector-side parameter mismatch
for _i, _r in enumerate(_CFG_ROWS):
    DC_PARAMS[_i + 1] = dict(name=_r[0], bus=_r[1], Pdc=float(_r[2]), Fit=_r[3], Fcool=_r[4], Faux=_r[5],
                             Fvfd=min(_r[6] * float(_P0.get("fvfd_scale", 1.0)), 1.0), mode=_r[7], Ebatt=_r[8],
                             Qmode=_r[9], Qset=_r[10], tau_cl=_TAUCL[_i] * float(_P0.get("tau_scale", 1.0)))
if "k_v" in _P0:
    CFG["K_V"] = float(_P0["k_v"])

NE39_LINES = [
    (1, 2, 0.0035, 0.0411), (1, 39, 0.001, 0.025), (2, 3, 0.0013, 0.0151), (2, 25, 0.007, 0.0086),
    (2, 30, 0.0, 0.0181), (3, 4, 0.0013, 0.0213), (3, 18, 0.0011, 0.0133), (4, 5, 0.0008, 0.0128),
    (4, 14, 0.0008, 0.0129), (5, 8, 0.0008, 0.0112), (6, 5, 0.0002, 0.0026), (6, 7, 0.0006, 0.0092),
    (6, 11, 0.0007, 0.0082), (7, 8, 0.0004, 0.0046), (8, 9, 0.0023, 0.0363), (9, 39, 0.001, 0.025),
    (10, 11, 0.0004, 0.0043), (10, 13, 0.0004, 0.0043), (10, 32, 0.0, 0.02), (12, 11, 0.0016, 0.0435),
    (12, 13, 0.0016, 0.0435), (13, 14, 0.0009, 0.0101), (14, 15, 0.0018, 0.0217), (15, 16, 0.0009, 0.0094),
    (16, 17, 0.0007, 0.0089), (16, 19, 0.0016, 0.0195), (16, 21, 0.0008, 0.0135), (16, 24, 0.0003, 0.0059),
    (17, 18, 0.0007, 0.0082), (17, 27, 0.0013, 0.0173), (19, 33, 0.0007, 0.0142), (19, 20, 0.0007, 0.0138),
    (20, 34, 0.0009, 0.018), (21, 22, 0.0008, 0.014), (22, 23, 0.0006, 0.0096), (22, 35, 0.0, 0.0143),
    (23, 24, 0.0022, 0.035), (23, 36, 0.0005, 0.0272), (25, 26, 0.0032, 0.0323), (25, 37, 0.0006, 0.0232),
    (26, 27, 0.0014, 0.0147), (26, 28, 0.0043, 0.0474), (26, 29, 0.0057, 0.0625), (28, 29, 0.0014, 0.0151),
    (29, 38, 0.0008, 0.0156), (6, 31, 0.0, 0.025),
]
_L = pd.DataFrame(NE39_LINES, columns=["f", "t", "r", "x"])
_NB = int(max(_L.f.max(), _L.t.max()))
_w = np.maximum(np.sqrt(_L.r.values ** 2 + _L.x.values ** 2), 1e-6)
_A = sp.coo_matrix((_w, (_L.f.values - 1, _L.t.values - 1)), shape=(_NB, _NB)).tocsr()
_A = _A + _A.T
BUS_DIST = dijkstra(_A, directed=False)
DC_BUSES = [DC_PARAMS[k]["bus"] for k in range(1, 7)]
DCDIST = np.array([[BUS_DIST[a - 1, b - 1] for b in DC_BUSES] for a in DC_BUSES])

# ----------------------------------------------------------------------------- section 5
USE_COLS = ["t", "Va_pu", "P_meas_MW", "P_total_MW", "Q_total_Mvar", "P_it_MW",
            "P_cool_MW", "P_aux_MW", "ups_state", "vdc_pu", "P_ai_pu", "P_it_served_MW"]
TEL_KEYS = ["Va_pu", "P_meas_MW", "P_total_MW", "Q_total_Mvar", "P_it_MW",
            "P_cool_MW", "P_aux_MW", "ups_state", "P_ai_pu", "P_it_served_MW"]
DATASET_DIR = os.environ.get("FGPT_DATASET", "data/dataset")

# ----------------------------------------------------------------------------- REV: sensitivity hooks
import json as _json
import zlib as _zlib
PERTURB = _json.loads(os.environ.get("FGPT_PERTURB", "{}") or "{}")
# Meter-error model per unit level (ASSUMED values, stated as such in the paper):
#   voltage: additive white noise, 0.001 pu std;  P/Q channels: white noise 0.2 % of reading std
#   plus a fixed gain error per (site, channel) drawn uniformly in +/-0.2 %.
NOISE_V_STD, NOISE_REL_STD, GAIN_REL = 1e-3, 2e-3, 2e-3
_NOISY = ["Va_pu", "P_meas_MW", "P_total_MW", "Q_total_Mvar", "P_it_MW", "P_cool_MW", "P_aux_MW"]


def _site_of(path):
    b = os.path.basename(path)
    i = b.find("_DC")
    return int(b[i + 3]) if i >= 0 else 0


def _perturb_record(rec, path):
    lev = float(PERTURB.get("noise", 0.0))
    if lev <= 0:
        return rec
    k = _site_of(path)
    rng_n = np.random.default_rng(_zlib.crc32(os.path.basename(path).encode()))
    for j, c in enumerate(_NOISY):
        x = rec[c].astype(np.float64)
        g = 1.0 + lev * GAIN_REL * np.random.default_rng(1000 * k + j).uniform(-1.0, 1.0)
        if c == "Va_pu":
            y = x * g + rng_n.normal(0.0, lev * NOISE_V_STD, len(x))
        else:
            y = x * g * (1.0 + rng_n.normal(0.0, lev * NOISE_REL_STD, len(x)))
        rec[c] = y.astype(np.float32)
    return rec
_DATA = {}
TVEC = None


def load_record(path):
    """EDGE: one CSV -> the notebook's per-record dict (t >= 2 s, float32 channels, float64 t)."""
    df = pd.read_csv(path)
    extra = {}
    df = df.loc[df.t >= CFG["T_DROP"] - 1e-9].reset_index(drop=True)
    for c in ("label", "attack_type"):
        if c in df:
            extra[c] = df[c].to_numpy()
    rec = {c: df[c].to_numpy(np.float64 if c == "t" else np.float32) for c in USE_COLS}
    return _perturb_record(rec, path), extra


def DATA(scen, k):
    global TVEC
    key = (scen, k)
    if key not in _DATA:
        rec, _ = load_record(os.path.join(DATASET_DIR, "clean", f"{scen}_DC{k}.csv"))
        _DATA[key] = rec
        if TVEC is None:
            TVEC = rec["t"].copy()
    return _DATA[key]


def tvec():
    if TVEC is None:
        DATA(CFG["CALIB_SCENARIO"], 1)
    return TVEC


# ----------------------------------------------------------------------------- section 6
RESID_CH = ["r_bal", "r_aux", "r_cool", "r_ups", "r_qp", "r_pv", "r_reg"]
SIG_CH = ["v_z", "p_z", "q_z"]
FEAT_CH = RESID_CH + SIG_CH
N_CH = len(FEAT_CH)


def telemetry(scen, k):
    return {c: DATA(scen, k)[c].astype(np.float64).copy() for c in TEL_KEYS}


def telemetry_from_record(rec):
    return {c: rec[c].astype(np.float64).copy() for c in TEL_KEYS}


def rolling_range(x, w):
    s = pd.Series(x)
    return (s.rolling(w, min_periods=1).max() - s.rolling(w, min_periods=1).min()).to_numpy()


def q_hat(tel, p, Vdn, Pcf):
    Pm = (1.0 - p["Fvfd"]) * Pcf * Vdn ** 2
    Pv = p["Fvfd"] * Pcf
    Qcool = TAN_COOL * (Pm + CFG["K_V"] * Pv)
    Qaux = p["Faux"] * p["Pdc"] * TAN_AUX * Vdn ** 2
    if p["Qmode"] == 0:
        Qit = tel["P_it_MW"] * TAN_IT
    elif p["Qmode"] == 1:
        Qit = np.full_like(Vdn, -p["Qset"] * p["Fit"] * p["Pdc"])
    else:
        lim = QLIM_PU * p["Fit"] * p["Pdc"]
        Qit = np.clip(-KQ_DROOP * (Vdn - 1.0) * p["Fit"] * p["Pdc"], -lim, lim)
    Qit = np.where(tel["ups_state"] != 0, 0.0, Qit)
    return Qit + Qcool + Qaux


def raw_residuals(tel, k):
    p = DC_PARAMS[k]
    V = tel["Va_pu"]
    r_bal = tel["P_total_MW"] - (tel["P_it_MW"] + tel["P_cool_MW"] + tel["P_aux_MW"])
    r_term = tel["P_meas_MW"] - tel["P_total_MW"]
    Vdn = np.sqrt(np.clip(tel["P_aux_MW"] / (p["Faux"] * p["Pdc"]), 0.0, None))
    r_aux = V - Vdn
    den = np.maximum((1.0 - p["Fvfd"]) * Vdn ** 2 + p["Fvfd"], 1e-6)
    Pcf = tel["P_cool_MW"] / den
    ref = p["Fcool"] * p["Pdc"] * (COOL_A + COOL_B * tel["P_ai_pu"])
    a = CFG["DT"] / p["tau_cl"]
    r_cool = np.zeros_like(Pcf)
    r_cool[1:] = Pcf[1:] - Pcf[:-1] - (ref[:-1] - Pcf[:-1]) * a
    if LEGACY_UPS:
        r_ups = np.where(tel["ups_state"] != 0, tel["P_it_MW"], 0.0)
    else:
        r_ups = ups_check(tel, k)
    r_qp = tel["Q_total_Mvar"] - q_hat(tel, p, Vdn, Pcf)
    return dict(r_bal=r_bal, r_aux=r_aux, r_cool=r_cool, r_ups=r_ups, r_qp=r_qp, r_term=r_term, Vdn=Vdn, Pcf=Pcf)


def ups_check(tel, k, return_parts=False):
    """REV: UPS-state consistency check in pu of F_it*P_dc (exactly 0 on clean data, see tests/).
    a: battery/diesel reported (m != 0) but the rectifier still draws grid power, ignored for UPS_G s after any
       reported change of m (genuine transfer transient);
    x: the reported non-grid episode started while the bus voltage stayed above the site's trip setting
       (min V over the last UPS_VWIN s > Vtrip + UPS_DV): a transfer the grid does not explain. Held for the episode.
    r_ups = max(a, x)."""
    p = DC_PARAMS[k]
    m = tel["ups_state"] != 0
    V = tel["Va_pu"]
    n = len(m)
    dt = CFG["DT"]
    G = int(round(UPS_G / dt))
    nw = int(round(UPS_VWIN / dt))
    a = np.zeros(n)
    x = np.zeros(n)
    last_change = -10 ** 9
    flag = 0.0
    base = p["Fit"] * p["Pdc"]
    vlim = VTRIP[k - 1] + UPS_DV
    for i in range(n):
        if i > 0 and m[i] != m[i - 1]:
            last_change = i
            if m[i]:
                flag = 1.0 if float(np.min(V[max(0, i - nw):i + 1])) > vlim else 0.0
        if not m[i]:
            flag = 0.0
        if m[i] and i - last_change >= G:
            a[i] = tel["P_it_MW"][i] / base
        x[i] = flag
    r = np.maximum(a, x)
    return (r, a, x) if return_parts else r


def robust_scale(x):
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    sd = 1.4826 * mad
    if not np.isfinite(sd) or sd <= 0:
        sd = float(np.std(x))
    return med, sd


def calibrate_client(k, scen=None):
    scen = scen or CFG["CALIB_SCENARIO"]
    tel = telemetry(scen, k)
    R = raw_residuals(tel, k)
    s_k = float(np.max(tel["P_total_MW"]))
    dV = rolling_range(tel["Va_pu"], CFG["CSEC_EVWIN"])
    dP = rolling_range(tel["P_total_MW"] / s_k, CFG["CSEC_EVWIN"])
    hi = np.quantile(dP, 0.99)
    S_k = float(np.quantile(dV, 0.99) / max(hi, 1e-9))
    aux_off = float(np.median(R["r_aux"]))
    pz_cal = tel["P_total_MW"] / s_k
    edges = np.quantile(pz_cal, [1 / 3, 2 / 3])
    buckets = np.digitize(pz_cal, edges)
    centroids = np.array([float(np.median(pz_cal[buckets == b])) for b in range(3)])
    cal = dict(s_k=s_k, S_k=S_k, aux_off=aux_off, reg_centroids=centroids, reg_edges=edges,
               dV_trig=float(np.quantile(dV, CFG["CSEC_QTRIG"])),
               v_mu=float(np.mean(tel["Va_pu"])), v_sd=float(np.std(tel["Va_pu"]) + 1e-12))
    stats = {}
    scaled = _scale_residuals(R, tel, k, cal)
    for c in FEAT_CH:
        med, sd = robust_scale(scaled[c])
        stats[c] = (med, max(sd, 1e-9))
    if not LEGACY_UPS:                     # REV: exact check, fixed physical scale (S01 has no transfer)
        stats["r_ups"] = (0.0, UPS_SCALE)
    cal["stats"] = stats
    return cal


def _scale_residuals(R, tel, k, cal):
    s = cal["s_k"]
    out = {"r_bal": R["r_bal"] / s, "r_aux": R["r_aux"] - cal["aux_off"], "r_cool": R["r_cool"] / s,
           "r_ups": R["r_ups"] / s if LEGACY_UPS else R["r_ups"], "r_qp": R["r_qp"] / s}
    dV = rolling_range(tel["Va_pu"], CFG["CSEC_EVWIN"])
    dP = rolling_range(tel["P_total_MW"] / s, CFG["CSEC_EVWIN"])
    out["r_pv"] = dV - cal["S_k"] * dP
    pz = tel["P_total_MW"] / s
    out["r_reg"] = np.min(np.abs(pz[:, None] - cal["reg_centroids"][None, :]), axis=1)
    out["v_z"] = (tel["Va_pu"] - cal["v_mu"]) / cal["v_sd"]
    out["p_z"] = pz
    out["q_z"] = tel["Q_total_Mvar"] / s
    return out


Z_CLIP = 50.0


def features(tel, k, cal):
    R = raw_residuals(tel, k)
    sc = _scale_residuals(R, tel, k, cal)
    X = np.empty((len(tel["Va_pu"]), N_CH), np.float32)
    for j, c in enumerate(FEAT_CH):
        mu, sd = cal["stats"][c]
        X[:, j] = np.clip((sc[c] - mu) / sd, -Z_CLIP, Z_CLIP)
    return X, R


def cal_to_json(cal):
    out = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in cal.items() if k != "stats"}
    out["stats"] = {c: list(v) for c, v in cal["stats"].items()}
    return out


def cal_from_json(d):
    cal = dict(d)
    cal["reg_centroids"] = np.asarray(d["reg_centroids"])
    cal["reg_edges"] = np.asarray(d["reg_edges"])
    cal["stats"] = {c: tuple(v) for c, v in d["stats"].items()}
    return cal


# ----------------------------------------------------------------------------- section 7
LBL_CLEAN, LBL_ATTACK, LBL_RECOVERY = 0, 1, -1


def _windows(rng, n_samp):
    n = int(rng.integers(CFG["ATK_NWIN"][0], CFG["ATK_NWIN"][1] + 1))
    T = CFG["T_END"]
    out = []
    for _ in range(n):
        dur = float(rng.uniform(*CFG["ATK_DUR"]))
        t0 = float(rng.uniform(CFG["ATK_TMIN"], max(CFG["ATK_TMIN"] + 1.0, T - dur - 1.0)))
        out.append((t0, t0 + dur))
    return sorted(out)


def _idx(t0, t1):
    T = tvec()
    i0 = int(np.searchsorted(T, t0))
    i1 = int(np.searchsorted(T, t1))
    return max(i0, 1), min(max(i1, i0 + 2), len(T))


def _profile(atype, rng, n, x_ref):
    tau = np.linspace(0.0, 1.0, n)
    if atype == 1:
        a = float(rng.uniform(*CFG["ATK_MAG"][1])) * (1 if rng.random() < 0.5 else -1)
        return 1.0 + a * np.ones(n), None
    if atype == 2:
        a = float(rng.uniform(*CFG["ATK_MAG"][2])) * (1 if rng.random() < 0.5 else -1)
        return 1.0 + a * tau, None
    if atype == 3:
        return None, None
    if atype == 4:
        sd = CFG["ATK_MAG"][4] * float(np.std(x_ref) + 1e-9)
        add = rng.normal(0.0, sd, n)
        imp = rng.random(n) < 0.01
        add[imp] += x_ref[imp] * (rng.uniform(-0.6, 0.6, imp.sum()))
        return None, add
    a = float(rng.uniform(*CFG["ATK_MAG"][5]))
    return 1.0 + a * np.sin(np.pi * tau), None


def _apply_channel(x, i0, i1, atype, rng, dt_replay):
    seg = x[i0:i1]
    n = len(seg)
    if atype == 3:
        shift = int(round(dt_replay / CFG["DT"]))
        j0 = max(0, i0 - shift)
        rep = x[j0:j0 + n]
        if len(rep) < n:
            rep = np.pad(rep, (0, n - len(rep)), mode="edge")
        x[i0:i1] = rep
        return
    m, add = _profile(atype, rng, n, seg)
    if m is not None:
        x[i0:i1] = seg * m
    if add is not None:
        x[i0:i1] = seg + add


GRID_CH = ["Va_pu", "P_total_MW", "Q_total_Mvar"]
FAC_CH = ["P_it_MW", "P_cool_MW", "P_aux_MW"]


def inject_T1_T2(scen, k, regime, seed):
    rng = np.random.default_rng(seed)
    tel = telemetry(scen, k)
    T = tvec()
    lab = np.zeros(len(T), np.int8)
    chans = GRID_CH if regime == "T1" else FAC_CH
    log = []
    for (t0, t1) in _windows(rng, len(T)):
        i0, i1 = _idx(t0, t1)
        atype = int(rng.integers(1, 6))
        dtr = float(rng.uniform(*CFG["ATK_MAG"][3]))
        sub = [c for c in chans if rng.random() < 0.8] or [chans[1]]
        for c in sub:
            _apply_channel(tel[c], i0, i1, atype, np.random.default_rng(rng.integers(1 << 30)), dtr)
        if regime == "T1":
            tel["P_meas_MW"][i0:i1] = tel["P_total_MW"][i0:i1]
        lab[i0:i1] = LBL_ATTACK
        log.append(dict(scenario=scen, dc=k, regime=regime, atype=atype,
                        t0=float(T[i0]), t1=float(T[i1 - 1]), channels=";".join(sub)))
    return tel, lab, log


def inject_T3(scen, k, seed, CAL, forge_voltage=True):
    p = DC_PARAMS[k]
    rng = np.random.default_rng(seed)
    tel = telemetry(scen, k)
    R0 = raw_residuals(tel, k)
    Pcf0, V0 = R0["Pcf"].copy(), R0["Vdn"].copy()
    Pai0, Pit0 = tel["P_ai_pu"].copy(), tel["P_it_MW"].copy()
    T = tvec()
    n = len(T)
    lab = np.zeros(n, np.int8)
    mult = np.ones(n)
    log = []
    for (t0, t1) in _windows(rng, n):
        i0, i1 = _idx(t0, t1)
        atype = int(rng.integers(1, 6))
        seg = Pai0[i0:i1]
        if atype == 3:
            shift = int(round(float(rng.uniform(*CFG["ATK_MAG"][3])) / CFG["DT"]))
            j0 = max(0, i0 - shift)
            rep = Pai0[j0:j0 + (i1 - i0)]
            if len(rep) < (i1 - i0):
                rep = np.pad(rep, (0, (i1 - i0) - len(rep)), mode="edge")
            m = rep / np.maximum(seg, 1e-6)
        else:
            m, add = _profile(atype, rng, i1 - i0, seg)
            if m is None:
                m = 1.0 + add / np.maximum(seg, 1e-6)
        mult[i0:i1] = m
        lab[i0:i1] = LBL_ATTACK
        i_rec = min(n, i1 + int(CFG["RECOVERY_S"] / CFG["DT"]))
        lab[i1:i_rec] = np.where(lab[i1:i_rec] == LBL_CLEAN, LBL_RECOVERY, lab[i1:i_rec])
        log.append(dict(scenario=scen, dc=k, regime="T3" if forge_voltage else "T3a",
                        atype=atype, t0=float(T[i0]), t1=float(T[i1 - 1]), channels="zA+zB"))
    if not log:
        return tel, lab, log
    # REV: forge only the deviation from the clean record, so every report equals the clean stream
    # before the first attack interval. Outside the intervals only the lagged cooling deviation remains,
    # and it decays inside the labelled 30-s recovery tail.
    clean = {c: v.copy() for c, v in tel.items()}
    Pai = np.where(lab == LBL_ATTACK, np.clip(Pai0 * mult, 0.03, 1.0), Pai0)
    Pit = np.where(tel["ups_state"] != 0, Pit0, Pit0 * Pai / np.maximum(Pai0, 1e-9))
    dref = p["Fcool"] * p["Pdc"] * COOL_B * (Pai - Pai0)
    a = CFG["DT"] / p["tau_cl"]
    dPcf = np.zeros(n)
    for i in range(1, n):
        dPcf[i] = dPcf[i - 1] + (dref[i - 1] - dPcf[i - 1]) * a
    Pcf = Pcf0 + dPcf
    Pcool0, Paux0 = clean["P_cool_MW"], clean["P_aux_MW"]

    def _pi(V):
        return (1.0 - p["Fvfd"]) * V ** 2 + p["Fvfd"]

    Vm = V0.copy()
    for _ in range(3 if forge_voltage else 1):
        Paux = Paux0 + p["Faux"] * p["Pdc"] * (Vm ** 2 - V0 ** 2)
        Pcool = Pcool0 + Pcf * _pi(Vm) - Pcf0 * _pi(V0)
        dP = (Pit - Pit0) + (Pcool - Pcool0) + (Paux - Paux0)
        if forge_voltage:
            Vm = V0 + CAL[k]["S_k"] * dP / CAL[k]["s_k"]
    Paux = Paux0 + p["Faux"] * p["Pdc"] * (Vm ** 2 - V0 ** 2)
    Pcool = Pcool0 + Pcf * _pi(Vm) - Pcf0 * _pi(V0)
    dP = (Pit - Pit0) + (Pcool - Pcool0) + (Paux - Paux0)
    dQ = q_hat(dict(tel, P_it_MW=Pit), p, Vm, Pcf) - q_hat(clean, p, V0, Pcf0)
    tel["Va_pu"] = clean["Va_pu"] + (Vm - V0)
    tel["P_it_MW"], tel["P_cool_MW"], tel["P_aux_MW"] = Pit, Pcool, Paux
    tel["P_total_MW"] = clean["P_total_MW"] + dP
    tel["P_meas_MW"] = clean["P_meas_MW"] + dP
    tel["Q_total_Mvar"] = clean["Q_total_Mvar"] + dQ
    tel["P_ai_pu"] = Pai
    tel["P_it_served_MW"] = clean["P_it_served_MW"] * Pai / np.maximum(Pai0, 1e-9)
    return tel, lab, log


def inject_U1(scen, k, seed):
    """REV (A3): UPS-state spoofing, false battery report. Over 2-4 random intervals (same interval rule
    as T1/T2) the reported state m is set to 1 wherever the true state is 0. No other channel changes.
    Expected detector evidence: r_ups = P_it > 0 and the reactive check (Q_it is taken as 0 when m != 0)."""
    rng = np.random.default_rng(seed)
    tel = telemetry(scen, k)
    T = tvec()
    lab = np.zeros(len(T), np.int8)
    log = []
    for (t0, t1) in _windows(rng, len(T)):
        i0, i1 = _idx(t0, t1)
        seg = tel["ups_state"][i0:i1]
        tel["ups_state"][i0:i1] = np.where(seg == 0, 1.0, seg)
        lab[i0:i1] = LBL_ATTACK
        log.append(dict(scenario=scen, dc=k, regime="U1", atype=0, t0=float(T[i0]), t1=float(T[i1 - 1]),
                        channels="ups_state"))
    return tel, lab, log


def hide_transfer(tel, scen, k):
    """REV (A3): UPS-state spoofing, hidden transfer (U2). During a control-path interval at site k the
    reported state is forced to 0 while the UPS is actually on battery or diesel. Used with S07 and S09."""
    spec = CONTROL_PATH.get(scen)
    if spec is None or k not in spec["targets"]:
        raise ValueError(f"U2 needs a control-path scenario that targets DC{k}")
    i0, i1 = _idx(spec["t0"], spec["t1"])
    tel["ups_state"][i0:i1] = 0.0
    return tel


def inject_T3_legacy(scen, k, seed, CAL, forge_voltage=True):
    """Original generator (rebuilds reports over the whole record). Ablation only: FGPT_LEGACY_T3=1."""
    p = DC_PARAMS[k]
    rng = np.random.default_rng(seed)
    tel = telemetry(scen, k)
    R0 = raw_residuals(tel, k)
    Pcf0, V0 = R0["Pcf"].copy(), R0["Vdn"].copy()
    Pai0, Pit0 = tel["P_ai_pu"].copy(), tel["P_it_MW"].copy()
    T = tvec()
    n = len(T)
    lab = np.zeros(n, np.int8)
    mult = np.ones(n)
    log = []
    for (t0, t1) in _windows(rng, n):
        i0, i1 = _idx(t0, t1)
        atype = int(rng.integers(1, 6))
        seg = Pai0[i0:i1]
        if atype == 3:
            shift = int(round(float(rng.uniform(*CFG["ATK_MAG"][3])) / CFG["DT"]))
            j0 = max(0, i0 - shift)
            rep = Pai0[j0:j0 + (i1 - i0)]
            if len(rep) < (i1 - i0):
                rep = np.pad(rep, (0, (i1 - i0) - len(rep)), mode="edge")
            m = rep / np.maximum(seg, 1e-6)
        else:
            m, add = _profile(atype, rng, i1 - i0, seg)
            if m is None:
                m = 1.0 + add / np.maximum(seg, 1e-6)
        mult[i0:i1] = m
        lab[i0:i1] = LBL_ATTACK
        i_rec = min(n, i1 + int(CFG["RECOVERY_S"] / CFG["DT"]))
        lab[i1:i_rec] = np.where(lab[i1:i_rec] == LBL_CLEAN, LBL_RECOVERY, lab[i1:i_rec])
        log.append(dict(scenario=scen, dc=k, regime="T3" if forge_voltage else "T3a",
                        atype=atype, t0=float(T[i0]), t1=float(T[i1 - 1]), channels="zA+zB"))
    if not log:
        return tel, lab, log
    Pai = np.clip(Pai0 * mult, 0.03, 1.0)
    ratio = Pai / np.maximum(Pai0, 1e-9)
    Pit = np.where(tel["ups_state"] != 0, Pit0, Pit0 * ratio)
    ref = p["Fcool"] * p["Pdc"] * (COOL_A + COOL_B * Pai)
    a = CFG["DT"] / p["tau_cl"]
    Pcf = np.empty(n)
    Pcf[0] = Pcf0[0]
    for i in range(1, n):
        Pcf[i] = Pcf[i - 1] + (ref[i - 1] - Pcf[i - 1]) * a
    V = V0.copy()
    for _ in range(3):
        Paux = p["Faux"] * p["Pdc"] * V ** 2
        Pcool = Pcf * ((1.0 - p["Fvfd"]) * V ** 2 + p["Fvfd"])
        Ptot = Pit + Pcool + Paux
        if not forge_voltage:
            break
        V = V0 + CAL[k]["S_k"] * (Ptot - tel["P_total_MW"]) / CAL[k]["s_k"]
    Paux = p["Faux"] * p["Pdc"] * V ** 2
    Pcool = Pcf * ((1.0 - p["Fvfd"]) * V ** 2 + p["Fvfd"])
    Ptot = Pit + Pcool + Paux
    tel["Va_pu"] = V + CAL[k]["aux_off"]
    tel["P_it_MW"], tel["P_cool_MW"], tel["P_aux_MW"] = Pit, Pcool, Paux
    tel["P_total_MW"], tel["P_meas_MW"] = Ptot, Ptot
    tel["P_ai_pu"] = Pai
    tel["P_it_served_MW"] = np.where(tel["ups_state"] == 3, 0.0, p["Fit"] * p["Pdc"] * Pai)
    tel["Q_total_Mvar"] = q_hat(tel, p, np.sqrt(np.clip(Paux / (p["Faux"] * p["Pdc"]), 0, None)), Pcf)
    return tel, lab, log


REGIMES = ["T1", "T2", "T3a", "T3"]


def make_attacked(scen, k, regime, seed, CAL):
    if regime in ("T1", "T2"):
        return inject_T1_T2(scen, k, regime, seed)
    if regime == "U1":
        return inject_U1(scen, k, seed)
    if os.environ.get("FGPT_LEGACY_T3") == "1":
        return inject_T3_legacy(scen, k, seed, CAL, forge_voltage=(regime == "T3"))
    return inject_T3(scen, k, seed, CAL, forge_voltage=(regime == "T3"))


def notebook_attack_seed(base_seed, scen, k):
    """build_winsets(): seed + 1000*k + 7*SCENARIOS.index(scen)."""
    return base_seed + 1000 * k + 7 * SCENARIOS.index(scen)


# ----------------------------------------------------------------------------- section 9
def window_starts(n):
    return np.arange(0, n - CFG["WIN"] - 1, CFG["HOP"], dtype=np.int64)


def eval_mask(ywin, t_end):
    """REV: windows used for reporting: labelled clean (0) or attack (1), and ending at or after WARMUP_S."""
    return (np.asarray(ywin) >= 0) & (np.asarray(t_end) >= CFG["WARMUP_S"])


def window_labels(lab, starts):
    W = CFG["WIN"]
    out = np.zeros(len(starts), np.int8)
    for j, s in enumerate(starts):
        seg = lab[s:s + W]
        na = int((seg == LBL_ATTACK).sum())
        nr = int((seg == LBL_RECOVERY).sum())
        if na >= CFG["LABEL_FRAC"] * W:
            out[j] = 1
        elif na == 0 and nr == 0:
            out[j] = 0
        else:
            out[j] = -1
    return out


RESID_IDX = np.arange(len(RESID_CH))
SIG_IDX = np.arange(len(RESID_CH), N_CH)


# REV (offline comparison): channels fed to the TCN. None = all N_CH channels (FedGuard-PT);
# FGPT_TCN_INPUT=raw -> only the three signal channels (raw-signal variant; targets and heads unchanged).
TCN_INPUT = SIG_IDX if os.environ.get("FGPT_TCN_INPUT") == "raw" else None


def tcn_in(x):
    return x if TCN_INPUT is None else x[..., TCN_INPUT]


class WinSet:
    def __init__(self, X, lab, k, reg_edges, pz_stats=None):
        """REV: the auxiliary class is assigned on the raw P/s_k scale of reg_edges. X holds the
        standardized p_z channel, so it is de-standardized with pz_stats=(median, scale) first.
        pz_stats=None reproduces the original (scale-mismatched) behaviour."""
        self.X, self.k = X, k
        self.starts = window_starts(len(X))
        self.y = window_labels(lab, self.starts)
        self.end = self.starts + CFG["WIN"] - 1
        pz = X[self.end, SIG_IDX[1]].astype(np.float64)
        if pz_stats is not None:
            pz = pz * pz_stats[1] + pz_stats[0]
        self.reg = np.digitize(pz, reg_edges).astype(np.int64)

    def arrays(self, idx):
        """EDGE: same as WinSet.tensors() but NumPy, windows as (B, L, C)."""
        W = CFG["WIN"]
        xb = np.stack([self.X[s:s + W] for s in self.starts[idx]])
        nxt = np.stack([self.X[s + W] for s in self.starts[idx]])
        return tcn_in(xb), nxt[:, SIG_IDX].copy(), nxt[:, RESID_IDX].copy(), self.reg[idx]


def winset(X, lab, k, cal):
    """REV: WinSet with the auxiliary class on the correct scale."""
    return WinSet(X, lab, k, cal["reg_edges"], cal["stats"]["p_z"])


class Merged(WinSet):
    def __init__(self, sets):
        self.k = sets[0].k
        self.X = np.concatenate([s.X for s in sets], 0)
        off, starts, ys, regs = 0, [], [], []
        for s in sets:
            starts.append(s.starts + off)
            ys.append(s.y)
            regs.append(s.reg)
            off += len(s.X)
        self.starts = np.concatenate(starts)
        self.y = np.concatenate(ys)
        self.reg = np.concatenate(regs)
        self.end = self.starts + CFG["WIN"] - 1


CONTROL_PATH = {
    "S07_atk_false_ups":    dict(targets=[1, 4], t0=140.0, t1=170.0, kind="false UPS transfer", code=1),
    "S08_atk_load_alter":   dict(targets=[2, 5], t0=100.0, t1=160.0, kind="load altering", code=2),
    "S09_atk_sensor_spoof": dict(targets=[3],    t0=180.0, t1=220.0, kind="voltage sensor spoof", code=4),
    "S10_atk_cooling":      dict(targets=[6],    t0=200.0, t1=250.0, kind="cooling setpoint", code=3),
    "S11_fault_plus_atk":   dict(targets=[6],    t0=62.0,  t1=92.0,  kind="load altering + fault", code=2),
}


def control_labels(scen, k):
    """0 clean, 1 attacked, -1 recovery tail (notebook section 8)."""
    lab = np.zeros(len(tvec()), np.int8)
    spec = CONTROL_PATH.get(scen)
    if spec is None or k not in spec["targets"]:
        return lab
    i0, i1 = _idx(spec["t0"], spec["t1"])
    lab[i0:i1] = LBL_ATTACK
    i_rec = min(len(tvec()), i1 + int(CFG["RECOVERY_S"] / CFG["DT"]))
    lab[i1:i_rec] = LBL_RECOVERY
    return lab


# ----------------------------------------------------------------------------- section 12 (CSEC)
DESCRIPTOR_FIELDS = ("t_onset", "sign", "depth", "tau_rec", "rho")
DESCRIPTOR_BYTES = 2 + 1 + 4 + 4 + 4


def detect_events(tel, k, cal):
    T = tvec()
    V = tel["Va_pu"]
    s = cal["s_k"]
    dV = rolling_range(V, CFG["CSEC_EVWIN"])
    dP = rolling_range(tel["P_total_MW"] / s, CFG["CSEC_EVWIN"])
    active = dV > cal["dV_trig"]
    eid = np.full(len(V), -1, np.int32)
    evs = []
    i = 0
    while i < len(V):
        if not active[i]:
            i += 1
            continue
        j = i
        while j < len(V) and active[j]:
            j += 1
        pk = i + int(np.argmax(dV[i:j]))
        depth = float(dV[pk])
        dpw = float(dP[pk])
        tail = V[pk:min(len(V), pk + 2000)]
        v0, vinf = float(V[pk]), float(np.median(tail[-200:])) if len(tail) > 200 else float(V[pk])
        thr = v0 + 0.63 * (vinf - v0)
        rec = np.flatnonzero((tail - thr) * np.sign(vinf - v0 + 1e-12) >= 0)
        tau_rec = float(rec[0] * CFG["DT"]) if len(rec) else float(len(tail) * CFG["DT"])
        evs.append(dict(dc=k, idx0=i, idx1=j, t_onset=float(T[i]),
                        sign=int(np.sign(float(V[pk]) - float(np.median(V)))),
                        depth=depth, tau_rec=tau_rec, rho=float(depth / max(dpw, 1e-9))))
        eid[i:j] = len(evs) - 1
        i = j
    return evs, eid


def csec_signals(tels, cals):
    n = len(tvec())
    E, EID = {}, {}
    for k, tel in tels.items():
        E[k], EID[k] = detect_events(tel, k, cals[k])
    corr = {k: np.zeros(n, np.float32) for k in tels}
    unc = {k: np.zeros(n, np.float32) for k in tels}
    others = sorted(tels)
    log = []
    for k in others:
        for e in E[k]:
            peers = []
            for j in others:
                if j == k:
                    continue
                for f in E[j]:
                    if abs(f["t_onset"] - e["t_onset"]) <= CFG["CSEC_DTALIGN"]:
                        peers.append((j, f))
                        break
            c = len(peers) / max(len(others) - 1, 1)
            depth_ratio = 0.0
            if peers:
                depth_ratio = float(np.clip(np.median([f["depth"] for _, f in peers]) / max(e["depth"], 1e-12), 0.0, 1.0))
            rho = np.nan
            if len(peers) >= 1:
                group = [(k, e)] + peers
                depths = np.array([f["depth"] for _, f in group])
                epi = group[int(np.argmax(depths))][0]
                dist = np.array([DCDIST[epi - 1, j - 1] for j, _ in group])
                if len(group) >= 3 and np.ptp(depths) > 0 and np.ptp(dist) > 0:
                    rho = float(spearmanr(depths, -dist).statistic)
                else:
                    rho = 1.0
            consistent = (len(peers) >= 1) and (not np.isnan(rho)) and (rho >= CFG["CSEC_RHOMIN"])
            corr[k][e["idx0"]:e["idx1"]] = (c * depth_ratio) if consistent else 0.0
            unc[k][e["idx0"]:e["idx1"]] = 0.0 if consistent else 1.0
            log.append(dict(dc=k, t_onset=e["t_onset"], depth=e["depth"], tau_rec=e["tau_rec"],
                            rho_pv=e["rho"], n_peers=len(peers), spearman=rho,
                            depth_ratio=depth_ratio, corroborated=bool(consistent)))
    return corr, unc, pd.DataFrame(log)


def csec_window(sig, starts):
    W = CFG["WIN"]
    c = np.cumsum(np.concatenate([[0.0], sig.astype(np.float64)]))
    return ((c[starts + W] - c[starts]) / W).astype(np.float32)


# ----------------------------------------------------------------------------- section 13 (fusion)
TERMS = [f"|{c}|" for c in RESID_CH] + ["e_rec", "e_fc"]


def fusion_ms(fusion, k):
    """REV: per-site term statistics (median, 1.4826 MAD) if the bundle has them, else the pooled ones."""
    by = fusion.get("ms_by_dc")
    if by is not None:
        d = by[str(k)]
        return np.asarray(d["mu"], np.float32), np.asarray(d["sd"], np.float32)
    return np.asarray(fusion["ms_mu"], np.float32), np.asarray(fusion["ms_sd"], np.float32)


def z_terms(T, ms):
    return np.clip((T - ms[0]) / ms[1], -Z_CLIP, Z_CLIP)


CHECK_TERMS = ["|r_bal|", "|r_aux|", "|r_cool|", "|r_ups|", "|r_qp|"]   # the five physical checks
# terms whose fusion weight is bounded at >= 0 (FGPT_BOUND_TERMS="|r_ups|" bounds only the UPS check)
BOUND_TERMS = [t for t in os.environ.get("FGPT_BOUND_TERMS", ";".join(CHECK_TERMS)).split(";") if t]
FUSION_NONNEG = os.environ.get("FGPT_LEGACY_FUSION") != "1"


class _Fusion:
    def __init__(self, w, b):
        self.coef_ = np.asarray(w, float)[None, :]
        self.intercept_ = np.array([float(b)])


def fit_fusion(Z, y, seed=CFG["SEED"], nonneg=None, C=1.0):
    """Logistic fusion, L2 penalty C=1 on the weights, intercept unpenalized (same objective as sklearn).
    REV: weights of the five physical-check terms are bounded at >= 0, so a larger physical violation can
    never lower the score. With no bound active the result equals sklearn's (tests/check_fusion.py).
    FGPT_LEGACY_FUSION=1 or nonneg=False gives the original unconstrained sklearn fit."""
    nonneg = FUSION_NONNEG if nonneg is None else nonneg
    if not nonneg:
        lr = LogisticRegression(max_iter=2000, C=C, random_state=seed)
        lr.fit(Z, y)
        return lr
    from scipy.optimize import minimize
    Z = np.asarray(Z, np.float64)
    ys = np.where(np.asarray(y) == 1, 1.0, -1.0)
    d = Z.shape[1]

    def f(theta):
        w, b = theta[:d], theta[d]
        z = ys * (Z @ w + b)
        loss = np.logaddexp(0.0, -z).sum()
        g = -ys * 0.5 * (1.0 - np.tanh(0.5 * z))           # d loss / d(Xw+b), numerically stable sigmoid(-z)*(-y)
        return C * loss + 0.5 * w @ w, np.r_[C * (Z.T @ g) + w, C * g.sum()]

    bounds = [(0.0, None) if t in BOUND_TERMS else (None, None) for t in TERMS] + [(None, None)]
    if len(bounds) != d + 1:
        raise ValueError("term count does not match TERMS")
    r = minimize(f, np.zeros(d + 1), jac=True, method="L-BFGS-B", bounds=bounds,
                 options=dict(maxiter=20000, maxfun=40000, ftol=1e-15, gtol=1e-10))
    if not r.success:
        print("[fusion] L-BFGS-B:", r.message)
    return _Fusion(r.x[:d], r.x[d])


# REV: physics layer. The UPS check is exactly 0 on clean data and the fusion has no validation evidence to weight it
# (validation attacks never change m), so it acts as a fixed rule outside the learned fusion: a window whose UPS-check
# term exceeds RULE_Z (mean violation above RULE_Z * UPS_SCALE = 0.05 pu of F_it*P_dc) is raised by RULE_BOOST, which
# puts it above any threshold and at the top of the ranking. The rule has no learned parameter.
RULE_TERM, RULE_Z, RULE_BOOST = "|r_ups|", 1.0, 1000.0


def default_rule(enabled=True):
    return dict(enabled=bool(enabled), term=RULE_TERM, z=RULE_Z, boost=RULE_BOOST)


def physics_rule(Z, rule):
    """Boolean per window: the physics rule fires. rule=None or disabled -> never fires (older bundles)."""
    Z = np.atleast_2d(Z)
    if not rule or not rule.get("enabled"):
        return np.zeros(len(Z), bool)
    return Z[:, TERMS.index(rule["term"])] > rule["z"]


def fused_score(coef, intercept, Z, corrw=None, uncw=None, kappa=0.0, eta=0.0, rule=None):
    s = Z @ np.asarray(coef).ravel() + float(intercept)
    if corrw is not None:
        s = s - kappa * corrw + eta * uncw
    if rule:
        s = s + rule["boost"] * physics_rule(Z, rule)
    return s


def best_f1_threshold(s, y):
    qs = np.quantile(s, np.linspace(0.01, 0.999, 300))
    f1 = [f1_score(y, (s > t).astype(int), zero_division=0) for t in qs]
    j = int(np.argmax(f1))
    return float(qs[j]), float(f1[j])


LAST_GRID = []   # (kappa, eta, F1) of the last tune_csec call, saved to fusion.json as "grid_f1"


def tune_csec(coef, intercept, Z, y, corrw, uncw, rule=None):
    best = (0.0, 0.0, -1.0, None)
    LAST_GRID.clear()
    for kap in CFG["KAPPA_GRID"]:
        for eta in CFG["ETA_GRID"]:
            s = fused_score(coef, intercept, Z, corrw, uncw, kap, eta, rule)
            thr, f1 = best_f1_threshold(s, y)
            LAST_GRID.append((float(kap), float(eta), float(f1)))
            if f1 > best[2]:
                best = (kap, eta, f1, thr)
    return best


def explain(Z, coef, names=TERMS):
    contrib = Z * np.asarray(coef).ravel()[None, :]
    j = np.argmax(contrib, 1)
    dom = np.array(names, dtype=object)[j]
    alg_idx = [names.index(f"|{a}|") for a in ("r_bal", "r_aux", "r_qp")]
    alg_fired = (Z[:, alg_idx] > 3.0).any(1)
    cs = np.where(alg_fired, "single-domain", "cross-domain")
    return dom, cs, contrib


def metrics_from(scores, y, thr):
    out = {}
    if len(np.unique(y)) > 1:
        out["roc_auc"] = float(roc_auc_score(y, scores))
        out["pr_auc"] = float(average_precision_score(y, scores))
    else:
        out["roc_auc"] = out["pr_auc"] = float("nan")
    pred = (scores > thr).astype(int)
    out["f1"] = float(f1_score(y, pred, zero_division=0))
    out["precision"] = float(precision_score(y, pred, zero_division=0))
    out["recall"] = float(recall_score(y, pred, zero_division=0))
    neg = (y == 0)
    out["fpr"] = float(pred[neg].mean()) if neg.any() else float("nan")
    out["n_pos"], out["n_neg"] = int((y == 1).sum()), int(neg.sum())
    return out


def detection_delay(scores, y, thr, tw):
    on = np.flatnonzero(y == 1)
    if len(on) == 0:
        return float("nan")
    d = []
    for r in np.split(on, np.flatnonzero(np.diff(on) > 1) + 1):
        fired = np.flatnonzero(scores[r] > thr)
        d.append(tw[r[fired[0]]] - tw[r[0]] if len(fired) else np.nan)
    d = np.array(d, float)
    return float(np.nanmean(d)) if np.isfinite(d).any() else float("nan")
