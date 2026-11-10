# FedGuard-PT revision experiment on the Jetson Orin Nano

This folder is the revised board pipeline for the FedGuard-PT manuscript. It replays the existing
Simulink dataset (IEEE 39-bus, six data centers, eleven 300-s scenarios). No new simulation is needed.
Every DC1 monitor stream in the `main` plan is replayed at **100 Hz wall clock**. This is a
hardware-in-the-loop replay of simulated telemetry, not field measurement.

## 1. What changed and which review comment it answers

| ID | Change | Files | Reviewer comment / limitation |
|---|---|---|---|
| A1 | T3/T3a forge only the deviation from the clean record. Reports equal the clean stream before the first attack; balance, auxiliary and reactive invariants are kept inside the attack. | `fgpt/nb.py` `inject_T3` | "T3 and T3a have identical FPR… samples outside the attack intervals" |
| A2 | Inner validation: each fold validates on a disturbance scenario that is not the held-out one and is removed from that fold's training (`CFG["VAL_SCENARIO"]`). CSEC grids widened to λc ∈ {0,1,2,4,8,16,32}, λu ∈ {0,0.5,1,…,1024} (powers of 2). | `train_edge.py`, `fgpt/nb.py` | "Validation uses S01, which has no grid disturbance… S01 is in the training data" |
| A3 | UPS-state spoofing. **U1**: reported state m set to 1 while the UPS is on grid (2–4 random intervals at the monitored site). **U2** (`ctrl_hide`): during the S07/S09 control-path transfer, m is reported as 0. U1/U2 are *not* used for fusion fitting, so they test generalisation. | `fgpt/nb.py` `inject_U1`, `hide_transfer`; `rt_monitor.py` | "No measurement attack spoofs m… UPS check never tested" |
| A4 | (i) Auxiliary power class assigned on the raw P/s_k scale (was: standardised p_z against raw edges, 85–90 % of windows fell in one class). (ii) Scoring-term statistics per site with the site's own model (was: DC1's model for every site, pooled). | `fgpt/nb.py` `winset`, `fusion_ms`; `train_edge.py` | "auxiliary-label scale mismatch and pooled fusion calibration" (previous Sec. V) |
| A5 | The 11 LOSO tests at DC2–DC6 (fast replay). | `run_full_experiment.py --plan sites` | "Only DC1 ran live" |
| A6 | FULL-model control-path streams at the attacked sites: S07@DC4, S08@DC2/DC5, S09@DC3, S10@DC6, S11@DC6 (real time). | `--plan attacked` | "S08–S11 describe an unattacked site" |
| A9 | Windows whose CSEC release is clipped at the end of the record are counted; metrics are also reported without them. | `fgpt/csec_rt.py`, `rt_monitor.py` | "end-of-record release clipping" |
| A14 | Sensitivity to meter error and model mismatch: the S02 fold retrained under assumed perturbations (see §6). | `fgpt/nb.py` `FGPT_PERTURB`, `--plan sens` | "idealized clean residual floor" |
| U | UPS check redefined (method change, §9): gated power part + transfer-at-normal-voltage flag, fixed physical scale; fusion weights of the five checks bounded at ≥ 0. | `fgpt/nb.py` `ups_check`, `fit_fusion`; `fgpt/stream.py`; `train_edge.py` | "UPS check never meaningfully tested" |

`k_v` (drive reactive fraction, the former `½`) is now `CFG["K_V"] = 0.5`, matching the DC-CLM v1 Simulink block
(`Q_vfd = Fvfd*Q_cool_base*0.5`).

Unchanged: TCN (3597 parameters, 3376 shared), federation (25 rounds, int8 + error feedback, trimmed mean),
payload (3504 B up / 13 504 B down per site per round), CSEC rules, A1–A5 repository files, control-path scenarios.

## 2. Setup (once)

Place this folder next to the dataset, so the layout is:

```
FedGuard-PT/
  dataset/            clean/ attacked/ attack_log.csv manifest.csv
  fgpt_jetson_rev/    this folder
```

```bash
cd FedGuard-PT/fgpt_jetson_rev
chmod +x *.sh
./setup_jetson.sh                 # apt: numpy pandas scipy sklearn matplotlib psutil tk tmux fonts
./get_data.sh                     # links ../dataset; prints 66 clean + 66 attacked files per site
python3 tests/check_generators.py data/dataset    # must end with "FAILURES: 0" (44 checks)
python3 tests/check_fusion.py                     # must print PASS (bounded fit and physics rule)
```

If `../dataset` is missing, `get_data.sh` sparse-clones it from `KIBRIA-SAROARE/FedGuard-DC`
(override with `REPO_URL=...`).

Keep the board state fixed for the whole run, as in the previous experiment:

- power mode 25 W (`sudo nvpmodel -q`), same fan setting;
- browser and other apps closed;
- run over SSH or on the console; the dashboard is only used by `live_capture.py`.

`run_full_experiment.py` records `nvpmodel -q` and `jetson_clocks --show` in `results_rev/experiment_plan_*.json`.

## 3. Run

```bash
tmux new -s fgpt
./run_experiment.sh quick         # optional first: about 1 h (2 trainings + 6 real-time streams)
./run_experiment.sh               # everything (resumable); Ctrl-b d to detach, tmux attach -t fgpt
```

If the run stops, start the same command again. Finished trainings and streams are skipped.
To run one plan: `./run_experiment.sh main` (or `attacked`, `sites`, `sens`, or `main,attacked`).

| plan | content | mode | duration (estimate) |
|---|---|---|---|
| `main` | idle 60 s; 5 LOSO folds + FULL trained on the board; DC1: 11 tests × 5 folds (`no_attack T1 T2 T3a T3 U1 A1–A5`) + FULL `ctrl` S07–S11 + `ctrl_hide` S07 = 61 streams; latency sweep; idle 60 s | real time | 6 trainings ≈ 70 min + 61 × 318 s ≈ 5.4 h |
| `attacked` | FULL model at the attacked sites, 8 streams | real time | ≈ 45 min |
| `sites` | DC2–DC6 × 5 folds × 11 tests = 275 streams | fast replay | ≈ 1–2 h |
| `sens` | S02 fold retrained for 6 perturbation cases + 11 DC1 tests per case (77 streams) | fast replay | ≈ 70 min + 30 min |
| live | 4 dashboard captures (S02 A1, no_attack, T3; S07) | real time | ≈ 20 min |

Training and stream times are estimates from the previous board run (639–711 s per training, 298 s per stream
+ 20 s cool-down). Fast-replay time on the board was not measured; in the 2-core cloud check below one stream took about 5 s.

Total for `./run_experiment.sh`: roughly 9–11 h.

## 4. Fast replay vs real time

`--no-realtime` streams compute exactly the same scores as real-time streams (the monitor verifies streamed vs batch
scores in every run, `streaming_vs_batch_max_abs_diff`). Only pacing differs, so their timing numbers are not valid.

- `paper_figures.py` uses only real-time streams (`results_rev/streams/`, DC1).
- `revision_summary.py` uses real-time DC1 streams plus fast-replay site and sensitivity streams, and labels the mode in each table.

Report timing (per-sample processing, deadline misses, resources) only from the real-time `main` and `attacked` plans.

## 5. Outputs

```
results_rev/folds/<fold>/bundle/fusion.json        gains, threshold, validation scenario, grid-edge flags, per-site term statistics
results_rev/streams/<fold>/<test>_<scenario>/      DC1 real-time streams (as before)
results_rev/streams_attacked/DC<k>/<test>_<scen>/  control path at the attacked sites
results_rev/streams_sites/DC<k>/<fold>/<test>_<fold>/
results_rev/sens/<case>/folds/... and /streams/...
results_rev/paper/       Fig_rt_*.pdf|png, Table_rt_*.tex|csv, numbers.json      (paper_figures.py)
results_rev/revision/    Table_rev_*.csv|tex, summary.json                       (revision_summary.py)
```

| table | content | manuscript use |
|---|---|---|
| `Table_rev_gains` | λc, λu, τ* per fold, validation scenario, whether a gain sits at the grid edge | Sec. IV-C text, reply to the validation comment |
| `Table_rev_dc1` | DC1 per regime incl. U1, intervals, delays, and metrics without clipped windows | replaces Table II |
| `Table_rev_control` | S07–S11 at DC1 and at the attacked sites; U2 hidden transfer | Sec. IV-E, reply to the UPS comment |
| `Table_rev_sites` | per-site detection DC1–DC6 | new table or text |
| `Table_rev_clipped` | clipped CSEC releases | Sec. V |
| `Table_rev_sensitivity` | meter error / mismatch cases with clean residual floors | Sec. V |

Copy numbers into the paper from these files only.

### 5.1 Warm-up exclusion (first 10 s of every 300-s record)

All reported detection metrics, tables and figures ignore windows that end before `WARMUP_S = 10 s`
(`fgpt/nb.py`, `nb.eval_mask`). Training, validation, calibration and the timing/resource numbers are unchanged
(timing still covers every sample). No attack starts before 20 s, so no attacked window is removed.

New runs apply it directly. For a run that is already finished (no re-run needed):

```bash
python3 recompute_metrics.py --root results_rev          # rewrites metrics.json from windows.csv; originals kept as metrics_before_recompute.json
python3 paper_figures.py --root results_rev
python3 revision_summary.py --root results_rev --dataset data/dataset
python3 ups_rule_analysis.py --root results_rev
```
`python3 recompute_metrics.py --root <copy> --warmup-s 0` reproduces the original files (checked on all 421 cloud streams).

### 5.2 Offline comparison table (Table `tab:offline`)

```bash
python3 offline_comparison.py --root results_rev --out results_cmp     # resumable; train stage then eval stage
```
Rows: centralized (one TCN on pooled windows), FedGuard-PT (the trained folds under `--root`, not retrained),
FedGuard-PT without CSEC (same TCNs, κ = η = 0 refit), raw-signal variant (TCN input = v_z, p_z, q_z only),
FedGuard-DC (notebook port, `fgpt/fgdc.py`, adaptations listed in its header), check-only rule (max of the five
standardized checks, threshold tuned on validation). FPR = no-attack streams at all six sites in S02, S03, S05, S06
(24 streams, warm-up excluded); the CSV also gives all five folds, 10-s post-event false alarms, and DC1 detection.
Outputs: `results_cmp/Table_offline_comparison.{tex,csv}`, `comparison_summary.json`, `comparison_streams.csv`,
`cross_check_pt.json` (FedGuard-PT batch scores vs the streamed scores). Time on a 2-core cloud machine: see VERIFICATION.md.

## 6. Sensitivity cases (assumed values)

The perturbation levels are assumptions chosen for a sensitivity study, not measured meter specifications.
State them as such in the paper.

| case | perturbation |
|---|---|
| `base` | none (the main S02 fold) |
| `noise1` | voltage noise 0.001 pu std; P/Q white noise 0.2 % of reading; fixed gain error per site and channel, uniform ±0.2 % |
| `noise3` | three times `noise1` |
| `kv04`, `kv06` | detector uses k_v = 0.4 or 0.6 (data generated with 0.5) |
| `fvfd09` | detector uses 0.9 × F_vfd |
| `tau12` | detector uses 1.2 × T_cool |

Noise is applied to every loaded record (calibration, training, validation, test, A-type files), seeded per file,
so runs are reproducible. Mismatch cases change only the detector's parameters.

## 7. Optional ablation (cloud or board)

`train_edge.py` flags restore each original behaviour: `--legacy-validation`, `--legacy-aux`, `--legacy-pooled-ms`,
`--legacy-grid`, `--legacy-ups`, `--legacy-fusion`; `FGPT_LEGACY_T3=1` restores the original T3 generator.
With all of them, the S02 fold reproduces the previous board run (λc = 8, λu = 2, τ* = −0.305, validation F1 0.7126).
The UPS mode is stored in `fusion.json`, and `rt_monitor.py` uses the mode the bundle was trained with.

```bash
FGPT_LEGACY_T3=1 python3 train_edge.py --dataset data/dataset --results results_abl/legacy/folds/S02_fault_100ms \
   --test-scenario S02_fault_100ms --legacy-validation --legacy-aux --legacy-pooled-ms --legacy-grid \
   --legacy-ups --legacy-fusion
```

## 8. Checks to do before using the numbers

1. `tests/check_generators.py` ends with `FAILURES: 0`.
2. Every `metrics.json` has `streaming_vs_batch_max_abs_diff` below 1e-5.
3. Real-time streams: `"realtime": true`, and `deadline_misses` is reported.
4. `Table_rev_gains` / the `[fusion]` lines: if a gain sits at the grid edge for any fold or sensitivity case, run
   `./probe_grid_edge.sh <name> <fold-dir> <scenario> '<perturb json>'`. It refits only the fusion stage into
   `probe/<name>/` with λc up to 512 (results are not touched) and prints F1 along λc. "unchanged" means the edge
   value is also the optimum of the wider grid (a flat top); otherwise report the difference or refit that case.
   Every `fusion.json` now stores the whole grid as `grid_f1`.

## 9. UPS check (method change)

**Problem found in the first cloud check.** `r_ups = 1{m != 0} P_it` was standardised on S01, which has no UPS
transfer, so its scale fell to 1e-9. A genuine transfer leaves P_it non-zero for a short transient (up to 17 MW at
DC3), so genuine transfers and U1 spoofs both clipped at z = 50, and the fusion learned a weight ≤ 0 for the check.
Calibrating on a scenario with transfers does not help: genuine transients are 10–54 of 29 801 samples, so the
median, MAD and 99th percentile are still exactly 0.

**New check** (`nb.ups_check`, streamed identically in `stream.py`), in pu of F_it·P_dc:

| part | rule | design value |
|---|---|---|
| power `a` | `1{m != 0} · P_it / (F_it P_dc)`, ignored for `G` after any reported change of m | `UPS_G = 2 s` |
| transfer `x` | a reported transfer (m: 0 → non-zero) whose bus voltage stayed above `Vtrip_k + δ` over the last 0.2 s; held for the episode | `UPS_DV = 0.02 pu`, `UPS_VWIN = 0.2 s` |
| check | `r_ups = max(a, x)`, fixed scale `UPS_SCALE = 0.05 pu` (no statistics) | |

Site values from `dc_params.m`: `Vtrip_v = [0.90 0.88 0.92 0.90 0.87 0.91]`, `Vret_v`, efficiencies 0.985/0.975.

Evidence (all reproduced by `tests/check_generators.py`):

- the check is exactly 0 on every record without a UPS-state attack (S01–S06, S08, S10, S11; all six sites);
- genuine transfers in the training data: longest grid-power transient 0.27 s (G = 2 s); highest onset voltage
  above Vtrip +0.0035 pu (δ = 0.02 pu). `train_edge.py` recomputes both per fold and stops if G or δ do not cover them;
- attack transfers start 0.040 (S07 DC1), 0.072 (S07 DC4) and 0.069 pu (S09 DC3) above Vtrip, so `x` flags the
  whole S07/S09 interval;
- U1: non-zero on every attacked sample at all six sites, 0 elsewhere.

**Limits to state in the paper.** A spoof whose onset coincides with a genuine low-voltage event is not flagged by `x`
and is seen by `a` only after G. U2 (transfer hidden by reporting m = 0) is not covered by this check: P_it is 0 and no
transition is reported. The thresholds G, δ and the scale are design values set from the simulated transfers, not
from field data.

**Fusion.** The logistic weights of the five check terms (`|r_bal| |r_aux| |r_cool| |r_ups| |r_qp|`) are bounded at
≥ 0 (L-BFGS-B, same penalised objective as scikit-learn; equal to scikit-learn when no bound is active,
`tests/check_fusion.py`). U1/U2 remain outside the validation mixture, so they still test generalisation.
`FGPT_BOUND_TERMS="|r_ups|"` bounds only the UPS check; `--legacy-fusion` removes all bounds. The trade-off between
these variants is measured in `VERIFICATION.md` (finding 3): the bound lowers T3/T3a ROC-AUC and keeps U1 detection
robust under model mismatch.

**Physics layer (fixed rule).** The fusion has no validation evidence to weight the UPS check (validation attacks never
change m, and the check is 0 on clean data), so the check also acts outside the learned fusion:

    score = fused score + RULE_BOOST · 1{UPS-check window term > RULE_Z},   RULE_Z = 1, RULE_BOOST = 1000

A term above 1 means the mean violation over the window exceeds UPS_SCALE = 0.05 pu of F_it·P_dc. The boost puts the
window above any threshold and at the top of the ranking, so ROC-AUC, F1 and the alarm all include the rule. It has no
learned parameter, is applied in threshold/gain tuning, streaming, batch parity and the live display, and is stored in
`fusion.json` (`physics_rule`); bundles without that key run without it. The explanation of a rule window is
`UPS rule`. `--no-ups-rule` trains a bundle without it (ablation). `ups_rule_analysis.py` separates the learned and
rule parts per stream, and every `metrics.json` reports how many windows the rule fired on, split by label.
