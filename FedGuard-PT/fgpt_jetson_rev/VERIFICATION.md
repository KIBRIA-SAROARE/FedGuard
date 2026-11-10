# Verification run (cloud, before the board run)

Machine: 2-core x86 cloud container, Python 3, NumPy 2.5.3, pandas 3.0.5, scikit-learn 1.9.1.
Command: `python3 run_full_experiment.py --plan all --no-realtime --cooldown-s 0 --idle-s 5`.
All streams are **fast replay**, so the timing numbers in `cloud_check/` are not board measurements.
Detection numbers are what the same code computes; the board run may differ slightly
(different BLAS/NumPy floating point; see check 2). Configuration: the package defaults, i.e. the
revised UPS check, fusion weights of all five checks bounded at ≥ 0, and the physics rule on the UPS check.

## Checks

| # | Check | Result |
|---|---|---|
| 1 | `tests/check_generators.py`: T3/T3a clean before the first attack (6 sites × 5 scenarios); invariants kept inside attacks; U1 changes only m; U2 hides the S07/S09 transfer; UPS check exactly 0 on every record without a UPS-state attack, flags all of S07 (DC1, DC4) and S09 (DC3), non-zero on every U1 sample at all six sites; auxiliary class near thirds | 44/44 PASS |
| 2 | `tests/check_fusion.py`: bounded fit equals scikit-learn (run to tol 1e-12) when no bound is active; bound respected when active; physics rule fires only above its level and adds exactly the boost; off for bundles without it | max weight diff 2.7e-8; PASS |
| 3 | Legacy regression: all legacy switches on (incl. `--legacy-ups --legacy-fusion`), S02 fold, vs the previous board run | λc 8 / λu 2 / val F1 0.7126 / 16 018 windows (identical); τ* −0.3056 vs −0.3049; A1 ROC-AUC 0.9979 (identical), no-attack FPR 1.08 % vs 1.07 %, T3 FPR 10.47 % (identical) |
| 4 | Every plan runs: 12 trainings, 421 streams (61 main + 8 attacked-site + 275 site + 77 sensitivity) | all exit 0 |
| 5 | Resume after an interruption (container restart mid-run, first revision) | finished stages skipped, run completed |
| 6 | Streamed vs batch released scores, all 421 streams; streamed vs batch UPS-check features | max 3.7e-6; features 6e-8 |
| 7 | Real-time mode (100 Hz pacing): S02 U1 at DC1 with the final configuration; S07 ctrl at DC1 before the rule was added | 298.0 s wall each; metrics identical to the fast replay of the same configuration (`cloud_check/realtime_*.txt`) |
| 8 | `--refit-fusion` reproduces a full training's fusion (S02) | identical τ*, F1, gains |
| 9 | CSEC gains inside the grid for all 6 folds and 6 sensitivity cases (λu grid to 1024; S05 fold plateaus from 512) | no gain at a grid edge |
| 10 | UPS design values cover the genuine transfers in every fold's training + validation scenarios | longest transient 0.27 s (G = 2 s); highest onset voltage +0.0035 pu above Vtrip (δ = 0.02 pu) |
| 11 | Clipped CSEC releases | 0 windows in every stream group |
| 12 | `live_capture.py` headless, `paper_figures.py`, `revision_summary.py`, `ups_rule_analysis.py` | figures and tables written; live banner names "UPS rule" on rule windows |
| 13 | Physics rule over all 421 streams | fired on 6371 attack windows and 0 clean windows; changes no metric except U1 (and A1, where the voltage scaling meets a genuine transfer onset) |

## Findings to know before the board run

1. **UPS check and physics rule (README §9).** DC1, 5 folds: U1 ROC-AUC 1.000 ± 0.000 (0.990 before the UPS change);
   U1 ROC-AUC 1.000 at all six sites and in every sensitivity case (it was 0.011–0.028 under noise3, kv04, kv06, fvfd09
   before the change). The rule raises U1 recall from 0.984 (learned fusion alone) to 1.000 and fires on no clean
   window. S07 and S09 are detected 0.08 s after onset (was 1.68–2.35 s). No-attack FPR at DC1 1.11 % (was 1.56 %).
2. **The learned fusion gives the UPS-check term weight 0 in 4 of 6 folds**, because the validation attacks never
   change m. This is why the rule exists; state it in the paper.
3. **Cost of bounding the fusion weights** (measured, DC1, mean over the 5 LOSO folds; ROC-AUC, FPR in brackets):

   | test | before UPS change | UPS change, unbounded fusion | UPS change, all 5 checks bounded (default) |
   |---|---|---|---|
   | no attack | (1.56 %) | (1.27 %) | (1.11 %) |
   | T3a | 0.754 (1.67 %) | 0.754 (1.71 %) | 0.675 (1.04 %) |
   | T3 | 0.822 (1.71 %) | 0.823 (1.71 %) | 0.736 (1.04 %) |
   | U1 | 0.990 (1.35 %) | 1.000 (1.35 %) | 0.996 (1.34 %), 1.000 with the rule |
   | A1 | 0.979 (2.06 %) | 0.981 (1.89 %) | 0.982 (1.96 %), 0.992 with the rule |

   Under model mismatch (S02 fold) the unbounded fusion learns negative check weights (|r_qp| −0.20 for kv04) and U1
   collapses to ROC-AUC 0.020–0.029 (noise3, kv04, kv06); bounded it stays 0.959–0.984 before the rule and 1.000
   with it. The bound costs T3/T3a accuracy (0.823 → 0.736). `FGPT_BOUND_TERMS` and `--legacy-fusion` select the
   other variants; `--no-ups-rule` removes the rule.
4. **Means over the 5 LOSO folds, DC1, default configuration**: `cloud_check/Table_rev_dc1.csv`.
5. **Attacked-site control path** (`cloud_check/Table_rev_control.csv`): S08@DC2 recall 0.227 and S11@DC6
   recall 0.347 remain weak; S07@DC1 FPR is 12.2 %.

Use the board run for the paper; these numbers only show that the pipeline works and what to expect.

## Revision 3: warm-up exclusion and offline comparison (cloud, 2-core)

| # | Check | Result |
|---|---|---|
| 14 | `recompute_metrics.py --warmup-s 0` on a copy of all 421 cloud streams | reproduces every original metrics.json (0 mismatches in AUC, PR-AUC, F1, precision, recall, FPR, counts, delay, provisional, 10-s disturbance window, rule counts) |
| 15 | New `rt_monitor.py` run (S02 no-attack, DC1) vs `recompute_metrics.py` on the old stream | FPR 0.00414 both; stream-batch 5.2e-7 |
| 16 | `tests/check_generators.py`, `tests/check_fusion.py` after the changes | 44/44 PASS; PASS |
| 17 | `offline_comparison.py`: FedGuard-PT batch scores vs the streamed scores (80 streams: 5 folds × 6 sites no-attack + DC1 10 tests) | max 5.8e-6 (stream CSV has 5 decimals); FPR/F1 identical except 2 streams that differ by 1 window each, where the window score equals the threshold to within 1e-5 |
| 18 | Raw-signal parameters / traffic | 3485 parameters, 3264 shared, 96.38 KiB/round (C = 16 inputs × 7 fewer channels) |
| 19 | CSEC κ at the grid maximum (32) for central S02, S05 and raw S02 | refit with κ grid to 256: κ stays 32, same F1, so the grid does not limit these rows |

Cloud results of the comparison: `cloud_check/comparison/`. Time: TCN variants ≈ 4–5 min per fold each, FedGuard-DC ≈ 2.5 min
per fold (150 rounds), evaluation ≈ 15 min (all on 2 cores, run in parallel). Board time not measured.
