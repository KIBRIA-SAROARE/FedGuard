# FedGuard-DC

Federated load forecasting and cyber-attack detection for AI data-center loads.

This repository contains the FedGuard-DC study materials and the FedGuard-PT dataset and Jetson replay experiment. Use `FedGuard-DC/` for the FedGuard-DC notebook, data and model materials. Use `FedGuard-PT/dataset/` with `FedGuard-PT/fgpt_jetson/` to reproduce the transmission-system replay experiment. Select the dataset required by the corresponding code; these directories should not be treated as interchangeable.

```text
FedGuard-DC/
  Copy_of_FedGuard_DC_NAPS2026.ipynb
  copy_of_fedguard_dc_naps2026.py
  Data/
  Figures/
  Modified IEEE 39 Bus with Data Center Simulink/
FedGuard-PT/
  dataset/
  fgpt_jetson/
```

---

## Transmission-system dataset (v2)

Dataset path: `FedGuard-PT/dataset/`.

Six heterogeneous AI data centers embedded in the IEEE 39-bus New England system, simulated in MATLAB/Simulink R2023b (Simscape Electrical Specialized Power Systems, **phasor** mode, 60 Hz).

| DC | Bus | Rated (MW) | Workload |
|---|---|---|---|
| DC1 | 4 | 350 | training |
| DC2 | 23 | 200 | inference |
| DC3 | 18 | 150 | mixed |
| DC4 | 16 | 300 | training |
| DC5 | 20 | 450 | mixed |
| DC6 | 8 | 250 | inference |

Each site uses its own UPS thresholds, battery size, dc-link parameters, cooling time constants and reactive-power mode, so the six clients are non-IID by construction.

### How it was generated

Each data center is a WECC-style composite load model extended with IT, cooling and auxiliary branches, a synthetic AI workload generator, and an averaged double-conversion UPS (rectifier under dc-link voltage control with current limit, battery with SoC, diesel start, cooling motor stall and restart). The workload trace is synthetic — it reproduces the structure of AI facility power (training iteration cycles, checkpoint dips, inference burstiness, diurnal envelope) rather than replaying a measured trace.

Eleven 300 s scenarios were simulated: normal operation, two three-phase faults, a line trip, a load step, a combined fault + line trip, four control-path attacks, and one case with a genuine fault and an attack in the same run. Signals are logged at 1 kHz and exported at 100 Hz (30 001 samples per file).

Two attack classes are kept separate on purpose:

- **Control-path attacks** are injected inside the model (false UPS transfer, load-altering, cooling setpoint manipulation, voltage-sensor spoofing), so the grid actually responds to them. These are scenarios S07–S11.
- **Measurement-path attacks** (scaling, ramp/drift, replay, spike, stealthy bias) are applied afterwards to the exported telemetry only, since they corrupt what a site reports without changing the physics. Files in `attacked/` carry per-sample `label` and `attack_type` columns.

### Scenarios

| Name | Event |
|---|---|
| `S01_normal` | no disturbance |
| `S02_fault_100ms` | 3φ-G fault, bus 16, 60.00–60.10 s |
| `S03_fault_250ms` | 3φ-G fault, bus 4, 60.00–60.25 s |
| `S04_line_trip` | line trip 90–95 s |
| `S05_load_step` | +200 MW at bus 20, t = 120 s |
| `S06_multi_event` | fault at bus 16 + line trip 150–300 s |
| `S07_atk_false_ups` | false UPS transfer, DC1 and DC4, 140–170 s |
| `S08_atk_load_alter` | load-altering, DC2 and DC5, 100–160 s |
| `S09_atk_sensor_spoof` | voltage-sensor spoofing, DC3, 180–220 s |
| `S10_atk_cooling` | cooling setpoint manipulation, DC6, 200–250 s |
| `S11_fault_plus_atk` | fault at bus 16 + load-altering on DC6, 62–92 s |

`S11` tests whether a detector can distinguish the load-altering attack on DC6 from the genuine grid fault. Evaluate attack detection at DC6 and false alarms at untargeted sites separately. The current Jetson experiment monitors DC1, so its S11 result measures untargeted-site false alarms rather than detection of the DC6 attack.

`clean/` contains exports before additional measurement-path FDIA is applied. S01–S06 are attack-free simulation scenarios. S07–S11 contain control-path attacks and must not be treated as benign records merely because they are stored in `clean/`. Files in `attacked/` add measurement-path corruption to the underlying scenario; distinguish the underlying control attack from the added measurement attack.

### File naming

```
clean/S<NN>_<event>_DC<k>.csv               k = 1..6
attacked/S<NN>_<event>_DC<k>_A<type>.csv    type = 1..5
raw/S<NN>_<event>.mat                       1 kHz, fields DC1..DC6
```

### Columns

| Column | Unit | Column | Unit |
|---|---|---|---|
| `t` | s | `ups_state` | 0 grid, 1 battery, 2 diesel, 3 IT tripped |
| `Va_pu` `Vb_pu` `Vc_pu` | pu | `soc` | 0–1 |
| `P_meas_MW` | MW | `vdc_pu` | pu |
| `P_total_MW` | MW | `P_batt_MW` | MW |
| `Q_total_Mvar` | Mvar | `cool_stall` | 0/1 |
| `P_it_MW` `P_cool_MW` `P_aux_MW` | MW | `P_ai_pu` | pu |
| | | `P_it_served_MW` | MW |

Attacked files add `label` (0/1) and `attack_type` (0–5).

### Preprocessing, trusted inputs and leakage

- Exclude samples with `t < 2 s` before constructing windows. This interval contains model soft-start transients. The Jetson loader retains samples from 2 s onward, so the processed record spans approximately 298 s.
- Measurement-path attacked files leave `ups_state`, `soc`, `vdc_pu`, `P_batt_MW` and `P_it_served_MW` uncorrupted as reference channels. Exclude these channels, and features derived from them, when evaluating a detector whose threat model provides no independently trusted internal telemetry.
- Never use `label`, `attack_type` or attack-window metadata as detector inputs. If trusted reference channels are used, identify them explicitly and report that experiment separately from a telemetry-only evaluation. The current Jetson feature pipeline uses `ups_state`; its results therefore require a stated assumption that this channel remains trustworthy.
- Keep each original scenario and its attacked derivatives in the same evaluation partition. Split records before constructing overlapping windows. Fit calibration, normalization, fusion coefficients and thresholds using the designated training or validation records only.

### Sampling and model limitations

The 100 Hz CSV exports retain every tenth sample of the 1 kHz records without anti-alias filtering. Frequencies above 50 Hz can therefore alias into the exported band. Use the 1 kHz `raw/*.mat` records for higher-frequency analysis or apply a documented low-pass filter before generating a new 100 Hz export. Filtering an existing 100 Hz CSV cannot undo aliasing. The 1 kHz logging rate does not make this phasor-mode simulation an EMT or switching-waveform dataset.

The AI workload is synthetic, not a measured facility trace.

The v2 records for DC1 and DC6 in S02, S06 and S11 contain dc-link overvoltage excursions. The model diagnosis attributes these excursions to rectifier current-limit release during cooling-motor restart; this explanation should be supported by a documented validation before being presented as a validated cause. Exclude these six scenario/site records and their corresponding attacked derivatives from analyses using `vdc_pu` or features derived from it. For analyses using other channels, document whether the records are retained and assess whether the excursions affect the reported results.

---

## Jetson replay experiment

See [the Jetson README](FedGuard-PT/fgpt_jetson/README.md) for setup, outputs and experiment stages.

The Jetson Orin Nano experiment replays simulated IEEE 39-bus telemetry with synthetic AI workloads at a scheduled 100 Hz wall-clock rate. It does not acquire live field measurements. DC1 is monitored; the other five sites provide precomputed peer-event information. Six federated clients are emulated in one process, and no physical network link is exercised. Communication figures describe update payload sizes rather than measured network traffic or transfer latency.

CSEC values are computed from complete records before replay and withheld until the implemented availability conditions are met. This tests delayed release of precomputed values, rather than online peer-event extraction and descriptor transport. Descriptor transport delay defaults to 0 ms. Report provisional and released decisions separately, including CSEC waiting time.

### Reproduction

From a checkout of this repository:

```bash
cd FedGuard-PT/fgpt_jetson
chmod +x *.sh
./setup_jetson.sh
./get_data.sh 1
./run_experiment.sh full
```

Record the exact repository commit, dataset hashes, Python and package versions, Jetson software version, power mode, clock settings, fan settings and command-line arguments. Use a fresh output directory when code, data or configuration changes, because completed stages are skipped on restart.

The experiment uses five held-out disturbance scenarios, S02–S06, plus a FULL fold for control-path tests. Calibration and fusion validation use S01. Windows contain 128 samples with a 40-sample hop at 100 Hz. Record the configuration and separate validation and test attack seeds with each result.

The Jetson configuration uses clipping with zero differential-privacy noise (`DP_SIGMA = 0.0`). These runs do not establish a differential-privacy guarantee. Federated processing avoids centralizing raw telemetry in the intended architecture, but model updates and event descriptors still require an explicit privacy threat model.

### Interpreting results

- T1, T2, T3a and T3 are summarized over five held-out-scenario streams each. A1–A5 are summarized over 25 streams: five attack types across five held-out scenarios. Error bars show population standard deviation across the included stream-level metrics, not confidence intervals or repeated-seed uncertainty.
- Processing latency, start lateness, deadline misses, decision cadence and CSEC release wait are distinct quantities. Report them separately.
- Detection delays in `attack_events` use simulation release times; they are not measured end-to-end wall-clock detection delays. Delay summaries include detected events only and must be accompanied by detected and total event counts.
- By default, `paper_figures.py` excludes streams whose `metrics.json` does not mark them as real-time. This flag identifies the pacing mode; scheduling performance must be assessed from the recorded timing and deadline-miss results.
- Use `results_rt/` for paced replay results. The earlier `results/` run measured throughput without wall-clock pacing; do not combine it with real-time results.
- The September 18 commit `5c96a0098295f347415ef09b2f5296135c196e8d` introduced a corrupted A1–A5 label in `Table_rt_detection.tex`. Use `A1--A5` in LaTeX tables. Its small floating-point changes in `numbers.json` should not be interpreted as performance improvements.

---

## Citation

Cite the publication corresponding to the experiment used, and record the repository commit and dataset version. Verify the publication title, author list and identifier against the publication record before copying its BibTeX entry. The previously supplied BibTeX entry had an empty author field and has been removed pending metadata verification.
