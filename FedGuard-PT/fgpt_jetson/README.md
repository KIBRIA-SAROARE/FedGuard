# FedGuard-PT real-time experiment on Jetson Orin Nano (one site, DC1)

The model, features, attacks, CSEC and fusion are ported from `FedGuard_PT_Phase2_Final.ipynb` (`fgpt/nb.py`, `fgpt/tcn_np.py`).
Every monitor stream replays one 300 s record at **100 Hz wall clock** on the board.
This is a hardware-in-the-loop replay of the simulated IEEE 39-bus telemetry, not live field measurements. State it that way in the paper.

## 1. Setup (once)

```bash
cd FedGuard-DC/FedGuard-PT/fgpt_jetson      # or wherever this folder is
chmod +x *.sh
./setup_jetson.sh     # apt: numpy pandas scipy sklearn matplotlib psutil tk tmux fonts
./get_data.sh 1       # links ../dataset if present, otherwise sparse-clones FedGuard-PT/dataset
```

Before starting, fix the board state and do not change it during the run:

- Keep the same power mode as your earlier run (`sudo nvpmodel -q` showed 25W).
- Close the browser and other apps.
- Keep the fan setting unchanged.

The run records `nvpmodel -q` and `jetson_clocks --show` in `results_rt/experiment_plan.json`.

## 2. Run

```bash
tmux new -s fgpt
./run_experiment.sh full        # Ctrl-b d to detach; tmux attach -t fgpt to return
```

- `./run_experiment.sh quick` is a short check: S02 fold with no_attack, T1, T3, A1, plus S07 control.
- If the run stops, run the same command again. Finished trainings and streams are skipped.

| stage | script | content |
|---|---|---|
| idle | `measure_idle.py` | 60 s at the start and 60 s at the end. Board power, CPU, RAM, temperature |
| training | `train_edge.py` | 5 leave-one-scenario-out folds (S02–S06 held out) + FULL fold. Six sites emulated in one process, 25 rounds, int8 uplink, trimmed mean |
| streams | `rt_monitor.py` | per LOSO fold: `no_attack`, `T1`, `T2`, `T3a`, `T3`, `A1`–`A5` on the held-out scenario (50 streams). FULL fold: control-path S07–S11 (5 streams) |
| latency sweep | `bench_latency.py` | TCN forward, batch 1, 1/2/4/6 threads |
| live | `live_capture.py` | real-time dashboard for S02-A1, S02-no_attack, S02-T3, S07-control. Snapshots at the grid event, attack onsets and the end |
| figures | `paper_figures.py` | figures, LaTeX tables, `numbers.json` |

**Duration estimate:**

- Training: the earlier board run measured 428.7 s plus 75.8 s of fusion fitting for one fold. Six folds are about 51 min.
- Streams: 55 × (298 s + 20 s cooldown) ≈ 4.9 h, plus per-stream preparation.
- Live captures: 4 × about 5 min.
- Total: roughly 6–7 h.

## 3. What each stream measures

| quantity | definition |
|---|---|
| feature update per sample | wall time to update the 10 channels for one 10 ms sample |
| sample with TCN + fusion | wall time of the samples (every 40th) that also run the TCN and fusion |
| start lateness | processing start minus the scheduled 10 ms arrival (Python sleep jitter plus any backlog) |
| deadline misses | samples whose processing ended after their 10 ms slot |
| CSEC release wait | time a window's score waits until its CSEC inputs are causally known (see below) |
| detection | notebook window metrics (ROC-AUC, F1, precision, recall, FPR), per-attack detection delay |
| resources | tegrastats every 0.5 s (VDD_IN, CPU/GPU/SOC rails, GR3D load, temperatures) and psutil process CPU/RSS |

**CSEC in real time.** The notebook computes CSEC over the whole record. `fgpt/csec_rt.py` keeps the same values but releases them only when a live site could know them:

- the site's own voltage event has ended;
- 2 s have passed since the event onset;
- every matched peer's event has ended, and its 15-byte descriptor has arrived.

Descriptor transport delay is assumed to be 0 ms (`--net-delay-ms` changes it). Each window therefore gets two decisions:

- a provisional score at window end, without CSEC;
- the released score, which equals the notebook score.

Both detection delays are reported.

## 4. Outputs

```
results_rt/experiment_plan.json  experiment_log.txt  idle_start/  idle_end/
results_rt/folds/<fold>/bundle/  train_log.json  resources_train.csv  tegrastats_train.log  (bench_latency.csv in the S02 fold)
results_rt/streams/<fold>/<test>_<scenario>/metrics.json  windows.csv  sample_timing.npz  signals.npz  resources.csv  tegrastats.log
results_rt/live/*.png|pdf
results_rt/paper/Fig_rt_*.pdf|png  Table_rt_*.tex|csv  numbers.json
```

## 5. Paper figures (`results_rt/paper/`)

| file | use |
|---|---|
| `Fig_rt_timeline_S02_fault_100ms_A1_S02_fault_100ms_zoom_40_120s` | fault at 60 s overlapping repository FDIA A1: voltage, reported vs facility power, provisional/released score, alarms |
| `Fig_rt_timeline_S02_fault_100ms_no_attack_*` | the same fault with no attack (false-alarm behaviour) |
| `Fig_rt_timeline_S02_fault_100ms_T3_*` | cross-domain T3 attack |
| `Fig_rt_timeline_FULL_ctrl_S07_atk_false_ups_zoom_120_210s` | control-path false UPS transfer at DC1 |
| `Fig_rt_detection_by_regime` | ROC-AUC, F1, FPR (mean ± std over the 5 LOSO folds) for T1, T2, T3a, T3, A1–A5, no attack |
| `Fig_rt_disturbance_false_alarms` | FPR per held-out disturbance scenario; label = false alarms / negative windows within 10 s of the event |
| `Fig_rt_detection_delay` | per-attack detection delay, provisional vs released |
| `Fig_rt_control_path` | S07–S11 at DC1 (S07 targets DC1; S08–S11 show DC1 false alarms) |
| `Fig_rt_latency` | CDF of per-sample processing and start lateness vs the 10 ms period; CDF of CSEC release wait |
| `Fig_rt_resources`, `Fig_rt_resources_timeseries_*` | idle vs training vs real-time monitoring: board power, process CPU, RSS, T_j |
| `Fig_rt_training_communication` | DC1 local training time per round (all folds), convergence, uplink/downlink bytes |
| `results_rt/live/*` | dashboard snapshots taken during the live runs |

Every value drawn is also written to `numbers.json`, `Table_rt_detection.tex`, `Table_rt_edge_resources.tex` and `Table_rt_all_streams.csv`. Copy numbers into the paper from those files only.
`paper_figures.py` refuses streams that were not run at 100 Hz wall clock.

