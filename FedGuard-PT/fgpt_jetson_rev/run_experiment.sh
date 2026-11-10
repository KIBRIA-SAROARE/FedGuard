#!/usr/bin/env bash
# Revision experiment on the Jetson -> tables and figures.   Usage (inside tmux):
#   ./run_experiment.sh            all plans: main, attacked, sites, sens (resumable)
#   ./run_experiment.sh quick      short pipeline check (about 1 h)
#   ./run_experiment.sh main       one plan only (main | attacked | sites | sens, or a comma list)
#   ./run_experiment.sh cmp        offline comparison table only (needs the finished main + sites plans)
set -euo pipefail
cd "$(dirname "$0")"
PLAN="${1:-all}"
OUT="${OUT:-results_rev}"
mkdir -p "$OUT"
{
echo "=== $(date) plan=$PLAN out=$OUT ==="
./get_data.sh            # stops here if the dataset is incomplete (fills missing attacked files)
python3 tests/check_generators.py data/dataset
python3 tests/check_fusion.py
if [ "$PLAN" = "cmp" ]; then
  python3 offline_comparison.py --root "$OUT" --out "${CMP_OUT:-results_cmp}" --dataset data/dataset
  echo "=== done $(date) ==="; exit 0
fi
python3 run_full_experiment.py --plan "$PLAN" --out "$OUT"

if [ "$PLAN" = "all" ] || [[ "$PLAN" == *main* ]]; then
  echo "=== live displays (real time, snapshots saved to $OUT/live) ==="
  [ -n "${DISPLAY:-}" ] || export MPLBACKEND=Agg
  python3 live_capture.py --no-hold --out "$OUT" --fold S02_fault_100ms --test A1
  python3 live_capture.py --no-hold --out "$OUT" --fold S02_fault_100ms --test no_attack
  python3 live_capture.py --no-hold --out "$OUT" --fold S02_fault_100ms --test T3
  python3 live_capture.py --no-hold --out "$OUT" --fold FULL --test ctrl --scenario S07_atk_false_ups
fi

echo "=== figures (DC1 real-time streams) and revision tables ==="
if [ "$PLAN" = "quick" ]; then
  python3 paper_figures.py --root "$OUT" || echo "paper_figures: quick plan has too few streams for some figures"
else
  python3 paper_figures.py --root "$OUT"
fi
python3 revision_summary.py --root "$OUT" --dataset data/dataset
python3 ups_rule_analysis.py --root "$OUT"
echo "=== done $(date) ==="
} 2>&1 | tee -a "$OUT/run_experiment_console.log"
