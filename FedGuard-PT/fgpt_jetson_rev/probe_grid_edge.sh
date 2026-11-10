#!/usr/bin/env bash
# Check whether a CSEC gain that sits at the grid edge is a real optimum or a plateau.
# Refits ONLY the fusion stage of a finished fold into probe/<name>/ (results_rev is not touched) with a wider grid,
# and prints validation F1 along lambda_c at the selected lambda_u.
# Usage: ./probe_grid_edge.sh <name> <fold-dir> <test-scenario> '<FGPT_PERTURB json>'
#   e.g. ./probe_grid_edge.sh noise3 results_rev/sens/noise3/folds/S02_fault_100ms S02_fault_100ms '{"noise": 3}'
set -euo pipefail
cd "$(dirname "$0")"
name=$1; src=$2; scen=$3; pert=${4:-"{}"}
dst=probe/$name
rm -rf "$dst"; mkdir -p probe; cp -r "$src" "$dst"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
FGPT_PERTURB="$pert" FGPT_KAPPA_GRID="0,1,2,4,8,16,32,64,128,256,512" \
  python3 train_edge.py --dataset data/dataset --results "$dst" --test-scenario "$scen" --refit-fusion | grep "\[fusion\]"
python3 - "$src/bundle/fusion.json" "$dst/bundle/fusion.json" <<'PY'
import json, sys
a, b = (json.load(open(p)) for p in sys.argv[1:3])
eta = b["eta"]
print(f"run:   lambda_c={a['kappa']} lambda_u={a['eta']} F1={a['val_f1']:.5f}")
print(f"probe: lambda_c={b['kappa']} lambda_u={b['eta']} F1={b['val_f1']:.5f}")
print(f"F1 along lambda_c at lambda_u={eta}:")
for k, e, f in b["grid_f1"]:
    if e == eta:
        print(f"   {k:6g}  {f:.5f}")
same = a["kappa"] == b["kappa"] and a["eta"] == b["eta"]
gain = b["val_f1"] - a["val_f1"]
print("VERDICT:", "unchanged: the run's gains are the optimum of the wider grid" if same else
      f"wider grid picks different gains (F1 +{gain:.5f}); report it, or refit this case with the wider grid")
PY
