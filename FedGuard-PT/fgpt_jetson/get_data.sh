#!/usr/bin/env bash
# Link or download the dataset used by the experiment.
#  - inside the FedGuard-DC repo (FedGuard-PT/fgpt_jetson): links ../dataset
#  - elsewhere: sparse clone of FedGuard-PT/dataset (66 clean CSVs, attacked CSVs for one DC, attack_log.csv)
# Usage: ./get_data.sh [DC=1]
set -euo pipefail
DC="${1:-1}"
cd "$(dirname "$0")"; mkdir -p data
if [ -d ../dataset/clean ]; then
  ln -sfn "$(cd ../dataset && pwd)" data/dataset
else
  REPO_URL="${REPO_URL:-https://github.com/KIBRIA-SAROARE/FedGuard.git}"
  [ -d data/repo/.git ] || git clone --filter=blob:none --no-checkout --depth 1 "$REPO_URL" data/repo
  cd data/repo
  git sparse-checkout init --no-cone
  D=/FedGuard-PT/dataset
  git sparse-checkout set "$D/attack_log.csv" "$D/manifest.csv" "$D/clean/*.csv" "$D/attacked/*_DC${DC}_A*.csv"
  git checkout
  cd ../..
  ln -sfn repo/FedGuard-PT/dataset data/dataset
fi
echo "clean: $(ls data/dataset/clean | wc -l) (need 66)   attacked for DC${DC}: $(ls data/dataset/attacked | grep -c "_DC${DC}_A") (need 30 for S02..S06)"
