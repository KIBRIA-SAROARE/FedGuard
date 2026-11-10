#!/usr/bin/env bash
# Link or download the dataset used by the experiment, and make sure every site's attacked files are present.
#  - inside the repository layout (FedGuard-PT/<this folder> next to FedGuard-PT/dataset): links ../dataset
#  - otherwise: sparse clone of FedGuard-PT/dataset (66 clean CSVs, attacked CSVs, attack_log.csv)
#  - if the linked dataset lacks attacked files for some site (e.g. it was fetched for DC1 only), the missing
#    files are downloaded and ADDED to it (existing files are never overwritten; files present in both copies
#    must be byte-identical, otherwise the script stops).
# Usage: ./get_data.sh
set -euo pipefail
cd "$(dirname "$0")"; mkdir -p data
REPO_URL="${REPO_URL:-https://github.com/KIBRIA-SAROARE/FedGuard-DC.git}"
D=/FedGuard-PT/dataset

sparse_clone() {   # $1 = target dir, rest = sparse patterns
  local dst=$1; shift
  [ -d "$dst/.git" ] || git clone --filter=blob:none --no-checkout --depth 1 "$REPO_URL" "$dst"
  git -C "$dst" sparse-checkout init --no-cone
  git -C "$dst" sparse-checkout set "$@"
  git -C "$dst" checkout
}

if [ -d ../dataset/clean ]; then
  ln -sfn "$(cd ../dataset && pwd)" data/dataset
elif [ ! -e data/dataset ]; then
  sparse_clone data/repo "$D/attack_log.csv" "$D/manifest.csv" "$D/clean/*.csv" "$D/attacked/*.csv"
  ln -sfn repo/FedGuard-PT/dataset data/dataset
fi

count() { ls data/dataset/attacked 2>/dev/null | grep -c "_DC${1}_A" || true; }
missing=0
for k in 1 2 3 4 5 6; do [ "$(count $k)" -ge 66 ] || missing=1; done

if [ "$missing" = 1 ]; then
  echo "attacked files incomplete; downloading the missing ones ..."
  sparse_clone data/repo_fill "$D/attacked/*.csv"
  SRC=data/repo_fill$D/attacked
  DST=$(cd data/dataset/attacked 2>/dev/null && pwd || (mkdir -p data/dataset/attacked && cd data/dataset/attacked && pwd))
  added=0; checked=0
  for f in "$SRC"/*.csv; do
    b=$(basename "$f")
    if [ -e "$DST/$b" ]; then
      cmp -s "$f" "$DST/$b" || { echo "ERROR: $b differs between your dataset and the repository; not mixing them"; exit 1; }
      checked=$((checked + 1))
    else
      cp "$f" "$DST/$b"; added=$((added + 1))
    fi
  done
  echo "added $added files; $checked existing files verified identical to the repository"
fi

NC=$(ls data/dataset/clean | wc -l)
echo "clean: $NC (need 66)"
ok=1
for k in 1 2 3 4 5 6; do
  c=$(count $k); echo "attacked for DC$k: $c (need 66: 11 scenarios x A0-A5)"; [ "$c" -ge 66 ] || ok=0
done
[ "$NC" -eq 66 ] && [ "$ok" = 1 ] || { echo "ERROR: dataset incomplete"; exit 1; }
echo "dataset complete"
