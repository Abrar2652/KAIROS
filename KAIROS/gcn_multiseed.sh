#!/usr/bin/env bash
# Multi-seed GCN backbone runs with fixed_tau=0.5 for Bitcoin-family.
# Usage: GPU=X DS=<bitcoinotc|bitotc|bitalpha> bash gcn_multiseed.sh
set -uo pipefail
cd "$(dirname "$0")"

GPU="${GPU:-2}"
DS="${DS:-bitcoinotc}"

case "$DS" in
  bitcoinotc) SNAPS=4; VIEWS=3; STRAT=sequential; DL=64;   EP=25  ;;
  bitotc)     SNAPS=4; VIEWS=4; STRAT=random;     DL=4096; EP=50  ;;
  bitalpha)   SNAPS=6; VIEWS=4; STRAT=sequential; DL=4096; EP=200 ;;
  *) echo "unknown $DS"; exit 1 ;;
esac

runs="/nas/home/jahin/KAIROS/runs"
for s in 24 42 7 13 99; do
  tag="${DS}_gcn_fixedtau05_s${s}"
  log="${runs}/${tag}.log"
  if grep -q "ablate-result" "$log" 2>/dev/null; then
    echo "[skip] $tag"; continue
  fi
  echo "[run ] GPU=$GPU $tag"
  CUDA_VISIBLE_DEVICES=$GPU python3 -u /nas/home/jahin/KAIROS/KAIROS/ablate.py \
    --dataset "$DS" --task classification \
    --snapshots "$SNAPS" --views "$VIEWS" --strategy "$STRAT" \
    --dataloader_size "$DL" --GPU 0 --epochs "$EP" --seed "$s" \
    --ablation fixed_tau --tau_val 0.5 \
    --backbone gcn > "$log" 2>&1
  echo "[done] $tag"
done
echo "GCN multi-seed for $DS on GPU $GPU complete."
