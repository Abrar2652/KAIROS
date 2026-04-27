#!/usr/bin/env bash
# Multi-seed anomaly with τ=0.5 for the 5 small datasets
set -uo pipefail
cd "$(dirname "$0")"

GPU="${GPU:-6}"
DS="${DS:-bitcoinotc}"

case "$DS" in
  bitcoinotc) SNAPS=4; VIEWS=4; STRAT=sequential; DL=64;   EP=25  ;;
  bitotc)     SNAPS=4; VIEWS=4; STRAT=sequential; DL=4096; EP=50  ;;
  bitalpha)   SNAPS=5; VIEWS=5; STRAT=sequential; DL=4096; EP=100 ;;
  dblp)       SNAPS=4; VIEWS=4; STRAT=sequential; DL=4096; EP=200 ;;
  tax51)      SNAPS=4; VIEWS=4; STRAT=sequential; DL=4096; EP=200 ;;
  *) echo "unknown $DS"; exit 1 ;;
esac

runs="/nas/home/jahin/KAIROS/runs"
for s in 24 42 7 13 99; do
  tag="${DS}_ano_fixedtau05_s${s}"
  log="${runs}/${tag}.log"
  # Check if ablate-result with anomaly task and seed
  if grep -q "ablate-result" "$log" 2>/dev/null && grep -q "task=anomaly_detection" "$log" 2>/dev/null && grep -q "seed=$s " "$log" 2>/dev/null; then
    echo "[skip] $tag"; continue
  fi
  echo "[run ] GPU=$GPU $tag"
  CUDA_VISIBLE_DEVICES=$GPU python3 -u /nas/home/jahin/KAIROS/KAIROS/ablate.py \
    --dataset "$DS" --task anomaly_detection \
    --snapshots "$SNAPS" --views "$VIEWS" --strategy "$STRAT" \
    --dataloader_size "$DL" --GPU 0 --epochs "$EP" --seed "$s" \
    --ablation fixed_tau --tau_val 0.5 > "$log" 2>&1
  echo "[done] $tag"
done
echo "Anomaly τ=0.5 multi-seed for $DS on GPU $GPU complete."
