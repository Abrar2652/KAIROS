#!/usr/bin/env bash
# Reddit multi-seed: GCN classification (4 remaining seeds) + anomaly (4 remaining seeds).
# Runs sequentially on GPU=${GPU:-6}. Each run ~3.5h.
set -uo pipefail
cd "$(dirname "$0")"

GPU="${GPU:-6}"
RUNS="/nas/home/jahin/KAIROS/runs"
PY="python3 -u /nas/home/jahin/KAIROS/KAIROS/ablate.py"

run_if_missing() {
  local tag="$1"; shift
  local log="${RUNS}/${tag}.log"
  if grep -q "ablate-result" "$log" 2>/dev/null; then
    echo "[skip] $tag"
    return 0
  fi
  echo "[run ] GPU=$GPU $tag"
  CUDA_VISIBLE_DEVICES=$GPU "$@" > "$log" 2>&1
  echo "[done] $tag"
}

# Wait until the prior fill_dashes queue is done (crude guard)
while pgrep -f "fill_dashes_queue.sh" > /dev/null 2>&1; do
  sleep 60
done

echo "=== Reddit GCN classification multi-seed (4 more seeds) ==="
for s in 42 7 13 99; do
  run_if_missing "reddit_gcn_fixedtau05_s${s}" \
    $PY --dataset reddit --task classification \
    --snapshots 5 --views 4 --strategy random \
    --dataloader_size 4096 --GPU 0 --epochs 200 --seed "$s" \
    --ablation fixed_tau --tau_val 0.5 --backbone gcn
done

echo "=== Reddit anomaly multi-seed (4 more seeds at tau=0.5) ==="
for s in 42 7 13 99; do
  run_if_missing "reddit_ano_fixedtau05_s${s}" \
    $PY --dataset reddit --task anomaly_detection \
    --snapshots 5 --views 4 --strategy random \
    --dataloader_size 4096 --GPU 0 --epochs 200 --seed "$s" \
    --ablation fixed_tau --tau_val 0.5
done

echo "=== Reddit multi-seed queue complete on GPU $GPU ==="
