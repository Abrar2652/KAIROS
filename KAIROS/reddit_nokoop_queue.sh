#!/usr/bin/env bash
# Reddit anomaly with λ_koop=0 (no_koop_anomaly mode) multi-seed.
# Runs AFTER reddit_multiseed_queue.sh finishes.
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

# Wait until prior queue finishes
while pgrep -f "reddit_multiseed_queue.sh" > /dev/null 2>&1; do
  sleep 60
done

echo "=== Reddit anomaly with λ_koop=0 multi-seed (4 more seeds) ==="
for s in 42 7 13 99; do
  run_if_missing "reddit_no_koop_anomaly_s${s}" \
    $PY --dataset reddit --task anomaly_detection \
    --snapshots 5 --views 4 --strategy random \
    --dataloader_size 4096 --GPU 0 --epochs 200 --seed "$s" \
    --ablation no_koop_anomaly
done

echo "=== Reddit λ_koop=0 queue complete on GPU $GPU ==="
