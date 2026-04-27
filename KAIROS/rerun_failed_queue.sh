#!/usr/bin/env bash
# Re-run seeds that died due to disk-full
set -uo pipefail
cd "$(dirname "$0")"

GPU="${GPU:-6}"
RUNS="/nas/home/jahin/KAIROS/runs"
PY="python3 -u /nas/home/jahin/KAIROS/KAIROS/ablate.py"

run_if_missing() {
  local tag="$1"; shift
  local log="${RUNS}/${tag}.log"
  if grep -q "ablate-result" "$log" 2>/dev/null; then
    echo "[skip] $tag"; return 0
  fi
  # Remove partial log (died mid-run)
  rm -f "$log" 2>/dev/null
  echo "[run ] GPU=$GPU $tag"
  CUDA_VISIBLE_DEVICES=$GPU "$@" > "$log" 2>&1
  echo "[done] $tag"
}

echo "=== Re-run failed Reddit anomaly λ=1 ==="
run_if_missing "reddit_ano_fixedtau05_s99" \
  $PY --dataset reddit --task anomaly_detection \
  --snapshots 5 --views 4 --strategy random \
  --dataloader_size 4096 --GPU 0 --epochs 200 --seed 99 \
  --ablation fixed_tau --tau_val 0.5

echo "=== Re-run failed Reddit no-Koopman anomaly λ=0 ==="
for s in 42 7 13 99; do
  run_if_missing "reddit_no_koop_anomaly_s${s}" \
    $PY --dataset reddit --task anomaly_detection \
    --snapshots 5 --views 4 --strategy random \
    --dataloader_size 4096 --GPU 0 --epochs 200 --seed "$s" \
    --ablation no_koop_anomaly
done

echo "=== Re-run Arxiv classification s99 ==="
run_if_missing "arxiv_gcn_fixedtau05_s99" \
  $PY --dataset arxiv --task classification \
  --snapshots 4 --views 4 --strategy sequential \
  --dataloader_size 4096 --GPU 0 --epochs 200 --seed 99 \
  --ablation fixed_tau --tau_val 0.5 --backbone gcn

echo "=== Run Arxiv anomaly all 5 seeds ==="
for s in 24 42 7 13 99; do
  run_if_missing "arxiv_ano_fixedtau05_s${s}" \
    $PY --dataset arxiv --task anomaly_detection \
    --snapshots 4 --views 4 --strategy sequential \
    --dataloader_size 4096 --GPU 0 --epochs 200 --seed "$s" \
    --ablation fixed_tau --tau_val 0.5
done

echo "=== rerun_failed_queue complete ==="
