#!/usr/bin/env bash
# KAIROS 5-seed on Elliptic (classification + anomaly).
set -uo pipefail
cd "$(dirname "$0")"

GPU="${GPU:-6}"
RUNS="/nas/home/jahin/KAIROS/runs"
PY="python3 -u /nas/home/jahin/KAIROS/KAIROS/ablate.py"

run_if_missing() {
  local tag="$1"; shift
  local log="${RUNS}/${tag}.log"
  if grep -q "ablate-result" "$log" 2>/dev/null; then echo "[skip] $tag"; return 0; fi
  rm -f "$log" 2>/dev/null
  echo "[run ] GPU=$GPU $tag"
  CUDA_VISIBLE_DEVICES=$GPU "$@" > "$log" 2>&1
  echo "[done] $tag"
}

# Wait for prior queues
while pgrep -f "rerun_failed_queue.sh\|email_eu_queue.sh\|multiseed_newds.sh" > /dev/null 2>&1; do
  sleep 60
done

DS=elliptic
SNAPS=4; VIEWS=4; STRAT=sequential; DL=4096; EP=100

echo "=== KAIROS: $DS classification (GCN, tau=0.5) ==="
for s in 24 42 7 13 99; do
  run_if_missing "${DS}_gcn_fixedtau05_s${s}" \
    $PY --dataset "$DS" --task classification \
    --snapshots "$SNAPS" --views "$VIEWS" --strategy "$STRAT" \
    --dataloader_size "$DL" --GPU 0 --epochs "$EP" --seed "$s" \
    --ablation fixed_tau --tau_val 0.5 --backbone gcn
done

echo "=== KAIROS: $DS anomaly detection ==="
for s in 24 42 7 13 99; do
  run_if_missing "${DS}_ano_fixedtau05_s${s}" \
    $PY --dataset "$DS" --task anomaly_detection \
    --snapshots "$SNAPS" --views "$VIEWS" --strategy "$STRAT" \
    --dataloader_size "$DL" --GPU 0 --epochs "$EP" --seed "$s" \
    --ablation fixed_tau --tau_val 0.5
done

echo "=== Elliptic queue complete on GPU $GPU ==="
