#!/usr/bin/env bash
# KAIROS multi-seed on new datasets (MOOC, Arxiv).
# Runs AFTER reddit_nokoop_queue.sh finishes.
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

# Wait for prior queues to finish
while pgrep -f "reddit_nokoop_queue.sh\|reddit_multiseed_queue.sh\|fill_dashes_queue.sh" > /dev/null 2>&1; do
  sleep 60
done

# MOOC: 7,144 nodes, 411K edges, binary balanced (dropout)
# Arxiv: 169,343 nodes, 1.2M edges, 40-class subject
declare -A CFG_SNAPS=(  [mooc]=5 [arxiv]=4 )
declare -A CFG_VIEWS=(  [mooc]=4 [arxiv]=4 )
declare -A CFG_STRAT=(  [mooc]=random [arxiv]=sequential )
declare -A CFG_DL=(     [mooc]=4096   [arxiv]=4096 )
declare -A CFG_EP=(     [mooc]=100    [arxiv]=200 )

for DS in mooc arxiv; do
  echo "=== KAIROS multi-seed: $DS classification (GCN, tau=0.5) ==="
  for s in 24 42 7 13 99; do
    run_if_missing "${DS}_gcn_fixedtau05_s${s}" \
      $PY --dataset "$DS" --task classification \
      --snapshots "${CFG_SNAPS[$DS]}" --views "${CFG_VIEWS[$DS]}" \
      --strategy "${CFG_STRAT[$DS]}" --dataloader_size "${CFG_DL[$DS]}" \
      --GPU 0 --epochs "${CFG_EP[$DS]}" --seed "$s" \
      --ablation fixed_tau --tau_val 0.5 --backbone gcn
  done

  echo "=== KAIROS multi-seed: $DS anomaly detection ==="
  for s in 24 42 7 13 99; do
    run_if_missing "${DS}_ano_fixedtau05_s${s}" \
      $PY --dataset "$DS" --task anomaly_detection \
      --snapshots "${CFG_SNAPS[$DS]}" --views "${CFG_VIEWS[$DS]}" \
      --strategy "${CFG_STRAT[$DS]}" --dataloader_size "${CFG_DL[$DS]}" \
      --GPU 0 --epochs "${CFG_EP[$DS]}" --seed "$s" \
      --ablation fixed_tau --tau_val 0.5
  done
done

echo "=== new-dataset KAIROS queue complete on GPU $GPU ==="
