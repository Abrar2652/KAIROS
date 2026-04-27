#!/usr/bin/env bash
# Master queue to fill all "pending" cells in TABLES.tex.
# Runs sequentially on GPU=${GPU:-6}. Ordered by priority.
set -uo pipefail
cd "$(dirname "$0")"

GPU="${GPU:-6}"
RUNS="/nas/home/jahin/KAIROS/runs"

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

PY="python3 -u /nas/home/jahin/KAIROS/KAIROS/ablate.py"

# ===== 1. Reddit GCN tuning test (seed 24) =====
echo "=== 1. Reddit GCN test (seed 24) ==="
run_if_missing "reddit_gcn_fixedtau05_s24" \
  $PY --dataset reddit --task classification \
  --snapshots 5 --views 4 --strategy random \
  --dataloader_size 4096 --GPU 0 --epochs 200 --seed 24 \
  --ablation fixed_tau --tau_val 0.5 --backbone gcn

# ===== 2. Tau-sweep fills (seed 24 only) =====
echo "=== 2. Tau-sweep fills ==="
# DBLP at tau = 0.2, 0.3, 0.7, 1.0 (already has 0.5)
for t in 02 03 07 10; do
  tv="0.${t#0}"; [ "$t" = "10" ] && tv="1.0"
  run_if_missing "dblp_fixedtau${t}_s24" \
    $PY --dataset dblp --task classification \
    --snapshots 4 --views 4 --strategy sequential \
    --dataloader_size 4096 --GPU 0 --epochs 200 --seed 24 \
    --ablation fixed_tau --tau_val "$tv"
done
# TAX51 at tau = 0.2, 0.3, 0.7, 1.0
for t in 02 03 07 10; do
  tv="0.${t#0}"; [ "$t" = "10" ] && tv="1.0"
  run_if_missing "tax51_fixedtau${t}_s24" \
    $PY --dataset tax51 --task classification \
    --snapshots 8 --views 5 --strategy random \
    --dataloader_size 4096 --GPU 0 --epochs 200 --seed 24 \
    --ablation fixed_tau --tau_val "$tv"
done
# Bitcoinotc and BITotc at tau=0.2 (missing)
run_if_missing "bitcoinotc_fixedtau02_s24" \
  $PY --dataset bitcoinotc --task classification \
  --snapshots 4 --views 3 --strategy sequential \
  --dataloader_size 64 --GPU 0 --epochs 25 --seed 24 \
  --ablation fixed_tau --tau_val 0.2
run_if_missing "bitotc_fixedtau02_s24" \
  $PY --dataset bitotc --task classification \
  --snapshots 4 --views 4 --strategy random \
  --dataloader_size 4096 --GPU 0 --epochs 50 --seed 24 \
  --ablation fixed_tau --tau_val 0.2

# ===== 3. Backbone dashes =====
echo "=== 3. Backbone fills ==="
# DBLP backbones (GCN, SGC, SAGE, H2GCN) - 5 seeds each
for bb in gcn sgc sage h2gcn; do
  for s in 24 42 7 13 99; do
    run_if_missing "dblp_${bb}_fixedtau05_s${s}" \
      $PY --dataset dblp --task classification \
      --snapshots 4 --views 4 --strategy sequential \
      --dataloader_size 4096 --GPU 0 --epochs 200 --seed "$s" \
      --ablation fixed_tau --tau_val 0.5 --backbone "$bb"
  done
done
# TAX51 backbones
for bb in gcn sgc sage h2gcn; do
  for s in 24 42 7 13 99; do
    run_if_missing "tax51_${bb}_fixedtau05_s${s}" \
      $PY --dataset tax51 --task classification \
      --snapshots 8 --views 5 --strategy random \
      --dataloader_size 4096 --GPU 0 --epochs 200 --seed "$s" \
      --ablation fixed_tau --tau_val 0.5 --backbone "$bb"
  done
done
# Bitcoinotc: SGC, SAGE, H2GCN (has GCN)
for bb in sgc sage h2gcn; do
  for s in 24 42 7 13 99; do
    run_if_missing "bitcoinotc_${bb}_fixedtau05_s${s}" \
      $PY --dataset bitcoinotc --task classification \
      --snapshots 4 --views 3 --strategy sequential \
      --dataloader_size 64 --GPU 0 --epochs 25 --seed "$s" \
      --ablation fixed_tau --tau_val 0.5 --backbone "$bb"
  done
done
# BITalpha: H2GCN only (has GCN/SGC/SAGE)
for s in 24 42 7 13 99; do
  run_if_missing "bitalpha_h2gcn_fixedtau05_s${s}" \
    $PY --dataset bitalpha --task classification \
    --snapshots 6 --views 4 --strategy sequential \
    --dataloader_size 4096 --GPU 0 --epochs 200 --seed "$s" \
    --ablation fixed_tau --tau_val 0.5 --backbone h2gcn
done

# ===== 4. Koopman ablation Reddit (seed 24) =====
echo "=== 4. Koopman Reddit anomaly ==="
run_if_missing "reddit_no_koop_anomaly_s24" \
  $PY --dataset reddit --task anomaly_detection \
  --snapshots 5 --views 4 --strategy random \
  --dataloader_size 4096 --GPU 0 --epochs 200 --seed 24 \
  --ablation no_koop_anomaly --tau_val 0.5

echo "=== fill_dashes_queue.sh complete on GPU $GPU ==="
