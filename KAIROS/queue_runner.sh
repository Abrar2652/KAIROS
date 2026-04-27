#!/usr/bin/env bash
# KAIROS experiment queue runner.
# Launches a fixed list of runs on a specified GPU sequentially.
# Usage: GPU=2 bash queue_runner.sh <queue_name>
#
# Queues:
#   gpu2_queue  : bitcoinotc canonical s={13,99}, then bitcoinotc fixed_tau s={42,7,13,99}
#   gpu4_queue  : bitotc canonical s={7,13,99}, then bitotc fixed_tau s={42,7,13,99}
#   gpu5_queue  : bitalpha canonical s={7,13,99}, then bitalpha fixed_tau s={42,7,13,99}
#   big_queue   : DBLP fixed_tau s=24, TAX51 fixed_tau s=24 (use GPU 6 or 7 after Reddit)

set -uo pipefail
cd "$(dirname "$0")"

GPU="${GPU:-2}"
QUEUE="${1:-}"

runs_dir="/nas/home/jahin/KAIROS/runs"

run() {
  # run <dataset> <task> <snaps> <views> <strat> <dl> <epochs> <seed> <ablation> [tau_val]
  local ds=$1 task=$2 snaps=$3 views=$4 strat=$5 dl=$6 ep=$7 seed=$8 abl=$9
  local tauval="${10:-0.5}"
  # Include task prefix in tag to avoid classification/anomaly collisions.
  local taskshort="clf"
  [[ "$task" == "anomaly_detection" ]] && taskshort="ano"
  local tag="${ds}_${taskshort}_${abl}_s${seed}"
  [[ "$abl" == "fixed_tau" ]] && tag="${ds}_${taskshort}_fixedtau${tauval/./}_s${seed}"
  [[ "$abl" == "none" ]] && tag="${ds}_${taskshort}_canonical_s${seed}"
  local log="${runs_dir}/${tag}.log"
  if grep -q "ablate-result" "$log" 2>/dev/null; then
    echo "[skip] $tag (already has result)"
    return
  fi
  echo "[run ] GPU=$GPU $tag"
  CUDA_VISIBLE_DEVICES=$GPU python3 -u ablate.py \
    --dataset "$ds" --task "$task" \
    --snapshots "$snaps" --views "$views" --strategy "$strat" \
    --dataloader_size "$dl" --GPU 0 --epochs "$ep" --seed "$seed" \
    --ablation "$abl" --tau_val "$tauval" \
    > "$log" 2>&1
  echo "[done] $tag"
}

case "$QUEUE" in
  gpu2_queue)
    # bitcoinotc canonical multi-seed + fixed_tau=0.5 multi-seed
    run bitcoinotc classification 4 3 sequential 64 25 13 none
    run bitcoinotc classification 4 3 sequential 64 25 99 none
    run bitcoinotc classification 4 3 sequential 64 25 42 fixed_tau 0.5
    run bitcoinotc classification 4 3 sequential 64 25 7  fixed_tau 0.5
    run bitcoinotc classification 4 3 sequential 64 25 13 fixed_tau 0.5
    run bitcoinotc classification 4 3 sequential 64 25 99 fixed_tau 0.5
    ;;
  gpu4_queue)
    # bitotc canonical multi-seed + fixed_tau=0.5 multi-seed
    run bitotc classification 4 4 random 4096 50 7  none
    run bitotc classification 4 4 random 4096 50 13 none
    run bitotc classification 4 4 random 4096 50 99 none
    run bitotc classification 4 4 random 4096 50 42 fixed_tau 0.5
    run bitotc classification 4 4 random 4096 50 7  fixed_tau 0.5
    run bitotc classification 4 4 random 4096 50 13 fixed_tau 0.5
    run bitotc classification 4 4 random 4096 50 99 fixed_tau 0.5
    ;;
  gpu5_queue)
    # bitalpha canonical multi-seed + fixed_tau=0.5 multi-seed (long)
    run bitalpha classification 6 4 sequential 4096 200 7  none
    run bitalpha classification 6 4 sequential 4096 200 13 none
    run bitalpha classification 6 4 sequential 4096 200 99 none
    run bitalpha classification 6 4 sequential 4096 200 42 fixed_tau 0.5
    run bitalpha classification 6 4 sequential 4096 200 7  fixed_tau 0.5
    run bitalpha classification 6 4 sequential 4096 200 13 fixed_tau 0.5
    run bitalpha classification 6 4 sequential 4096 200 99 fixed_tau 0.5
    ;;
  big_queue)
    # DBLP + TAX51 fixed_tau=0.5 check (no regression)
    run dblp classification 4 4 sequential 4096 200 24 fixed_tau 0.5
    run tax51 classification 8 5 random 4096 200 24 fixed_tau 0.5
    ;;
  anomaly_ablations)
    # no_koop_anomaly ablation on each dataset (S2 removed, S1+S3 only)
    run dblp       anomaly_detection 4 4 sequential 4096 200 24 no_koop_anomaly
    run bitcoinotc anomaly_detection 4 4 sequential 64   25  24 no_koop_anomaly
    run bitotc     anomaly_detection 4 4 sequential 4096 50  24 no_koop_anomaly
    run bitalpha   anomaly_detection 5 5 sequential 4096 100 24 no_koop_anomaly
    run tax51      anomaly_detection 4 4 sequential 4096 200 24 no_koop_anomaly
    ;;
  anomaly_seed42)
    # Anomaly detection with seed=42 (one extra seed for variance)
    run dblp       anomaly_detection 4 4 sequential 4096 200 42 none
    run bitcoinotc anomaly_detection 4 4 sequential 64   25  42 none
    run bitotc     anomaly_detection 4 4 sequential 4096 50  42 none
    run bitalpha   anomaly_detection 5 5 sequential 4096 100 42 none
    run tax51      anomaly_detection 4 4 sequential 4096 200 42 none
    ;;
  dblp_multiseed)
    # DBLP classification multi-seed (canonical, 4 additional seeds)
    run dblp classification 4 4 sequential 4096 200 42 none
    run dblp classification 4 4 sequential 4096 200 7  none
    run dblp classification 4 4 sequential 4096 200 13 none
    run dblp classification 4 4 sequential 4096 200 99 none
    ;;
  tax51_multiseed)
    # TAX51 classification multi-seed (canonical, 4 additional seeds)
    run tax51 classification 8 5 random 4096 200 42 none
    run tax51 classification 8 5 random 4096 200 7  none
    run tax51 classification 8 5 random 4096 200 13 none
    run tax51 classification 8 5 random 4096 200 99 none
    ;;
  alpha_sweep_bitcoin)
    # PPR α sweep on Bitcoin-family classification (seed 24)
    for a in 0.05 0.10 0.20 0.30; do
      tag1="bitcoinotc_alpha${a/./}_s24"
      log1="/nas/home/jahin/KAIROS/runs/${tag1}.log"
      if grep -q "ablate-result" "$log1" 2>/dev/null; then echo "skip $tag1"; continue; fi
      echo "run $tag1"
      CUDA_VISIBLE_DEVICES=$GPU python3 -u ablate.py \
        --dataset bitcoinotc --task classification \
        --snapshots 4 --views 3 --strategy sequential \
        --dataloader_size 64 --GPU 0 --epochs 25 --seed 24 \
        --ablation none --alpha $a > "$log1" 2>&1
    done
    for a in 0.05 0.10 0.20 0.30; do
      tag2="bitotc_alpha${a/./}_s24"
      log2="/nas/home/jahin/KAIROS/runs/${tag2}.log"
      if grep -q "ablate-result" "$log2" 2>/dev/null; then echo "skip $tag2"; continue; fi
      echo "run $tag2"
      CUDA_VISIBLE_DEVICES=$GPU python3 -u ablate.py \
        --dataset bitotc --task classification \
        --snapshots 4 --views 4 --strategy random \
        --dataloader_size 4096 --GPU 0 --epochs 50 --seed 24 \
        --ablation none --alpha $a > "$log2" 2>&1
    done
    ;;
  lambda_koop_sweep)
    # λ_koop sweep on anomaly (seed 24) — test sensitivity of anomaly S2 signal
    for lk in 0.0 0.5 2.0; do
      for ds_cfg in "bitcoinotc 4 4 sequential 64 25" "dblp 4 4 sequential 4096 200"; do
        read ds snaps views strat dl ep <<< "$ds_cfg"
        tag="${ds}_anomaly_lk${lk/./}_s24"
        log="/nas/home/jahin/KAIROS/runs/${tag}.log"
        if grep -q "ablate-result" "$log" 2>/dev/null; then echo "skip $tag"; continue; fi
        echo "run $tag"
        CUDA_VISIBLE_DEVICES=$GPU python3 -u ablate.py \
          --dataset "$ds" --task anomaly_detection \
          --snapshots "$snaps" --views "$views" --strategy "$strat" \
          --dataloader_size "$dl" --GPU 0 --epochs "$ep" --seed 24 \
          --ablation none --lambda_koop "$lk" > "$log" 2>&1
      done
    done
    ;;
  dblp_tax_fixedtau_multiseed)
    # DBLP + TAX51 fixed_tau=0.5 at seeds {42, 7, 13, 99}
    for s in 42 7 13 99; do
      run dblp classification 4 4 sequential 4096 200 $s fixed_tau 0.5
    done
    for s in 42 7 13 99; do
      run tax51 classification 8 5 random 4096 200 $s fixed_tau 0.5
    done
    ;;
  *)
    echo "Unknown queue: $QUEUE"
    echo "Available: gpu2_queue gpu4_queue gpu5_queue big_queue"
    exit 1
    ;;
esac

echo "Queue $QUEUE on GPU $GPU complete."
