#!/usr/bin/env bash
# Launch simple baselines in PARALLEL on GPU 6 (sharing with CLDG++).
# Each baseline uses < 2 GB, so we can run many simultaneously.
# Final new-dataset set: MOOC + Arxiv + Elliptic (email_eu dropped).
set -u
cd "$(dirname "$0")"

GPU="${GPU:-6}"
RUNS="/nas/home/jahin/KAIROS/runs"
mkdir -p "$RUNS"

run_bg() {
  local method=$1; local ds=$2; local task=$3
  local tag="baseline_${method}_${ds}_${task}_s24"
  local log="${RUNS}/${tag}.log"
  if grep -q "\[baseline-result\]" "$log" 2>/dev/null; then
    echo "[skip] $tag"
    return 0
  fi
  rm -f "$log"
  echo "[bg  ] GPU=$GPU $tag"
  CUDA_VISIBLE_DEVICES=$GPU python3 -u /nas/home/jahin/KAIROS/KAIROS/simple_baselines.py \
    --method "$method" --dataset "$ds" --task "$task" --seed 24 --epochs 100 > "$log" 2>&1 &
}

# Wave 1: parallel classification on 3 new datasets x 6 methods = 18 runs
for method in lp gcn gat sage dgi ccassg; do
  for ds in mooc arxiv elliptic; do
    run_bg "$method" "$ds" classification
    sleep 5
  done
done

echo ""
echo "All classification baselines launched as background jobs."
echo "Running count: $(jobs -p | wc -l)"
wait
echo "=== Wave 1 classification complete ==="

# Wave 2: anomaly (18 more runs)
for method in lp gcn gat sage dgi ccassg; do
  for ds in mooc arxiv elliptic; do
    run_bg "$method" "$ds" anomaly_detection
    sleep 5
  done
done
wait
echo "=== Wave 2 anomaly complete ==="
