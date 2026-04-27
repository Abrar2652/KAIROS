#!/usr/bin/env bash
# KAIROS — per-dataset hyperparameter tuning on validation Acc.
#
# For each Bitcoin-family dataset, sweep hidden_dim ∈ {64, 128},
# n_layers ∈ {1, 2}, lr is fixed at 4e-3 (ablate.py doesn't expose lr),
# τ ∈ {0.07-learnable, 0.5-fixed} = 8 configs per dataset at seed=24.
#
# Runs sequentially on the specified GPU. Skips entries with existing results.
# Reads validation Acc from the log (best_val_acc * ... field).
#
# Usage: GPU=2 bash per_dataset_tune.sh <dataset>

set -uo pipefail
cd "$(dirname "$0")"

GPU="${GPU:-2}"
DS="${1:-bitcoinotc}"

# Per-dataset base configs
case "$DS" in
  bitcoinotc) SNAPS=4; VIEWS=3; STRAT=sequential; DL=64;   EP=25  ;;
  bitotc)     SNAPS=4; VIEWS=4; STRAT=random;     DL=4096; EP=50  ;;
  bitalpha)   SNAPS=6; VIEWS=4; STRAT=sequential; DL=4096; EP=200 ;;
  *) echo "Unknown dataset: $DS"; exit 1 ;;
esac

runs_dir="/nas/home/jahin/KAIROS/runs"
mkdir -p "$runs_dir"

for hidden in 64 128; do
  for nlayers in 1 2; do
    for tau_mode in canonical fixed05; do
      if [ "$tau_mode" = canonical ]; then
        abl="none"; tauflag=""
      else
        abl="fixed_tau"; tauflag="--tau_val 0.5"
      fi
      tag="${DS}_tune_h${hidden}_l${nlayers}_${tau_mode}_s24"
      log="${runs_dir}/${tag}.log"
      if grep -q "ablate-result" "$log" 2>/dev/null; then
        echo "[skip] $tag (exists)"
        continue
      fi
      echo "[run ] GPU=$GPU $tag"
      CUDA_VISIBLE_DEVICES=$GPU python3 -u /nas/home/jahin/KAIROS/KAIROS/ablate.py \
        --dataset "$DS" --task classification \
        --hidden_dim "$hidden" --n_layers "$nlayers" \
        --snapshots "$SNAPS" --views "$VIEWS" --strategy "$STRAT" \
        --dataloader_size "$DL" --GPU 0 --epochs "$EP" --seed 24 \
        --ablation "$abl" $tauflag > "$log" 2>&1
      echo "[done] $tag"
    done
  done
done

echo "Per-dataset tune for $DS on GPU $GPU complete."
