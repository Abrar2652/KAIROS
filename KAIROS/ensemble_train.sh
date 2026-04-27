#!/usr/bin/env bash
# Re-run 5 seeds of a backbone + τ=0.5 with embedding saving for ensemble eval.
# Usage: GPU=X DS=<ds> BB=<backbone> bash ensemble_train.sh
set -uo pipefail
cd "$(dirname "$0")"

GPU="${GPU:-0}"
DS="${DS:-bitotc}"
BB="${BB:-gcn}"
TAUSHORT="05"

case "$DS" in
  bitcoinotc) SNAPS=4; VIEWS=3; STRAT=sequential; DL=64;   EP=25  ;;
  bitotc)     SNAPS=4; VIEWS=4; STRAT=random;     DL=4096; EP=50  ;;
  bitalpha)   SNAPS=6; VIEWS=4; STRAT=sequential; DL=4096; EP=200 ;;
  dblp)       SNAPS=4; VIEWS=4; STRAT=sequential; DL=4096; EP=200 ;;
  tax51)      SNAPS=8; VIEWS=5; STRAT=random;     DL=4096; EP=200 ;;
  *) echo "unknown $DS"; exit 1 ;;
esac

runs="/nas/home/jahin/KAIROS/runs"
embeds="/nas/home/jahin/KAIROS/runs/embeds"
mkdir -p "$embeds"

for s in 24 42 7 13 99; do
  tag="${DS}_${BB}_tau${TAUSHORT}_s${s}_ensembletrain"
  log="${runs}/${tag}.log"
  embed="${embeds}/${DS}_${BB}_tau${TAUSHORT}_s${s}.pt"
  if [ -f "$embed" ]; then
    echo "[skip] $tag (embed exists)"; continue
  fi
  echo "[run ] GPU=$GPU $tag"
  CUDA_VISIBLE_DEVICES=$GPU python3 -u /nas/home/jahin/KAIROS/KAIROS/ablate.py \
    --dataset "$DS" --task classification \
    --snapshots "$SNAPS" --views "$VIEWS" --strategy "$STRAT" \
    --dataloader_size "$DL" --GPU 0 --epochs "$EP" --seed "$s" \
    --ablation fixed_tau --tau_val 0.5 \
    --backbone "$BB" \
    --save_embed_path "$embed" > "$log" 2>&1
  echo "[done] $tag → $embed"
done
echo "Ensemble training for $DS/$BB on GPU $GPU complete."
