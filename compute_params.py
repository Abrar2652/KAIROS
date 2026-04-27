"""
KAIROS — parameter count and FLOPs estimate
=============================================
Counts trainable parameters in KAIROS vs a reference CLDG++ configuration.
Approximates forward FLOPs for a sample graph.

Usage:
  python3 compute_params.py
"""

import sys
import os

sys.path.insert(0, '/nas/home/jahin/KAIROS/KAIROS')
os.environ.setdefault('DGLBACKEND', 'pytorch')

import torch
from models import KairosEncoder, KoopmanHead


def fmt(n):
    if n >= 1e6: return f'{n/1e6:.2f}M'
    if n >= 1e3: return f'{n/1e3:.2f}K'
    return str(n)


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def params_breakdown(model):
    parts = {}
    for name, p in model.named_parameters():
        key = name.split('.')[0]
        parts.setdefault(key, 0)
        parts[key] += p.numel()
    return parts


def main():
    # Standard KAIROS config from run_experiments.py
    in_feats = 128   # positional encoding
    hidden_dim = 128
    n_layers = 2
    embed_dim = 64

    print('=' * 70)
    print('KAIROS encoder — trainable parameter count')
    print('=' * 70)
    print(f'in_feats={in_feats}, hidden_dim={hidden_dim}, '
          f'n_layers={n_layers}, embed_dim={embed_dim}')
    print()

    kairos = KairosEncoder(
        in_feats=in_feats, hidden_dim=hidden_dim,
        n_layers=n_layers, embed_dim=embed_dim, dropout=0.0,
    )
    total, trainable = count_params(kairos)
    print(f'Total params      : {fmt(total)}   ({total:,})')
    print(f'Trainable params  : {fmt(trainable)}   ({trainable:,})')
    print()

    breakdown = params_breakdown(kairos)
    print('Breakdown by submodule:')
    for k, v in sorted(breakdown.items(), key=lambda x: -x[1]):
        pct = 100 * v / total
        print(f'  {k:20s}  {fmt(v):>10s}  ({pct:5.1f}%)')
    print()

    # Compare to approximate CLDG++ (single GCN backbone, no projectors)
    # CLDG++ architecture (based on CLDG paper):
    #   - Input: 128d (features + PE) + S (128d PPR) concatenated = 256d
    #   - GCN layer 1: 256 → 128
    #   - GCN layer 2: 128 → 64
    #   - No projection heads, no Koopman, no dual encoder
    cldg_l1 = 256 * 128 + 128         # weight + bias
    cldg_l2 = 128 * 64 + 64
    cldg_total = cldg_l1 + cldg_l2
    print('=' * 70)
    print('CLDG++ reference (approx, single GCN, no projectors):')
    print(f'  Total params: {fmt(cldg_total)}   ({cldg_total:,})')
    print(f'  KAIROS/CLDG++ ratio: {trainable/cldg_total:.2f}x')
    print('=' * 70)
    print()

    # Reporting table
    print('Reporter-friendly summary:')
    print(f'| Method  | Backbone      | Projector | Koopman | Params    |')
    print(f'| CLDG++  | single GCN    | —         | —       | ~{fmt(cldg_total)}    |')
    print(f'| KAIROS  | dual GAT (orig+PPR) | SimCLR v2 | K: {embed_dim}×{embed_dim} | {fmt(trainable)} |')


if __name__ == '__main__':
    main()
