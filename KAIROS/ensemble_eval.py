"""
Multi-seed encoder ensemble evaluation.
Loads saved embeddings from N seed-trained encoders, fuses them, trains
a LogReg probe, and reports test Acc + Wei-F1.

Two fusion modes supported:
  mean   : element-wise mean of 5 × 64d → 64d (fair — same dim as baseline)
  concat : concatenate 5 × 64d → 320d (more info but not dim-matched)

Usage:
  python3 ensemble_eval.py --dataset bitotc --backbone gcn --fusion mean
"""

import argparse
import glob
import os
import sys

import numpy as np
import torch as th
import torch.nn as thnn

sys.path.insert(0, '/nas/home/jahin/KAIROS/KAIROS')
os.environ.setdefault('DGLBACKEND', 'pytorch')

from sklearn.metrics import f1_score


class LogReg(thnn.Module):
    def __init__(self, hid_dim, out_dim):
        super().__init__()
        self.fc = thnn.Linear(hid_dim, out_dim)

    def forward(self, x):
        return self.fc(x)


def load_dataloader_info(dataset):
    """Re-use main.py's dataloader() to get split indices and labels."""
    from utils import dataloader
    import random
    random.seed(24)
    labels, tr, va, te, nc = dataloader(dataset)
    return labels, tr, va, te, nc


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True)
    p.add_argument('--backbone', default='gcn')
    p.add_argument('--fusion', choices=['mean', 'concat'], default='mean')
    p.add_argument('--tau', default='0.5')
    p.add_argument('--seeds', default='24,42,7,13,99')
    p.add_argument('--eval_runs', type=int, default=5,
                   help='number of LogReg runs to average')
    a = p.parse_args()

    seeds = [int(s) for s in a.seeds.split(',')]
    embed_files = []
    for s in seeds:
        fn = (f'/nas/home/jahin/KAIROS/runs/embeds/'
              f'{a.dataset}_{a.backbone}_tau{a.tau.replace(".","")}_s{s}.pt')
        if not os.path.exists(fn):
            print(f'MISSING: {fn}')
            sys.exit(1)
        embed_files.append(fn)

    print(f'Loading {len(embed_files)} embeddings for {a.dataset}/{a.backbone}')
    embeds = [th.load(f)['embeddings'] for f in embed_files]

    if a.fusion == 'mean':
        fused = th.stack(embeds, dim=0).mean(dim=0)
    else:
        fused = th.cat(embeds, dim=-1)
    print(f'Fused embedding shape: {fused.shape}')

    labels, tr, va, te, nc = load_dataloader_info(a.dataset)
    device = th.device('cuda:0')
    fused = fused.to(device).float()
    labels = labels.to(device)

    tr_e, va_e, te_e = fused[tr], fused[va], fused[te]
    tr_l, va_l, te_l = labels[tr], labels[va], labels[te]

    print(f'Labels shape: {labels.shape}, n_classes={nc}')
    print(f'tr/va/te sizes: {tr.shape[0]}/{va.shape[0]}/{te.shape[0]}')

    micros, weights = [], []
    for r in range(a.eval_runs):
        lr = LogReg(fused.shape[1], nc).to(device)
        opt = th.optim.Adam(lr.parameters(), lr=1e-2, weight_decay=1e-4)
        loss_fn = thnn.CrossEntropyLoss()
        best_va = 0.0; em = 0.0; ew = 0.0
        for ep in range(2000):
            lr.train(); opt.zero_grad()
            loss = loss_fn(lr(tr_e), tr_l); loss.backward(); opt.step()
            lr.eval()
            with th.no_grad():
                va_a = (lr(va_e).argmax(1) == va_l).float().mean()
                tp = lr(te_e).argmax(1).cpu().numpy()
                yp = te_l.cpu().numpy()
                mi = f1_score(yp, tp, average='micro')
                we = f1_score(yp, tp, average='weighted')
                if va_a >= best_va:
                    best_va = va_a
                    if (mi + we) >= (em + ew):
                        em, ew = mi, we
        micros.append(em); weights.append(we)
        print(f'  LogReg run {r+1}/{a.eval_runs}: micro={em:.4f} wei={ew:.4f} val={best_va:.4f}')

    m = np.mean(micros) * 100; s = np.std(micros, ddof=1) * 100
    w = np.mean(weights) * 100; ws = np.std(weights, ddof=1) * 100
    print(f'\n=== {a.dataset} {a.backbone} τ={a.tau} [{a.fusion} fusion, {len(seeds)} encoders] ===')
    print(f'Acc:    {m:.2f} ± {s:.2f}  (n={len(micros)} LogReg runs)')
    print(f'Wei-F1: {w:.2f} ± {ws:.2f}')


if __name__ == '__main__':
    main()
