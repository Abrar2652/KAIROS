"""Minimal CLDG anomaly detection runner.

CLDG anomaly = single GCN encoder + InfoNCE temporal contrastive loss +
cross-snapshot cosine-distance anomaly score (matches what the CLDG++ paper
reports as the "CLDG" baseline in Table 4).

Usage:
  python3 cldg_anomaly.py --dataset mooc --snapshots 5 --views 4 --epochs 100 --seed 24
"""

import argparse
import copy
import random
import sys

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch import GraphConv
from sklearn.metrics import roc_auc_score

sys.path.insert(0, '/nas/home/jahin/KAIROS/KAIROS')
from utils import load_to_dgl_graph, sampling_layer


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); th.manual_seed(seed)
    if th.cuda.is_available(): th.cuda.manual_seed_all(seed)
    th.backends.cudnn.deterministic = True


class CLDGEncoder(nn.Module):
    """Single GCN encoder used in CLDG."""
    def __init__(self, in_feat, hid, n_layers, n_classes):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GraphConv(in_feat, hid, norm='both', activation=F.relu, allow_zero_in_degree=True))
        for _ in range(n_layers - 2):
            self.layers.append(GraphConv(hid, hid, norm='both', activation=F.relu, allow_zero_in_degree=True))
        self.layers.append(GraphConv(hid, n_classes, norm='both', allow_zero_in_degree=True))
    def forward(self, g, x):
        for l in self.layers:
            x = l(g, x)
        return x


def temporal_contrast_loss(emb_i, emb_j, tau=0.5, batch_size=4096):
    """Symmetric InfoNCE between matched-node embeddings of two snapshots.
    Uses mini-batches for large graphs to avoid O(N^2) memory."""
    N = emb_i.size(0)
    h1 = F.normalize(emb_i, dim=1)
    h2 = F.normalize(emb_j, dim=1)
    if N <= batch_size:
        logits = h1 @ h2.t() / tau
        targets = th.arange(N, device=h1.device)
        return 0.5 * (F.cross_entropy(logits, targets) + F.cross_entropy(logits.t(), targets))
    # Mini-batch InfoNCE: sample a random subset of anchors, contrast against subset of negatives
    perm = th.randperm(N, device=h1.device)[:batch_size]
    a1 = h1[perm]; a2 = h2[perm]
    logits = a1 @ a2.t() / tau
    targets = th.arange(batch_size, device=h1.device)
    return 0.5 * (F.cross_entropy(logits, targets) + F.cross_entropy(logits.t(), targets))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True)
    p.add_argument('--snapshots', type=int, default=5)
    p.add_argument('--views', type=int, default=4)
    p.add_argument('--strategy', type=str, default='random')
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--hidden_dim', type=int, default=128)
    p.add_argument('--n_classes', type=int, default=64)
    p.add_argument('--n_layers', type=int, default=2)
    p.add_argument('--seed', type=int, default=24)
    p.add_argument('--gpu', type=int, default=0)
    args = p.parse_args()

    set_seed(args.seed)
    device = th.device(f'cuda:{args.gpu}' if th.cuda.is_available() else 'cpu')

    # Load anomaly-injected graph (uses KAIROS utils)
    graph, feat_list, alab = load_to_dgl_graph(args.dataset, task='anomaly_detection', snapshots=args.snapshots)
    in_feat = feat_list[0].shape[1]

    encoder = CLDGEncoder(in_feat, args.hidden_dim, args.n_layers, args.n_classes).to(device)
    optimizer = th.optim.Adam(encoder.parameters(), lr=4e-3, weight_decay=5e-4)

    edges_time = graph.edata['time'].tolist()
    span = max(edges_time) - min(edges_time)
    max_t, min_t = max(edges_time), min(edges_time)

    print(f'[CLDG-anomaly] dataset={args.dataset} seed={args.seed} epochs={args.epochs}')

    best_loss, best_state = float('inf'), None
    for epoch in range(args.epochs):
        encoder.train()
        T, T_idx = sampling_layer(args.snapshots, args.views, span, args.strategy)
        # Build temporal subgraphs
        subs, sub_feats = [], []
        for start, idx in zip(T, T_idx):
            end = min(start + span / args.snapshots, max_t)
            mask = (graph.edata['time'] >= start) & (graph.edata['time'] <= end)
            sg = dgl.edge_subgraph(graph, mask, relabel_nodes=False)
            sg = dgl.add_self_loop(sg)
            subs.append(sg.to(device))
            sub_feats.append(feat_list[idx].to(device))

        # Forward each view
        emb_views = [encoder(subs[i], sub_feats[i]) for i in range(args.views)]

        # Pairwise temporal contrastive loss
        loss = th.tensor(0.0, device=device)
        n_pairs = 0
        for i in range(args.views):
            for j in range(i + 1, args.views):
                loss = loss + temporal_contrast_loss(emb_views[i], emb_views[j])
                n_pairs += 1
        loss = loss / max(n_pairs, 1)

        optimizer.zero_grad(); loss.backward(); optimizer.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_state = {k: v.clone() for k, v in encoder.state_dict().items()}

    encoder.load_state_dict(best_state); encoder.eval()

    # Evaluation: compute embeddings on each snapshot, then cross-snapshot cosine dist + std
    snapshot_embs = []
    with th.no_grad():
        # Use full graph for each snapshot, with snapshot-specific feat
        full_g = dgl.add_self_loop(graph).to(device)
        for k in range(args.snapshots):
            h = encoder(full_g, feat_list[k].to(device))
            snapshot_embs.append(h)

    # CLDG++-style anomaly score
    score = None
    for i in range(len(snapshot_embs)):
        for j in range(i + 1, len(snapshot_embs)):
            sim = 1 - F.cosine_similarity(snapshot_embs[i], snapshot_embs[j], dim=1)
            sim = sim.unsqueeze(1)
            score = sim if score is None else th.cat((score, sim), dim=1)
    ano_score = score.mean(dim=1) + score.std(dim=1)
    auc = roc_auc_score(alab.squeeze(), ano_score.cpu().numpy())
    print(f'[cldg-anomaly-result] dataset={args.dataset} seed={args.seed} AUC={auc:.4f}')


if __name__ == '__main__':
    main()
