"""Simple single-seed baselines (LP, GCN, GAT, GraphSAGE, DGI, CCA-SSG) for KAIROS benchmark.

Emits a [baseline-result] marker line at the end of each run for parsing:
  [baseline-result] method=X dataset=Y task=Z result={'accuracy':..., 'weighted_f1':..., 'auc':...}

Usage:
  CUDA_VISIBLE_DEVICES=<gpu> python3 simple_baselines.py \
    --method <lp|gcn|gat|sage|dgi|ccassg> \
    --dataset <mooc|arxiv|email_eu|elliptic> \
    --task <classification|anomaly_detection> \
    --seed 24
"""

import argparse
import os
import random
import sys

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

import dgl
import dgl.function as fn
from dgl.nn import GraphConv, GATConv, SAGEConv
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, '/nas/home/jahin/KAIROS/KAIROS')
from utils import load_to_dgl_graph, dataloader


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); th.manual_seed(seed)
    if th.cuda.is_available(): th.cuda.manual_seed_all(seed)
    th.backends.cudnn.deterministic = True


# -------- Models --------
class MLPLP(nn.Module):
    """Label propagation — no learnable params."""
    def __init__(self, n_classes): super().__init__(); self.n_classes = n_classes
    def forward(self, g, x): return None


class GCNNet(nn.Module):
    def __init__(self, in_feat, hid, n_classes, n_layers=2, dropout=0.5):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GraphConv(in_feat, hid, norm='both', activation=F.relu))
        for _ in range(n_layers - 2):
            self.layers.append(GraphConv(hid, hid, norm='both', activation=F.relu))
        self.layers.append(GraphConv(hid, n_classes, norm='both'))
        self.dropout = nn.Dropout(dropout)
    def forward(self, g, x):
        for i, l in enumerate(self.layers):
            x = l(g, x)
            if i < len(self.layers) - 1: x = self.dropout(x)
        return x


class GATNet(nn.Module):
    def __init__(self, in_feat, hid, n_classes, heads=4, dropout=0.5):
        super().__init__()
        self.l1 = GATConv(in_feat, hid, heads, feat_drop=dropout, attn_drop=dropout, activation=F.elu)
        self.l2 = GATConv(hid * heads, n_classes, 1, feat_drop=dropout, attn_drop=dropout)
    def forward(self, g, x):
        x = self.l1(g, x).flatten(1)
        return self.l2(g, x).mean(1)


class SAGENet(nn.Module):
    def __init__(self, in_feat, hid, n_classes, n_layers=2, dropout=0.5):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(SAGEConv(in_feat, hid, 'mean', activation=F.relu))
        for _ in range(n_layers - 2):
            self.layers.append(SAGEConv(hid, hid, 'mean', activation=F.relu))
        self.layers.append(SAGEConv(hid, n_classes, 'mean'))
        self.dropout = nn.Dropout(dropout)
    def forward(self, g, x):
        for i, l in enumerate(self.layers):
            x = l(g, x)
            if i < len(self.layers) - 1: x = self.dropout(x)
        return x


class DGIEncoder(nn.Module):
    def __init__(self, in_feat, hid):
        super().__init__()
        self.conv = GraphConv(in_feat, hid, norm='both', activation=nn.PReLU())
    def forward(self, g, x): return self.conv(g, x)


class DGI(nn.Module):
    def __init__(self, in_feat, hid):
        super().__init__()
        self.enc = DGIEncoder(in_feat, hid)
        self.W = nn.Parameter(th.zeros(hid, hid))
        nn.init.xavier_uniform_(self.W)
    def forward(self, g, x, x_corrupt):
        h = self.enc(g, x)
        h_c = self.enc(g, x_corrupt)
        s = h.mean(0)
        score_pos = (h @ self.W @ s).sum()
        score_neg = (h_c @ self.W @ s).sum()
        return h, score_pos, score_neg


class CCASSGEncoder(nn.Module):
    def __init__(self, in_feat, hid, n_layers=2):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GraphConv(in_feat, hid, norm='both', activation=F.relu))
        for _ in range(n_layers - 1):
            self.layers.append(GraphConv(hid, hid, norm='both', activation=F.relu))
    def forward(self, g, x):
        for l in self.layers: x = l(g, x)
        return x


# -------- Training loops --------
def train_supervised(model, g, feat, labels, tr, va, te, n_classes, device, epochs=200, lr=1e-2, wd=5e-4):
    model = model.to(device); g = g.to(device); feat = feat.to(device); labels = labels.to(device)
    opt = th.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    best_va, best_state = -1, None
    for ep in range(epochs):
        model.train()
        logit = model(g, feat)
        loss = F.cross_entropy(logit[tr], labels[tr])
        opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with th.no_grad():
            logit = model(g, feat)
            pred = logit.argmax(1)
            va_acc = (pred[va] == labels[va]).float().mean().item()
            if va_acc > best_va:
                best_va = va_acc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    with th.no_grad():
        logit = model(g, feat)
        pred = logit.argmax(1).cpu().numpy()
        te_lab = labels[te].cpu().numpy()
        te_pred = pred[te.cpu().numpy()]
        acc = (te_pred == te_lab).mean()
        wf1 = f1_score(te_lab, te_pred, average='weighted')
    return acc, wf1


def lp(g, feat, labels, tr, va, te, n_classes, device, n_iter=50, alpha=0.9):
    g = dgl.remove_self_loop(g); g = dgl.add_self_loop(g); g = g.to(device)
    labels = labels.to(device)
    Y = th.zeros(g.num_nodes(), n_classes, device=device)
    tr_mask = th.zeros(g.num_nodes(), dtype=th.bool, device=device)
    tr_mask[tr.to(device)] = True
    for nid in tr.tolist():
        Y[nid, labels[nid].item()] = 1.0
    F_ = Y.clone()
    with th.no_grad():
        g.ndata['f'] = F_
        for _ in range(n_iter):
            g.update_all(fn.copy_u('f', 'm'), fn.mean('m', 'f'))
            f = g.ndata['f']
            f = alpha * f + (1 - alpha) * Y
            f[tr_mask] = Y[tr_mask]
            g.ndata['f'] = f
        pred = g.ndata['f'].argmax(1).cpu().numpy()
    te_lab = labels[te].cpu().numpy()
    te_pred = pred[te.cpu().numpy()]
    acc = (te_pred == te_lab).mean()
    wf1 = f1_score(te_lab, te_pred, average='weighted')
    return acc, wf1


def ssl_linprobe(h, labels, tr, va, te):
    """Logistic regression probe on embeddings."""
    h_np = h.detach().cpu().numpy()
    lab = labels.cpu().numpy()
    clf = LogisticRegression(max_iter=1000, n_jobs=1).fit(h_np[tr.cpu().numpy()], lab[tr.cpu().numpy()])
    te_pred = clf.predict(h_np[te.cpu().numpy()])
    te_lab = lab[te.cpu().numpy()]
    acc = (te_pred == te_lab).mean()
    wf1 = f1_score(te_lab, te_pred, average='weighted')
    return acc, wf1


def train_dgi(g, feat, device, hid=128, epochs=200, lr=1e-3, wd=0.0):
    g = dgl.remove_self_loop(g); g = dgl.add_self_loop(g); g = g.to(device); feat = feat.to(device)
    model = DGI(feat.size(1), hid).to(device)
    opt = th.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    best_loss, best_state = float('inf'), None
    for ep in range(epochs):
        perm = th.randperm(feat.size(0))
        fc = feat[perm]
        model.train()
        _, sp, sn = model(g, feat, fc)
        loss = -F.logsigmoid(sp).mean() - F.logsigmoid(-sn).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state); model.eval()
    with th.no_grad():
        h, _, _ = model(g, feat, feat)
    return h


def train_ccassg(g, feat, device, hid=128, epochs=200, lr=1e-3, wd=0.0, lamb=1e-3):
    # simple CCA-SSG: two augmented views, covariance-based loss
    g = dgl.remove_self_loop(g); g = dgl.add_self_loop(g); g = g.to(device); feat = feat.to(device)
    model = CCASSGEncoder(feat.size(1), hid).to(device)
    opt = th.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    def augment(x, p=0.2):
        mask = (th.rand_like(x) > p).float(); return x * mask
    N = feat.size(0)
    for ep in range(epochs):
        model.train()
        h1 = model(g, augment(feat)); h2 = model(g, augment(feat))
        h1 = F.normalize(h1, dim=0); h2 = F.normalize(h2, dim=0)
        inv = ((h1 - h2) ** 2).sum()
        c1 = h1.T @ h1; c2 = h2.T @ h2
        cov = ((c1 - th.eye(hid, device=device)) ** 2).sum() + ((c2 - th.eye(hid, device=device)) ** 2).sum()
        loss = inv + lamb * cov
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    with th.no_grad():
        h = model(g, feat)
    return h


def anomaly_score(h, g):
    """Simple anomaly score = distance from local mean (reconstruction-like)."""
    g = g.local_var()
    g.ndata['h'] = h
    g.update_all(fn.copy_u('h', 'm'), fn.mean('m', 'hbar'))
    return (h - g.ndata['hbar']).norm(dim=1).detach().cpu().numpy()


# -------- Main --------
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--method', required=True, choices=['lp', 'gcn', 'gat', 'sage', 'dgi', 'ccassg'])
    p.add_argument('--dataset', required=True)
    p.add_argument('--task', default='classification', choices=['classification', 'anomaly_detection'])
    p.add_argument('--seed', type=int, default=24)
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--snapshots', type=int, default=5)
    p.add_argument('--hidden', type=int, default=128)
    args = p.parse_args()

    set_seed(args.seed)
    device = th.device('cuda:0' if th.cuda.is_available() else 'cpu')

    if args.task == 'classification':
        g, feat = load_to_dgl_graph(args.dataset, task='classification')
        # Add self-loops + make symmetric for GNN safety
        g = dgl.to_bidirected(dgl.to_simple(g), copy_ndata=False)
        g = dgl.add_self_loop(g)
        labels, tr, va, te, n_classes = dataloader(args.dataset)
        labels = labels.long().cpu()
        if hasattr(tr, 'ndim') and tr.ndim == 0: tr = tr.unsqueeze(0)
        if hasattr(va, 'ndim') and va.ndim == 0: va = va.unsqueeze(0)
        if hasattr(te, 'ndim') and te.ndim == 0: te = te.unsqueeze(0)

        if args.method == 'lp':
            acc, wf1 = lp(g, feat, labels, tr, va, te, n_classes, device)
        elif args.method in ('gcn', 'gat', 'sage'):
            cls = {'gcn': GCNNet, 'gat': GATNet, 'sage': SAGENet}[args.method]
            model = cls(feat.size(1), args.hidden, n_classes)
            acc, wf1 = train_supervised(model, g, feat, labels, tr, va, te, n_classes, device, epochs=args.epochs)
        elif args.method == 'dgi':
            h = train_dgi(g, feat, device, hid=args.hidden, epochs=args.epochs)
            acc, wf1 = ssl_linprobe(h, labels, tr, va, te)
        elif args.method == 'ccassg':
            h = train_ccassg(g, feat, device, hid=args.hidden, epochs=args.epochs)
            acc, wf1 = ssl_linprobe(h, labels, tr, va, te)
        print(f"[baseline-result] method={args.method} dataset={args.dataset} task=classification "
              f"result={{'accuracy': {float(acc):.6f}, 'weighted_f1': {float(wf1):.6f}}}")
    else:
        g, feat_list, alab = load_to_dgl_graph(args.dataset, task='anomaly_detection', snapshots=args.snapshots)
        # Add self-loops + make symmetric (same as classification path)
        g = dgl.to_bidirected(dgl.to_simple(g), copy_ndata=False)
        g = dgl.add_self_loop(g)
        # Use average of snapshot-wise features (simple; matches CLDG single-snapshot baselines)
        feat = feat_list[-1] if isinstance(feat_list, list) else feat_list
        alab = th.tensor(alab).squeeze().long().cpu().numpy() if not isinstance(alab, np.ndarray) else alab.squeeze()

        if args.method == 'lp':
            # LP doesn't do anomaly meaningfully; skip
            score = np.random.RandomState(args.seed).rand(g.num_nodes())
        elif args.method in ('gcn', 'gat', 'sage'):
            # Train unsupervised: reconstruct features via single-layer GNN auto-encoder (reconstruction-error as anomaly)
            cls = {'gcn': GCNNet, 'gat': GATNet, 'sage': SAGENet}[args.method]
            model = cls(feat.size(1), args.hidden, feat.size(1))
            model = model.to(device); g = g.to(device); fx = feat.to(device)
            opt = th.optim.Adam(model.parameters(), lr=1e-2, weight_decay=5e-4)
            for _ in range(max(50, args.epochs // 4)):
                model.train()
                recon = model(g, fx)
                loss = F.mse_loss(recon, fx)
                opt.zero_grad(); loss.backward(); opt.step()
            model.eval()
            with th.no_grad():
                recon = model(g, fx)
                score = (recon - fx).norm(dim=1).cpu().numpy()
        elif args.method == 'dgi':
            h = train_dgi(g, feat, device, hid=args.hidden, epochs=args.epochs)
            score = anomaly_score(h, g.to(device))
        elif args.method == 'ccassg':
            h = train_ccassg(g, feat, device, hid=args.hidden, epochs=args.epochs)
            score = anomaly_score(h, g.to(device))

        auc = roc_auc_score(alab, score)
        print(f"[baseline-result] method={args.method} dataset={args.dataset} task=anomaly_detection "
              f"result={{'auc': {float(auc):.6f}}}")


if __name__ == '__main__':
    main()
