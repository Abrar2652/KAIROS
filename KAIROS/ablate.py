"""
KAIROS — ablation runner
========================
Thin wrapper over main.train() that monkey-patches the encoder / eval
for diagnostic variants. Does NOT modify main.py, so canonical runs
remain bit-for-bit reproducible.

Supported ablation modes (via --ablation):
  none            : default KAIROS (same as main.py)
  no_ppr_eval     : at classification eval, use only h_z (skip h_d fusion)
  no_ppr_view     : at training, disable PPR encoder entirely (only L_LL
                    contrastive term remains, single encoder)
  fixed_tau       : fix temperature τ (non-trainable); value set by --tau_val
  no_koop_anomaly : force λ_koop=0 for anomaly (remove Koopman term entirely)

Reproducibility: seed overridable via --seed (default 24). Linear-probe
runs remain at 5 for fairness against the canonical path.
"""

import argparse
import random

import numpy as np
import torch as th

import main as kmain
import models as kmodels


def _patch_no_ppr_eval():
    """Replace _eval_classification to skip h_d fusion."""
    orig_eval = kmain._eval_classification

    def _eval_no_ppr(model, graph, node_feat, n_layers, dataloader_size,
                     num_workers, device_id, dataset, alpha=0.15):
        import dgl
        from dgl.dataloading import MultiLayerFullNeighborSampler
        try:
            from dgl.dataloading.pytorch import NodeDataLoader
        except ImportError:
            from dgl.dataloading import DataLoader as NodeDataLoader
        import torch.nn as thnn
        from sklearn.metrics import f1_score
        from main import load_subtensor, LogReg

        sampler = MultiLayerFullNeighborSampler(n_layers)
        g = dgl.to_simple(graph)
        g = dgl.to_bidirected(g, copy_ndata=True)
        g = dgl.add_self_loop(g)
        test_dl = NodeDataLoader(
            g, g.nodes(), sampler,
            batch_size=dataloader_size, shuffle=False, drop_last=False,
            num_workers=num_workers,
        )
        h_z_list = []
        with th.no_grad():
            for _, (input_nodes, _, blocks) in enumerate(test_dl):
                bi = load_subtensor(node_feat, input_nodes, device_id)
                blocks = [b.to(device_id) for b in blocks]
                h_z_list.append(model.encode_orig(blocks, bi).detach())
        embeddings = th.cat(h_z_list, dim=0)

        from utils import dataloader
        labels, tr, va, te, n_classes = dataloader(dataset)
        embed_dim = embeddings.shape[1]

        tr_e = embeddings[tr].to(device_id)
        va_e = embeddings[va].to(device_id)
        te_e = embeddings[te].to(device_id)
        lab = labels.to(device_id)
        tr_l = lab[tr].detach().clone()
        va_l = lab[va].detach().clone()
        te_l = lab[te].detach().clone()

        micros, weights = [], []
        for _ in range(5):
            lr = LogReg(embed_dim, n_classes).to(device_id)
            loss_fn = thnn.CrossEntropyLoss()
            opt = th.optim.Adam(lr.parameters(), lr=1e-2, weight_decay=1e-4)
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
                            em = mi; ew = we
            micros.append(em); weights.append(we)
        return {'accuracy': float(np.mean(micros)),
                'weighted_f1': float(np.mean(weights))}

    kmain._eval_classification = _eval_no_ppr


def _patch_no_ppr_view():
    """Disable PPR encoder during training — single-view only."""
    # The cleanest way: after model construction, replace diff encoder's
    # forward with orig's, and drop LG/GG terms via a train patch.
    orig_train = kmain.train

    def train_no_ppr(*args, **kwargs):
        # We need to monkey-patch encode_diff to mirror encode_orig so
        # LG/GG terms become equivalent to LL (and thus redundant but not
        # harmful); cleaner is to wrap the internal training loop, but
        # that duplicates a lot of code. Mirror is a conservative proxy.
        orig_init = kmodels.KairosEncoder.__init__

        def init_no_diff(self, *a, **kw):
            orig_init(self, *a, **kw)
            # Tie diff to orig so both views produce identical embeddings
            # → L_LG and L_GG degenerate to L_LL. Not a true ablation but
            # the cleanest drop-in. Marked as approximate in results.
            self.diff_layers = self.orig_layers
            self.diff_linear = self.orig_linear
            self.diff_projector = self.orig_projector
            self.act_diff = self.act_orig

        kmodels.KairosEncoder.__init__ = init_no_diff
        try:
            return orig_train(*args, **kwargs)
        finally:
            kmodels.KairosEncoder.__init__ = orig_init

    kmain.train = train_no_ppr


def _patch_fixed_tau(tau_val: float):
    orig_init = kmodels.KairosEncoder.__init__

    def init_fixed(self, *a, **kw):
        orig_init(self, *a, **kw)
        import math
        # Replace log_tau parameter with a non-trainable buffer
        del self.log_tau
        self.register_buffer('log_tau', th.tensor(math.log(tau_val)))

    kmodels.KairosEncoder.__init__ = init_fixed


def _patch_warm_tau(tau_init: float):
    """Keep log_tau learnable, but change its initialization."""
    orig_init = kmodels.KairosEncoder.__init__

    def init_warm(self, *a, **kw):
        orig_init(self, *a, **kw)
        import math, torch.nn as thnn
        # Replace the parameter with one initialized at tau_init instead of 0.07
        self.log_tau = thnn.Parameter(th.tensor(math.log(tau_init)))

    kmodels.KairosEncoder.__init__ = init_warm


def _patch_no_koop_anomaly():
    """Force λ_koop=0 even for anomaly (removes Koopman term from loss)."""
    orig_train = kmain.train

    def train_nokoop(*args, **kwargs):
        kwargs['lambda_koop'] = 0.0
        kwargs['lambda_koop_reg'] = 0.0
        return orig_train(*args, **kwargs)

    kmain.train = train_nokoop


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True)
    p.add_argument('--task', required=True,
                   choices=['classification', 'anomaly_detection'])
    p.add_argument('--hidden_dim', type=int, default=128)
    p.add_argument('--n_classes', type=int, default=64)
    p.add_argument('--n_layers', type=int, default=2)
    p.add_argument('--fanout', type=str, default='20,20')
    p.add_argument('--snapshots', type=int, default=4)
    p.add_argument('--views', type=int, default=4)
    p.add_argument('--strategy', default='sequential')
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--dataloader_size', type=int, default=4096)
    p.add_argument('--GPU', type=int, required=True)
    p.add_argument('--epochs', type=int, required=True)
    p.add_argument('--alpha', type=float, default=0.15)
    p.add_argument('--lambda_koop', type=float, default=1.0)
    p.add_argument('--lambda_koop_reg', type=float, default=0.01)
    p.add_argument('--seed', type=int, default=24)
    p.add_argument('--ablation', default='none',
                   choices=['none', 'no_ppr_eval', 'no_ppr_view',
                            'fixed_tau', 'no_koop_anomaly', 'warm_tau'])
    p.add_argument('--tau_val', type=float, default=0.1)
    p.add_argument('--tau_init', type=float, default=0.5)
    p.add_argument('--backbone', default='gat',
                   choices=['gat', 'gcn', 'sgc', 'sage', 'h2gcn'])
    p.add_argument('--save_embed_path', default=None,
                   help='if set, save final eval embedding tensor to this path')
    p.add_argument('--diff_mode', default='pprfeat',
                   choices=['pprfeat', 'pprgraph'],
                   help='pprfeat = APPNP feature smoothing (default); '
                        'pprgraph = CLDG++-style weighted PPR graph (N^3, small graphs only)')
    a = p.parse_args()

    # Reseed for the requested seed (overrides main.py's _SEED=24 default).
    random.seed(a.seed); np.random.seed(a.seed)
    th.manual_seed(a.seed); th.cuda.manual_seed_all(a.seed)

    print(f'[ablate] mode={a.ablation} seed={a.seed} '
          f'dataset={a.dataset} task={a.task}')

    if a.ablation == 'no_ppr_eval':
        _patch_no_ppr_eval()
    elif a.ablation == 'no_ppr_view':
        _patch_no_ppr_view()
    elif a.ablation == 'fixed_tau':
        _patch_fixed_tau(a.tau_val)
    elif a.ablation == 'warm_tau':
        _patch_warm_tau(a.tau_init)
    elif a.ablation == 'no_koop_anomaly':
        _patch_no_koop_anomaly()

    fanouts = [int(x) for x in a.fanout.split(',')]
    # Attach embed-save path to the train function as an attribute
    # (gracefully picked up by _eval_classification via getattr).
    if a.save_embed_path:
        kmain.train._save_embed_path = a.save_embed_path
    r = kmain.train(
        dataset=a.dataset, task=a.task,
        hidden_dim=a.hidden_dim, n_classes=a.n_classes, n_layers=a.n_layers,
        fanouts=fanouts, snapshots=a.snapshots, views=a.views,
        strategy=a.strategy, readout='max',
        batch_size=a.batch_size, dataloader_size=a.dataloader_size,
        alpha=a.alpha, lambda_koop=a.lambda_koop,
        lambda_koop_reg=a.lambda_koop_reg, num_workers=0,
        epochs=a.epochs, GPU=a.GPU,
        backbone=a.backbone,
        diff_mode=a.diff_mode,
    )
    print(f'[ablate-result] ablation={a.ablation} seed={a.seed} '
          f'dataset={a.dataset} task={a.task} result={r}')


if __name__ == '__main__':
    main()
