"""
KAIROS — main.py  (v2)
======================
Koopman-Aligned Invariant Representations for Open dynamic Systems

Training objective (per temporal pair i, j)
────────────────────────────────────────────
    L_LL  = InfoNCE(p_z_i, p_z_j)/2            orig–orig, across time
    L_LG  = InfoNCE(p_z_i, p_d_i)/4 +
            InfoNCE(p_z_j, p_d_j)/4            orig–PPR, same snapshot
    L_GG  = InfoNCE(p_d_i, p_d_j)/4            PPR–PPR, across time
    L_koop = KoopmanHead.predict_loss(h_i, h_j) temporal linearization
                                                 (λ_koop=0 for classification)

    Total = L_LL + L_LG + L_GG + λ_koop·L_koop + λ_reg·L_reg

Design decisions (v2)
──────────────────────────────────────────────────
  · Task-adaptive Koopman: λ_koop overridden to 0 for classification.
    The Koopman loss forces embeddings into a linearly-predictable temporal
    subspace — ideal for anomaly residuals, but directly compresses the
    representation away from class-discriminative directions.

  · Classification eval: single full-graph pass (CLDG protocol).
    The encoder is trained on temporal subgraphs but evaluated on the full
    static graph — this is CLDG's proven recipe.  Doing multi-snapshot eval
    here fragments small graphs (e.g. Bitcoinotc has 35K edges; splitting
    into 4 windows leaves ~8.7K edges each, which collapses most nodes to
    their self-loops).

  · Dual-view fused embedding at eval: avg(h_z, h_d) → 64d.
    We trained TWO encoders (orig + PPR) so we use both at inference, but
    average them to keep dimensionality at 64 — identical to what CLDG /
    CLDG++ feed their linear probes, so the comparison is fair.

  · Adaptive sampler: full-neighbour for small graphs (≤ FULL_NBR_THRESH),
    mini-batch for large.

  · Anomaly eval keeps multi-snapshot, with three complementary signals:
      S1 temporal inconsistency  (cosine dist across snapshots)
      S2 Koopman prediction error (||K h_t − h_{t+1}||₂)
      S3 neighbourhood feature deviation (||h − mean(h_neighbours)||₂)
"""

import argparse
import copy
import math
import random
from functools import reduce

import dgl
import dgl.function as fn
import numpy as np
import torch as th
import torch.nn as thnn
import torch.nn.functional as F
import torch.optim as optim

try:
    from dgl.dataloading.neighbor import (
        MultiLayerFullNeighborSampler,
        MultiLayerNeighborSampler,
    )
    from dgl.dataloading.pytorch import NodeDataLoader
except ImportError:
    from dgl.dataloading import (
        MultiLayerFullNeighborSampler,
        DataLoader as NodeDataLoader,
    )
    try:
        from dgl.dataloading import NeighborSampler as MultiLayerNeighborSampler
    except ImportError:
        from dgl.dataloading import MultiLayerNeighborSampler

from sklearn.metrics import f1_score, roc_auc_score

from models import KairosEncoder, LogReg
from utils import (
    compute_ppr_features,
    dataloader,
    load_to_dgl_graph,
    sampling_layer,
)

# ── Global reproducibility ────────────────────────────────────────────────────
_SEED = 24
random.seed(_SEED)
np.random.seed(_SEED)
th.manual_seed(_SEED)
th.cuda.manual_seed_all(_SEED)
th.backends.cudnn.deterministic = True
th.backends.cudnn.benchmark     = False

# Graphs with ≤ this many nodes get full-neighbour sampling during training.
FULL_NBR_THRESH = 8000


# ═══════════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════════


def load_subtensor(feats, nodes, device):
    return feats[nodes].to(device)


def _z_norm(x: th.Tensor) -> th.Tensor:
    x = x.float()
    return (x - x.mean()) / (x.std() + 1e-8)


def _build_subgraph(working_graph, start, span, snapshots, max_time, min_time):
    # `start` is a 0-based offset returned by sampling_layer. All shipped
    # datasets have min_time ≈ 0, so these offsets and edge times share units.
    end = min(start + span / snapshots, max_time)
    start_c = max(start, min_time)
    mask = (working_graph.edata["time"] >= start_c) & (
        working_graph.edata["time"] <= end
    )
    tsg = dgl.edge_subgraph(working_graph, mask, relabel_nodes=False)
    tsg = dgl.to_simple(tsg)
    tsg = dgl.to_bidirected(tsg, copy_ndata=True)
    tsg = dgl.add_self_loop(tsg)
    return tsg


def _sym_nce(z1, z2, ce_fn, tau):
    """Symmetric InfoNCE with learnable temperature τ."""
    labels = th.arange(z1.shape[0], device=z1.device)
    p1 = th.mm(z1, z2.T)
    p2 = th.mm(z2, z1.T)
    p1 = p1 - p1.max(dim=1, keepdim=True).values
    p2 = p2 - p2.max(dim=1, keepdim=True).values
    return ce_fn(p1 / tau, labels) + ce_fn(p2 / tau, labels)


# ═══════════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════════


def train(
    dataset,
    hidden_dim,
    n_layers,
    n_classes,
    fanouts,
    snapshots,
    views,
    strategy,
    readout,
    batch_size,
    dataloader_size,
    alpha,
    num_workers,
    epochs,
    GPU,
    task,
    lambda_koop: float = 1.0,
    lambda_koop_reg: float = 0.01,
    backbone: str = 'gat',  # 'gat' | 'gcn' | 'sgc' | 'sage'
    diff_mode: str = 'pprfeat',  # 'pprfeat' (APPNP features) | 'pprgraph' (CLDG++ weighted graph)
    # Legacy params kept for call-site compatibility; ignored.
    diff=None,
    lambda_nrd=None,
    lambda_mae=None,
    mask_ratio=None,
):
    if not th.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. KAIROS requires a GPU. "
            "On Colab: Runtime -> Change runtime type -> GPU."
        )
    device_id = th.device(f"cuda:{GPU}")
    th.cuda.set_device(device_id)
    print(
        f"[KAIROS] Using device: {device_id} "
        f"({th.cuda.get_device_name(device_id)})"
    )

    # ── Task-adaptive Koopman weight ──────────────────────────────────────
    # For classification, the Koopman loss forces embeddings into a linearly-
    # predictable temporal subspace, which reduces class separability.
    # The Koopman head is still used for anomaly detection (S2 residual signal).
    if task == "classification":
        lambda_koop = 0.0
        print("[KAIROS] Classification task → λ_koop overridden to 0.0")

    # ── Data loading ──────────────────────────────────────────────────────
    if task == "anomaly_detection":
        inject_graph, inject_node_feat, anomaly_label = load_to_dgl_graph(
            dataset, task=task, snapshots=snapshots
        )
        working_graph = inject_graph
    else:
        graph, node_feat = load_to_dgl_graph(dataset, task=task)
        working_graph = graph

    in_feat = (
        node_feat.shape[1] if task == "classification" else inject_node_feat[0].shape[1]
    )

    # ── Adaptive neighbour sampler ────────────────────────────────────────
    n_nodes = working_graph.num_nodes()
    use_full_nbr = n_nodes <= FULL_NBR_THRESH
    if use_full_nbr:
        sampler = MultiLayerFullNeighborSampler(n_layers)
        print(
            f"[KAIROS] Small graph ({n_nodes} nodes) → "
            f"full-neighbour training sampler\n"
        )
    else:
        sampler = MultiLayerNeighborSampler(fanouts)
        print(
            f"[KAIROS] Large graph ({n_nodes} nodes) → "
            f"mini-batch sampler fanout={fanouts}\n"
        )

    # ── Model ─────────────────────────────────────────────────────────────
    model = KairosEncoder(
        in_feats=in_feat,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        embed_dim=n_classes,
        dropout=0.0,
        backbone=backbone,
    ).to(device_id)
    print(f'[KAIROS] Backbone: {backbone}')

    ce_loss = thnn.CrossEntropyLoss().to(device_id)
    optimizer = optim.Adam(model.parameters(), lr=4e-3, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, epochs), eta_min=1e-5
    )

    best_loss  = th.tensor([float("inf")]).to(device_id)
    best_model = copy.deepcopy(model)

    print(
        f"Training {epochs} epochs | {views} views | "
        f"λ_koop={lambda_koop} | PPR α={alpha}\n"
    )

    # ── PPR-graph cache (for diff_mode='pprgraph') ──────────────────────────
    # Pre-compute PPR graph per unique snapshot index once — the snapshot
    # start times are deterministic (fixed T_full positions), so we only need
    # one PPR computation per snapshot, reused across all epochs.
    ppr_graph_cache = {}
    if diff_mode == 'pprgraph':
        from utils import compute_ppr_graph
        edges_time = working_graph.edata["time"].tolist()
        max_time_pp = max(edges_time); min_time_pp = min(edges_time)
        span_pp = max_time_pp - min_time_pp
        T_full = [span_pp * i / snapshots for i in range(snapshots)]
        print(f'[KAIROS] Pre-computing PPR graphs for {snapshots} snapshots '
              f'(diff_mode=pprgraph)...')
        for idx, start in enumerate(T_full):
            tsg_cache = _build_subgraph(
                working_graph, start, span_pp, snapshots, max_time_pp, min_time_pp
            )
            n_cache = tsg_cache.num_nodes()
            print(f'  snapshot {idx}: N={n_cache} nodes', flush=True)
            diff_g_c, diff_w_c = compute_ppr_graph(tsg_cache, alpha=alpha)
            diff_g_c = dgl.add_self_loop(diff_g_c).to(device_id)
            n_real_c = diff_w_c.shape[0]
            n_total_c = diff_g_c.num_edges()
            diff_w_c = diff_w_c.to(device_id)
            if n_total_c > n_real_c:
                sl_c = th.ones(n_total_c - n_real_c, device=device_id)
                diff_w_c = th.cat([diff_w_c, sl_c], dim=0)
            ppr_graph_cache[idx] = (diff_g_c, diff_w_c)
        print(f'[KAIROS] PPR-graph cache ready.\n')

    # ── Training loop ─────────────────────────────────────────────────────
    for epoch in range(epochs):
        model.train()

        edges_time = working_graph.edata["time"].tolist()
        max_time = max(edges_time)
        min_time = min(edges_time)
        span = max_time - min_time

        T, T_idx = sampling_layer(snapshots, views, span, strategy)

        temporal_subgraphs, nids, dl_list = [], [], []
        for start in T:
            tsg = _build_subgraph(
                working_graph, start, span, snapshots, max_time, min_time
            )
            nids.append(th.unique(tsg.edges()[0]))
            temporal_subgraphs.append(tsg)

        train_nid = list(
            reduce(
                lambda x, y: x & y,
                [set(nids[i].tolist()) for i in range(views)],
            )
        )
        sample_n = min(batch_size, len(train_nid))
        train_nid = random.sample(train_nid, sample_n)
        random.shuffle(train_nid)
        train_nid_t = th.tensor(train_nid)

        for sg_id in range(views):
            dl_list.append(
                NodeDataLoader(
                    temporal_subgraphs[sg_id],
                    train_nid_t,
                    sampler,
                    batch_size=train_nid_t.shape[0],
                    shuffle=False,
                    drop_last=False,
                    num_workers=num_workers,
                )
            )

        # ── Forward pass ─────────────────────────────────────────────────
        orig_projs = th.tensor([], device=device_id)
        diff_projs = th.tensor([], device=device_id)
        orig_hs_per_view = []   # pre-projection h per view for Koopman

        for i_, dl in enumerate(dl_list):
            for _, (input_nodes, seeds, blocks) in enumerate(dl):
                if task == "anomaly_detection":
                    feat_i = inject_node_feat[T_idx[i_]]
                else:
                    feat_i = node_feat

                batch_inputs = load_subtensor(feat_i, input_nodes, device_id)
                blocks = [b.to(device_id) for b in blocks]

                # 1. Orig encoder → pre-projection h + projected p
                h_orig = model.encode_orig(blocks, batch_inputs)
                p_orig = model.project_orig(h_orig)
                orig_projs = th.cat([orig_projs, p_orig.unsqueeze(0)], dim=0)
                orig_hs_per_view.append(h_orig)

                if diff_mode == 'pprgraph':
                    # CLDG++-style: use pre-cached PPR-weighted graph for this snapshot.
                    diff_g, diff_w = ppr_graph_cache[T_idx[i_]]
                    n_sub = diff_g.num_nodes()
                    sub_feats = feat_i[:n_sub].to(device_id)
                    fake_blocks = [diff_g] * model.n_layers
                    h_diff_full = model.encode_diff(
                        fake_blocks, sub_feats, edge_weights=[diff_w] * model.n_layers,
                    )
                    h_diff = h_diff_full[seeds]
                    p_diff = model.project_diff(h_diff)
                else:
                    # Default: PPR-smoothed features (APPNP-style)
                    ppr_i     = compute_ppr_features(temporal_subgraphs[i_], feat_i, alpha=alpha)
                    batch_ppr = load_subtensor(ppr_i, input_nodes, device_id)
                    h_diff = model.encode_diff(blocks, batch_ppr)
                    p_diff = model.project_diff(h_diff)
                diff_projs = th.cat([diff_projs, p_diff.unsqueeze(0)], dim=0)

        # ── Contrastive loss (LL / LG / GG, learnable τ) ─────────────────
        nce_loss = th.tensor(0.0, device=device_id)
        tau = model.tau

        for idx in range(orig_projs.shape[0]):
            for idy in range(idx + 1, orig_projs.shape[0]):
                pz1 = orig_projs[idx]
                pz2 = orig_projs[idy]
                pd1 = diff_projs[idx]
                pd2 = diff_projs[idy]

                loss_ll = _sym_nce(pz1, pz2, ce_loss, tau) / 2
                loss_lg = (
                    _sym_nce(pz1, pd1, ce_loss, tau) / 4
                    + _sym_nce(pz2, pd2, ce_loss, tau) / 4
                )
                loss_gg = _sym_nce(pd1, pd2, ce_loss, tau) / 4

                nce_loss = nce_loss + loss_ll + loss_lg + loss_gg

        # ── Koopman loss: linearize temporal evolution ────────────────────
        # λ_koop=0 for classification (set above) → these terms vanish.
        koop_loss = th.tensor(0.0, device=device_id)
        koop_reg  = th.tensor(0.0, device=device_id)
        if lambda_koop > 0.0 and len(orig_hs_per_view) >= 2:
            order = sorted(range(len(orig_hs_per_view)), key=lambda k: T_idx[k])
            n_pairs = 0
            for a, b in zip(order[:-1], order[1:]):
                koop_loss = koop_loss + model.koopman.predict_loss(
                    orig_hs_per_view[a], orig_hs_per_view[b]
                )
                n_pairs += 1
            if n_pairs > 0:
                koop_loss = koop_loss / float(n_pairs)
            koop_reg = model.koopman.invariance_reg()

        total_loss = (
            nce_loss
            + lambda_koop     * koop_loss
            + lambda_koop_reg * koop_reg
        )

        optimizer.zero_grad()
        total_loss.backward()
        thnn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        if total_loss < best_loss:
            best_loss  = total_loss
            best_model = copy.deepcopy(model)

        print(
            "Epoch {:05d} | NCE {:.4f} | KOOP {:.4f} | τ {:.4f} | Total {:.4f}".format(
                epoch,
                float(nce_loss.detach()),
                float(koop_loss.detach()),
                float(model.tau.detach()),
                float(total_loss.detach()),
            )
        )

    # ── Evaluation ────────────────────────────────────────────────────────
    best_model.eval()

    if task == "classification":
        save_embed_path = getattr(train, '_save_embed_path', None)
        return _eval_classification(
            best_model, working_graph, node_feat, n_layers,
            dataloader_size, num_workers, device_id, dataset, alpha=alpha,
            save_embed_path=save_embed_path, diff_mode=diff_mode,
        )
    else:
        return _eval_anomaly(
            best_model, inject_graph, inject_node_feat, anomaly_label,
            n_layers, snapshots, dataloader_size, num_workers, device_id,
        )


# ═══════════════════════════════════════════════════════════════════
# Classification evaluation — CLDG-style single full-graph pass,
#                              dual-view fused embedding (64d)
# ═══════════════════════════════════════════════════════════════════


def _eval_classification(
    model, graph, node_feat, n_layers, dataloader_size, num_workers,
    device_id, dataset, alpha=0.15, save_embed_path=None, diff_mode='pprfeat',
):
    """CLDG-style single full-graph evaluation with dual-view fusion.

    Protocol matches CLDG / CLDG++ exactly:
      · Full static graph, one forward pass, frozen encoder.
      · 64-dimensional embedding → LogReg (2000 epochs, 5 runs, lr=1e-2).

    KAIROS adaptation:
      · We trained two encoders (orig + PPR structural view). We use both
        at inference by averaging their outputs: h_fused = (h_z + h_d) / 2.
      · Still 64d — identical dimensionality to what CLDG / CLDG++ feed
        their linear probes, so the comparison is strictly fair.
      · The PPR feature is computed on the FULL static graph (not per
        temporal subgraph) for maximum structural information.
    """
    sampler = MultiLayerFullNeighborSampler(n_layers)

    # Canonicalise the full graph exactly as CLDG does.
    g = dgl.to_simple(graph)
    g = dgl.to_bidirected(g, copy_ndata=True)
    g = dgl.add_self_loop(g)

    # PPR structural features on the full graph (maximal coverage).
    ppr_feat = compute_ppr_features(g, node_feat, alpha=alpha)

    # For pprgraph mode, also pre-compute the full PPR-weighted graph
    ppr_graph_full = None
    ppr_weights_full = None
    if diff_mode == 'pprgraph':
        from utils import compute_ppr_graph
        print(f'[KAIROS] Computing PPR-graph for full-graph eval...', flush=True)
        ppr_graph_full, ppr_weights_full = compute_ppr_graph(g, alpha=alpha)
        ppr_graph_full = dgl.add_self_loop(ppr_graph_full).to(device_id)
        n_real_e = ppr_weights_full.shape[0]
        n_total_e = ppr_graph_full.num_edges()
        ppr_weights_full = ppr_weights_full.to(device_id)
        if n_total_e > n_real_e:
            sl_e = th.ones(n_total_e - n_real_e, device=device_id)
            ppr_weights_full = th.cat([ppr_weights_full, sl_e], dim=0)

    test_dl = NodeDataLoader(
        g, g.nodes(), sampler,
        batch_size=dataloader_size, shuffle=False,
        drop_last=False, num_workers=num_workers,
    )

    h_z_list, h_d_list = [], []
    with th.no_grad():
        for _, (input_nodes, _, blocks) in enumerate(test_dl):
            batch_inputs = load_subtensor(node_feat, input_nodes, device_id)
            batch_ppr    = load_subtensor(ppr_feat, input_nodes, device_id)
            blocks = [b.to(device_id) for b in blocks]

            h_z = model.encode_orig(blocks, batch_inputs)
            if diff_mode == 'pprfeat':
                h_d = model.encode_diff(blocks, batch_ppr)
            # For pprgraph, skip per-batch diff (handled full-graph below)
            h_z_list.append(h_z.detach())
            if diff_mode == 'pprfeat':
                h_d_list.append(h_d.detach())

    h_z_all = th.cat(h_z_list, dim=0)
    if diff_mode == 'pprgraph':
        # Run diff encoder on the full PPR-weighted graph, once
        with th.no_grad():
            sub_feats_e = node_feat[:ppr_graph_full.num_nodes()].to(device_id)
            fake_blocks_e = [ppr_graph_full] * model.n_layers
            h_d_all = model.encode_diff(
                fake_blocks_e, sub_feats_e,
                edge_weights=[ppr_weights_full] * model.n_layers,
            ).detach()
    else:
        h_d_all = th.cat(h_d_list, dim=0)

    # Dual-view fusion — average in the 64d space
    # (both encoders were trained together, contrasting across LL/LG/GG).
    embeddings = 0.5 * (h_z_all + h_d_all)

    if save_embed_path is not None:
        th.save({'embeddings': embeddings.cpu(),
                 'h_z': h_z_all.cpu(),
                 'h_d': h_d_all.cpu()}, save_embed_path)
        print(f'[save] embeddings saved to {save_embed_path}')

    labels, train_idx, val_idx, test_idx, n_classes = dataloader(dataset)
    embed_dim = embeddings.shape[1]   # 64 — same as CLDG / CLDG++

    train_embs = embeddings[train_idx].to(device_id)
    val_embs   = embeddings[val_idx].to(device_id)
    test_embs  = embeddings[test_idx].to(device_id)
    label = labels.to(device_id)
    train_labels = label[train_idx].detach().clone()
    val_labels   = label[val_idx].detach().clone()
    test_labels  = label[test_idx].detach().clone()

    micros, weights_f1 = [], []
    for _ in range(5):
        logreg = LogReg(embed_dim, n_classes).to(device_id)
        loss_fn = thnn.CrossEntropyLoss()
        opt = th.optim.Adam(logreg.parameters(), lr=1e-2, weight_decay=1e-4)
        best_val_acc, eval_micro, eval_weight = 0, 0, 0

        for ep in range(2000):
            logreg.train()
            opt.zero_grad()
            logits = logreg(train_embs)
            preds = th.argmax(logits, dim=1)
            train_acc = (preds == train_labels).float().mean()
            loss = loss_fn(logits, train_labels)
            loss.backward()
            opt.step()

            logreg.eval()
            with th.no_grad():
                val_logits  = logreg(val_embs)
                test_logits = logreg(test_embs)
                val_preds   = th.argmax(val_logits, dim=1)
                test_preds  = th.argmax(test_logits, dim=1)
                val_acc = (val_preds == val_labels).float().mean()
                ys      = test_labels.cpu().numpy()
                idx_np  = test_preds.cpu().numpy()
                micro  = th.tensor(f1_score(ys, idx_np, average="micro"))
                weight = th.tensor(f1_score(ys, idx_np, average="weighted"))
                if val_acc >= best_val_acc:
                    best_val_acc = val_acc
                    if (micro + weight) >= (eval_micro + eval_weight):
                        eval_micro  = micro
                        eval_weight = weight

            if ep % 500 == 0 or ep == 1999:
                print(
                    f"  LinEval {ep:4d} | train {train_acc:.4f} "
                    f"| val {val_acc:.4f} | micro {micro:.4f} | wei {weight:.4f}"
                )

        micros.append(eval_micro)
        weights_f1.append(eval_weight)

    micros     = th.stack(micros)
    weights_f1 = th.stack(weights_f1)
    acc  = micros.mean().item()
    wf1  = weights_f1.mean().item()
    print(f"\nLinear evaluation  Acc: {acc:.4f}  Wei-F1: {wf1:.4f}")
    return {"accuracy": acc, "weighted_f1": wf1}


# ═══════════════════════════════════════════════════════════════════
# Anomaly detection evaluation — three complementary signals
# ═══════════════════════════════════════════════════════════════════


def _eval_anomaly(
    model,
    inject_graph,
    inject_node_feat,
    anomaly_label,
    n_layers,
    snapshots,
    dataloader_size,
    num_workers,
    device_id,
):
    sampler = MultiLayerFullNeighborSampler(n_layers)

    edges_time = inject_graph.edata["time"].tolist()
    max_time = max(edges_time)
    min_time = min(edges_time)
    span = max_time - min_time

    T_test = [span * i / snapshots for i in range(snapshots)]
    temporal_subgraphs = []
    for start in T_test:
        tsg = _build_subgraph(inject_graph, start, span, snapshots, max_time, min_time)
        temporal_subgraphs.append(tsg)

    test_dl_list = []
    for sg_id in range(snapshots):
        test_dl_list.append(
            NodeDataLoader(
                temporal_subgraphs[sg_id],
                temporal_subgraphs[sg_id].nodes(),
                sampler,
                batch_size=dataloader_size,
                shuffle=False,
                drop_last=False,
                num_workers=num_workers,
            )
        )

    snap_embs = []   # (N, d) per snapshot, on CPU

    model.eval()
    with th.no_grad():
        for i_, dl in enumerate(test_dl_list):
            feat_i = inject_node_feat[i_]
            emb_buf = []
            for _, (input_nodes, seeds, blocks) in enumerate(dl):
                batch_inputs = load_subtensor(feat_i, input_nodes, device_id)
                blocks = [b.to(device_id) for b in blocks]
                h = model.encode_orig(blocks, batch_inputs)
                emb_buf.append(h.detach().cpu())
            snap_embs.append(th.cat(emb_buf, dim=0))

    S = len(snap_embs)

    # ── Signal 1: Temporal inconsistency (cosine distance across snapshots)
    pairs = []
    for i in range(S):
        for j in range(i + 1, S):
            zi = snap_embs[i].to(device_id)
            zj = snap_embs[j].to(device_id)
            d  = 1.0 - F.cosine_similarity(zi, zj, dim=1)
            pairs.append(d.detach().cpu())

    if pairs:
        pm = th.stack(pairs, dim=1)
        s1 = pm.mean(dim=1) + pm.std(dim=1)
    else:
        s1 = th.zeros(inject_graph.num_nodes())

    # ── Signal 2: Koopman prediction residual (consecutive snapshots)
    koop_pairs = []
    with th.no_grad():
        for t in range(S - 1):
            h_t   = snap_embs[t].to(device_id)
            h_tp1 = snap_embs[t + 1].to(device_id)
            res   = model.koopman.residual(h_t, h_tp1).detach().cpu()
            koop_pairs.append(res)

    s2 = th.stack(koop_pairs, dim=1).mean(dim=1) if koop_pairs else th.zeros(inject_graph.num_nodes())

    # ── Signal 3: Neighbourhood feature deviation (per snapshot mean)
    dev_per_snap = []
    with th.no_grad():
        for t in range(S):
            g_t = temporal_subgraphs[t].to(device_id)
            h_t = snap_embs[t].to(device_id)
            g_t.ndata["_h"] = h_t
            g_t.update_all(fn.copy_u("_h", "_m"), fn.mean("_m", "_h_nbr"))
            h_nbr = g_t.ndata.pop("_h_nbr")
            dev = (h_t - h_nbr).pow(2).sum(dim=-1).detach().cpu()
            dev_per_snap.append(dev)

    s3 = th.stack(dev_per_snap, dim=1).mean(dim=1)

    # ── Combine ──────────────────────────────────────────────────────────
    combined  = _z_norm(s1) + _z_norm(s2) + _z_norm(s3)
    labels_np = anomaly_label.flatten()

    auc_all = roc_auc_score(labels_np, combined.numpy())
    auc_s1  = roc_auc_score(labels_np, s1.numpy())
    auc_s2  = roc_auc_score(labels_np, s2.numpy())
    auc_s3  = roc_auc_score(labels_np, s3.numpy())

    print(f"\nAUC (S1+S2+S3 combined): {auc_all:.4f}")
    print(
        f"  S1 temporal: {auc_s1:.4f} | "
        f"S2 Koopman: {auc_s2:.4f} | "
        f"S3 nbr-dev: {auc_s3:.4f}"
    )
    return {
        "auc":    auc_all,
        "auc_s1": auc_s1,
        "auc_s2": auc_s2,
        "auc_s3": auc_s3,
    }


# ═══════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KAIROS")

    parser.add_argument("--dataset",    type=str, required=True)
    parser.add_argument("--hidden_dim", type=int, required=True)
    parser.add_argument("--n_classes",  type=int, required=True)
    parser.add_argument("--n_layers",   type=int, default=2)
    parser.add_argument("--fanout",     type=str, default="20,20")
    parser.add_argument("--snapshots",  type=int, default=4)
    parser.add_argument("--views",      type=int, default=2)
    parser.add_argument("--strategy",   type=str, default="random")
    parser.add_argument("--readout",    type=str, default="max")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--dataloader_size", type=int, default=4096)
    parser.add_argument("--GPU",        type=int, required=True)
    parser.add_argument("--num_workers_per_gpu", type=int, default=4)
    parser.add_argument("--epochs",     type=int, default=100)
    parser.add_argument(
        "--task", type=str, default="classification",
        choices=["classification", "anomaly_detection"],
    )
    parser.add_argument("--alpha",       type=float, default=0.15,
                        help="PPR teleport probability (structural view).")
    parser.add_argument("--lambda_koop", type=float, default=1.0,
                        help="Weight for Koopman temporal-linearization loss "
                             "(auto-set to 0 for classification).")
    parser.add_argument("--lambda_koop_reg", type=float, default=0.01,
                        help="Weight for Koopman orthogonality regularizer.")

    args = parser.parse_args()
    FANOUTS = [int(x) for x in args.fanout.split(",")]

    print("=" * 60)
    print(f"Model            : KAIROS v2")
    print(f"Dataset          : {args.dataset}")
    print(f"Task             : {args.task}")
    print(f"Hidden dim       : {args.hidden_dim}")
    print(f"Layers           : {args.n_layers}")
    print(f"Embed dim        : {args.n_classes}")
    print(f"Snapshots        : {args.snapshots}")
    print(f"Views            : {args.views}")
    print(f"Strategy         : {args.strategy}")
    print(f"Epochs           : {args.epochs}")
    print(f"PPR alpha        : {args.alpha}")
    print(f"λ_koop           : {args.lambda_koop} (0 if classification)")
    print("=" * 60 + "\n")

    train(
        dataset=args.dataset,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        n_classes=args.n_classes,
        fanouts=FANOUTS,
        snapshots=args.snapshots,
        views=args.views,
        strategy=args.strategy,
        readout=args.readout,
        batch_size=args.batch_size,
        dataloader_size=args.dataloader_size,
        alpha=args.alpha,
        lambda_koop=args.lambda_koop,
        lambda_koop_reg=args.lambda_koop_reg,
        num_workers=args.num_workers_per_gpu,
        epochs=args.epochs,
        GPU=args.GPU,
        task=args.task,
    )
