"""
KAIROS — utils.py
=================
Data utilities: graph loading, PPR structural features,
anomaly injection, and temporal sampling helpers.
Identical data-loading protocol to CLDG / CLDG++ for fair comparison.
"""

import math
import os
import random

import dgl
import dgl.function as fn
import numpy as np
import pandas as pd
import torch as th
from scipy.spatial.distance import euclidean

random.seed(24)


# ═══════════════════════════════════════════════════════════════════
# Graph loading
# ═══════════════════════════════════════════════════════════════════


def load_to_dgl_graph(dataset: str, task: str = "classification", snapshots: int = 4):
    edges = pd.read_csv(
        os.path.join("../Data/", dataset, f"{dataset}.txt"),
        sep=" ",
        names=["start_idx", "end_idx", "time"],
    )
    src = edges.start_idx.to_numpy()
    dst = edges.end_idx.to_numpy()

    graph = dgl.graph((src, dst))
    graph.edata["time"] = th.tensor(edges.time.tolist(), dtype=th.float32)

    node_feat = position_encoding(graph.num_nodes(), emb_size=128)

    if task == "anomaly_detection":
        m, n, k = 15, 20, 50
        if dataset in ("bitcoinotc", "bitotc", "bitalpha"):
            n = 10
        elif dataset in ("dblp", "tax"):
            n = 20
        elif dataset in ("tax51", "reddit"):
            n = 200
        elif dataset == "arxiv":
            n = 100
        elif dataset == "mooc":
            n = 20
        elif dataset == "elliptic":
            n = 100
        graph, feat_list, label = inject_anomaly(graph, node_feat, m, n, k, snapshots)
        return graph, feat_list, label

    return graph, node_feat


def inject_anomaly(g, feat, m: int, n: int, k: int, s: int):
    """Identical injection protocol to CLDG++."""
    num_node = g.num_nodes()
    all_idx = list(range(num_node))
    random.shuffle(all_idx)
    anomaly_idx = all_idx[: m * n * 2]
    structure_anomaly_idx = anomaly_idx[: m * n]
    attribute_anomaly_idx = anomaly_idx[m * n :]

    label = np.zeros((num_node, 1), dtype=np.uint8)
    label[anomaly_idx, 0] = 1

    print("Constructing structural anomaly nodes...")
    u_list, v_list, t_list = [], [], []
    max_time = float(g.edata["time"].max())
    min_time = float(g.edata["time"].min())
    for n_ in range(n):
        cur = structure_anomaly_idx[n_ * m : (n_ + 1) * m]
        t = random.uniform(min_time, max_time)
        for i in cur:
            for j in cur:
                u_list.append(i)
                v_list.append(j)
                t_list.append(t)

    ori_ne = g.num_edges()
    g = dgl.add_edges(
        g,
        th.tensor(u_list),
        th.tensor(v_list),
        {"time": th.tensor(t_list, dtype=th.float32)},
    )
    print(f"Done. {len(structure_anomaly_idx)} structural nodes ({g.num_edges()-ori_ne:.0f} edges)\n")

    print("Constructing attribute anomaly nodes...")
    feat_list = []
    ori_feat = feat.clone()
    attr_splits = split_list(attribute_anomaly_idx, s)
    for lst in attr_splits:
        f = ori_feat.clone()
        for i_ in lst:
            picked = random.sample(all_idx, k)
            max_dist, max_idx = 0.0, i_
            for j_ in picked:
                d = euclidean(ori_feat[i_].numpy(), ori_feat[j_].numpy())
                if d > max_dist:
                    max_dist, max_idx = d, j_
            f[i_] = ori_feat[max_idx]
        feat_list.append(f)
    print(f"Done. {len(attribute_anomaly_idx)} attribute anomaly nodes\n")

    return g, feat_list, label


# ═══════════════════════════════════════════════════════════════════
# Label loading
# ═══════════════════════════════════════════════════════════════════


def dataloader(dataset: str):
    edges = pd.read_csv(
        os.path.join("../Data/", dataset, f"{dataset}.txt"),
        sep=" ",
        names=["start_idx", "end_idx", "time"],
    )
    label_df = pd.read_csv(
        os.path.join("../Data/", dataset, "node2label.txt"),
        sep=" ",
        names=["nodeidx", "label"],
    )
    g = dgl.graph((edges.start_idx.to_numpy(), edges.end_idx.to_numpy()))
    labels = th.full((g.number_of_nodes(),), -1).cuda()
    for nid, lab in zip(label_df.nodeidx.tolist(), label_df.label.tolist()):
        labels[nid] = lab - min(label_df.label.tolist())

    train_mask = th.full((g.number_of_nodes(),), False)
    val_mask = th.full((g.number_of_nodes(),), False)
    test_mask = th.full((g.number_of_nodes(),), False)

    random.seed(24)
    tr_idx = th.LongTensor([])
    va_idx = th.LongTensor([])
    te_idx = th.LongTensor([])
    for i in range(min(label_df.label), max(label_df.label) + 1):
        idx = label_df[label_df.label == i].nodeidx.tolist()
        random.shuffle(idx)
        n10 = int(len(idx) / 10)
        n20 = int(len(idx) / 5)
        tr_idx = th.cat((tr_idx, th.LongTensor(idx[:n10])))
        va_idx = th.cat((va_idx, th.LongTensor(idx[n10:n20])))
        te_idx = th.cat((te_idx, th.LongTensor(idx[n20:])))

    train_mask.index_fill_(0, tr_idx, True).cuda()
    val_mask.index_fill_(0, va_idx, True).cuda()
    test_mask.index_fill_(0, te_idx, True).cuda()

    return (
        labels,
        th.nonzero(train_mask, as_tuple=False).squeeze(),
        th.nonzero(val_mask, as_tuple=False).squeeze(),
        th.nonzero(test_mask, as_tuple=False).squeeze(),
        label_df.label.nunique(),
    )


# ═══════════════════════════════════════════════════════════════════
# Feature helpers
# ═══════════════════════════════════════════════════════════════════


def compute_ppr_graph(graph, alpha: float = 0.2, self_loop: bool = True,
                      threshold: float = 1e-4, device='cuda:0'):
    """CLDG++-style: compute dense PPR matrix and return a weighted graph.

    PPR = α · (I - (1-α) · D^{-1/2} A D^{-1/2})^{-1}

    Uses torch.linalg.inv on GPU for speed. For N=5881, ~1-2 seconds on GPU
    vs ~2 min on CPU with scipy.

    Returns: (diff_graph, edge_weight_tensor).
    """
    g = dgl.to_simple(graph)
    g = dgl.to_bidirected(g, copy_ndata=False)
    n = g.num_nodes()
    src, dst = g.edges()
    # Build dense adjacency on GPU
    a = th.zeros((n, n), dtype=th.float32, device=device)
    a[src.long(), dst.long()] = 1.0
    if self_loop:
        a = a + th.eye(n, dtype=th.float32, device=device)
    d_vec = a.sum(dim=1).clamp(min=1.0)
    d_inv_sqrt = d_vec.pow(-0.5)
    a_norm = d_inv_sqrt.unsqueeze(1) * a * d_inv_sqrt.unsqueeze(0)
    i_mat = th.eye(n, dtype=th.float32, device=device)
    ppr = alpha * th.linalg.inv(i_mat - (1 - alpha) * a_norm)  # N×N dense
    # Threshold and build weighted graph
    mask = ppr.abs() > threshold
    rows, cols = th.nonzero(mask, as_tuple=True)
    weights = ppr[rows, cols].detach().cpu().contiguous()
    diff_graph = dgl.graph((rows.cpu(), cols.cpu()), num_nodes=n)
    # Free GPU memory
    del a, a_norm, i_mat, ppr, mask
    th.cuda.empty_cache()
    return diff_graph, weights


def compute_ppr_features(graph, feat: th.Tensor, alpha: float = 0.15, k: int = 5) -> th.Tensor:
    """
    Approximate PPR-smoothed features via k-step power iteration (APPNP style).

        H^(0) = feat
        H^(l) = (1 - alpha) * D^{-1/2} A D^{-1/2} H^(l-1) + alpha * feat

    This is the 'S' structural view used by CLDG++, computed on-the-fly
    without offline eigenvector decomposition.  Five iterations give a
    closer approximation of the true PPR matrix than three, at the
    cost of a few extra sparse matmuls.  Empirically this helps on
    graphs with long-range structural dependencies (Reddit, TAX51).
    """
    g = dgl.to_simple(graph)
    g = dgl.to_bidirected(g, copy_ndata=False)
    g = dgl.add_self_loop(g)
    degs = g.in_degrees().float().clamp(min=1.0)
    norm = degs.pow(-0.5)               # D^{-1/2}

    h = feat.clone().float()
    feat_f = feat.float()
    for _ in range(k):
        g.ndata["_h"] = h * norm.unsqueeze(1)
        g.update_all(fn.copy_u("_h", "_m"), fn.sum("_m", "_agg"))
        h_agg = g.ndata.pop("_agg") * norm.unsqueeze(1)
        h = (1.0 - alpha) * h_agg + alpha * feat_f
    return h


def position_encoding(max_len: int, emb_size: int) -> th.Tensor:
    pe = th.zeros(max_len, emb_size)
    position = th.arange(0, max_len).unsqueeze(1)
    div_term = th.exp(
        th.arange(0, emb_size, 2) * -(math.log(10000.0) / emb_size)
    )
    pe[:, 0::2] = th.sin(position * div_term)
    pe[:, 1::2] = th.cos(position * div_term)
    return pe


def split_list(lst, s: int):
    avg = len(lst) // s
    rem = len(lst) % s
    return [
        lst[i * avg + min(i, rem) : (i + 1) * avg + min(i + 1, rem)]
        for i in range(s)
    ]


# ═══════════════════════════════════════════════════════════════════
# Neighbourhood feature helpers for NRD
# ═══════════════════════════════════════════════════════════════════


def compute_neighbour_stats(subgraph, node_feat: th.Tensor, device):
    """
    For each node v in *subgraph*, compute:
      neigh_mean : mean of all NEIGHBOUR features (excluding self-loop)
      deg        : in-degree (after bidirection + self-loop addition)

    Returns tensors on *device*.
    """
    N = node_feat.shape[0]
    feat = node_feat.to(device)

    # Remove self-loops for mean-neighbour computation
    g_no_sl = dgl.remove_self_loop(subgraph).to(device)
    in_deg = g_no_sl.in_degrees().float().to(device)  # (N,)

    g_no_sl.ndata["h"] = feat
    g_no_sl.update_all(fn.copy_u("h", "m"), fn.mean("m", "neigh_mean"))

    neigh_mean = g_no_sl.ndata["neigh_mean"]         # (N, d)  — 0 if isolated

    # Log-normalised degree: (N, 1)
    deg_norm = th.log1p(in_deg).unsqueeze(1) / math.log1p(
        float(in_deg.max()) + 1.0
    )

    return neigh_mean, deg_norm


# ═══════════════════════════════════════════════════════════════════
# Temporal view sampling  (unified, always returns T, T_idx)
# ═══════════════════════════════════════════════════════════════════


def sampling_layer(snapshots: int, views: int, span: float, strategy: str):
    T_full = [span * i / snapshots for i in range(snapshots)]

    def nearest(t):
        return min(range(snapshots), key=lambda i: abs(T_full[i] - t))

    if strategy == "random":
        T = [random.uniform(0, span * (snapshots - 1) / snapshots) for _ in range(views)]
        T_idx = [nearest(t) for t in T]

    elif strategy == "low_overlap":
        if (0.75 * views + 0.25) > snapshots:
            raise ValueError("views too large for low_overlap strategy.")
        start = random.uniform(0, span - (0.75 * views + 0.25) * span / snapshots)
        T = [start + (0.75 * i * span) / snapshots for i in range(views)]
        T_idx = [nearest(t) for t in T]

    elif strategy == "high_overlap":
        if (0.25 * views + 0.75) > snapshots:
            raise ValueError("views too large for high_overlap strategy.")
        start = random.uniform(0, span - (0.25 * views + 0.75) * span / snapshots)
        T = [start + (0.25 * i * span) / snapshots for i in range(views)]
        T_idx = [nearest(t) for t in T]

    elif strategy == "sequential":
        if views > snapshots:
            raise ValueError("views > snapshots.")
        sampled = random.sample(T_full, views)
        T = sampled
        T_idx = [T_full.index(t) for t in T]

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return T, T_idx
