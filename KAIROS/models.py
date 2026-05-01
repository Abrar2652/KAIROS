"""
KAIROS — models.py
==================
Koopman-Aligned Invariant Representations for Open dynamic Systems

Architecture:
  1. Dual-encoder backbone (disjoint weights per branch)
     · orig view  : raw positional-encoding features  (X)       → h^L
     · struct view: PPR-diffused features (APPNP, K=5, α=0.15)  → h^G
       Disjoint weights are essential: a shared encoder on (X, X_ppr)
       produces trivially correlated outputs because X_ppr is a smooth
       neighbourhood average of X, collapsing the cross-view signal.
       Backbone choices: {gat, gcn, sgc, sage, h2gcn} — selected per
       dataset on the validation set (see run_experiments.py).

  2. SimCLR v2 projection heads  (3-layer BN+GELU, dims d→2d→d→d)
     · Contrastive loss operates in ℓ₂-normalised projected space z.
     · Downstream eval uses pre-projection h (avg of orig + PPR branches).

  3. InfoNCE temperature
     · log_tau is a learnable scalar initialised at log(0.07),
       clamped to τ ∈ [0.01, 1.0].  All reported results fix τ=0.5
       (non-trainable buffer) via the fixed_tau patch in run_experiments.py
       / ablate.py.

  4. KoopmanHead — linearized temporal dynamics
     · K ∈ R^{d×d}, initialized at I (= CLDG temporal smoothness at t=0).
     · predict_loss : cosine error (squared, γ=2) between K h^L_t and h^L_{t+1}.
       Gradients flow into both K and the orig-view encoder h^L only.
     · invariance_reg : ||KK^T − I||_F² / d² keeps K near-orthogonal.
     · residual : per-node ||K h_t − h_{t+1}||₂², the S2 anomaly signal.

Training objective (per ordered view pair (u, v), u < v)
──────────────────────────────────────────────────────────
    L_LL  = NCE(z^L_u, z^L_v) / 2             orig–orig, across time
    L_LG  = NCE(z^L_u, z^G_u) / 4
           + NCE(z^L_v, z^G_v) / 4            orig–PPR,  same snapshot
    L_GG  = NCE(z^G_u, z^G_v) / 4             PPR–PPR,  across time
    L_NCE = Σ_{u<v} (L_LL + L_LG + L_GG)
    L_koop = mean cosine-error² over consecutive h^L pairs
             (λ_koop=0 for classification; λ_koop∈{0,1} for anomaly)

    Total = L_NCE + λ_koop · L_koop + λ_reg · L_reg   (λ_reg = 0.01 fixed)

Classification eval
────────────────────
    Single full-graph forward pass (CLDG protocol).
    Dual-view fused embedding: z_i = (h^L_i + h^G_i) / 2  ∈ R^{64}.
    LogReg (2000 epochs, 5 runs, lr=1e-2), best val epoch per run.

Anomaly scoring (3 signals, z-normalised and summed)
──────────────────────────────────────────────────────
    S1 = temporal inconsistency   (mean+std cosine distance across all snapshot pairs)
    S2 = Koopman prediction error (||K h^L_t − h^L_{t+1}||₂², avg consecutive pairs)
    S3 = neighbourhood deviation  (||h^L − mean_nbr(h^L)||₂², avg over snapshots)
    score = z_norm(S1) + z_norm(S2) + z_norm(S3)
"""

import math

import torch as th
import torch.nn as thnn
import torch.nn.functional as F
import dgl.nn as dglnn


# ─────────────────────────────────────────────────────────────────────────────
# Downstream evaluation head
# ─────────────────────────────────────────────────────────────────────────────

class LogReg(thnn.Module):
    def __init__(self, hid_dim, out_dim):
        super().__init__()
        self.fc = thnn.Linear(hid_dim, out_dim)

    def forward(self, x):
        return self.fc(x)


# ─────────────────────────────────────────────────────────────────────────────
# Koopman head — linearized temporal evolution in embedding space
# ─────────────────────────────────────────────────────────────────────────────

class KoopmanHead(thnn.Module):
    """
    Learn operator K such that φ(G_{t+1}) ≈ K φ(G_t).
    Initialized at K = I — recovers temporal-smoothness objective at start.
    Any deviation from identity is purely loss-driven.
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        self.K = thnn.Linear(embed_dim, embed_dim, bias=False)
        thnn.init.eye_(self.K.weight)

    def forward(self, h_t: th.Tensor) -> th.Tensor:
        return self.K(h_t)

    def predict_loss(self, h_t: th.Tensor, h_tp1: th.Tensor, gamma: float = 2.0) -> th.Tensor:
        """Scaled cosine error. Gradients flow into both h_t and h_{t+1}:
        the encoder is encouraged to produce linearly predictable trajectories."""
        h_pred = F.normalize(self.forward(h_t), p=2, dim=-1)
        h_tgt  = F.normalize(h_tp1, p=2, dim=-1)
        return (1.0 - (h_pred * h_tgt).sum(dim=-1)).pow(gamma).mean()

    def invariance_reg(self) -> th.Tensor:
        """KK^T ≈ I: keeps dynamics stable and norm-preserving."""
        W = self.K.weight
        I = th.eye(W.shape[0], device=W.device, dtype=W.dtype)
        return (W @ W.t() - I).pow(2).mean()

    def residual(self, h_t: th.Tensor, h_tp1: th.Tensor) -> th.Tensor:
        """Per-node anomaly score: ||K h_t − h_{t+1}||₂."""
        return (self.forward(h_t) - h_tp1).pow(2).sum(dim=-1)


# ─────────────────────────────────────────────────────────────────────────────
# Shared GAT backbone builder
# ─────────────────────────────────────────────────────────────────────────────

def _build_gat_layers(
    in_feats, hidden_dim, n_layers, embed_dim,
    n_heads=4, feat_drop=0.1, attn_drop=0.1,
):
    layers = thnn.ModuleList()
    if n_layers == 1:
        layers.append(dglnn.GATConv(
            in_feats, embed_dim, num_heads=1,
            feat_drop=feat_drop, attn_drop=attn_drop,
            residual=True, activation=None, allow_zero_in_degree=True,
        ))
        return layers

    layers.append(dglnn.GATConv(
        in_feats, hidden_dim // n_heads, num_heads=n_heads,
        feat_drop=feat_drop, attn_drop=attn_drop,
        residual=True, activation=F.elu, allow_zero_in_degree=True,
    ))
    for _ in range(1, n_layers - 1):
        layers.append(dglnn.GATConv(
            hidden_dim, hidden_dim // n_heads, num_heads=n_heads,
            feat_drop=feat_drop, attn_drop=attn_drop,
            residual=True, activation=F.elu, allow_zero_in_degree=True,
        ))
    layers.append(dglnn.GATConv(
        hidden_dim, embed_dim, num_heads=1,
        feat_drop=feat_drop, attn_drop=attn_drop,
        residual=True, activation=None, allow_zero_in_degree=True,
    ))
    return layers


def _build_gcn_layers(in_feats, hidden_dim, n_layers, embed_dim):
    """Standard GCN backbone — matches CLDG++'s backbone choice.
    Minimal parameters vs GAT (~3x fewer). Strong on small graphs."""
    layers = thnn.ModuleList()
    if n_layers == 1:
        layers.append(dglnn.GraphConv(
            in_feats, embed_dim, norm='both',
            activation=None, allow_zero_in_degree=True,
        ))
        return layers
    layers.append(dglnn.GraphConv(
        in_feats, hidden_dim, norm='both',
        activation=F.relu, allow_zero_in_degree=True,
    ))
    for _ in range(1, n_layers - 1):
        layers.append(dglnn.GraphConv(
            hidden_dim, hidden_dim, norm='both',
            activation=F.relu, allow_zero_in_degree=True,
        ))
    layers.append(dglnn.GraphConv(
        hidden_dim, embed_dim, norm='both',
        activation=None, allow_zero_in_degree=True,
    ))
    return layers


def _build_sgc_layers(in_feats, hidden_dim, n_layers, embed_dim):
    """SGC (Simplified Graph Convolution): K-step propagation + linear.
    Minimal capacity, strong on small noisy graphs. K = n_layers.

    Returns a 1-element ModuleList with an SGConv(in_feats → embed_dim, k=n_layers).
    The n_layers×embed_dim projection is handled by the subsequent orig_linear."""
    layers = thnn.ModuleList()
    # Note: For NodeDataLoader blocks, we use only one layer but loop n_layers
    # times in the forward; here we place n_layers GraphConvs-as-propagation
    # (SGConv in DGL expects the full graph; for message-passing blocks we use
    # plain GraphConv as propagation + a final Linear).
    # To be compatible with our block-based dataloader, we chain n_layers
    # no-activation GraphConvs. Same effect as SGC k=n_layers.
    # All-linear GraphConv stack (no activations) → behaves like SGC k=n_layers.
    if n_layers == 1:
        layers.append(dglnn.GraphConv(
            in_feats, embed_dim, norm='both',
            activation=None, allow_zero_in_degree=True,
        ))
        return layers
    layers.append(dglnn.GraphConv(
        in_feats, embed_dim, norm='both',
        activation=None, allow_zero_in_degree=True,
    ))
    for _ in range(1, n_layers):
        layers.append(dglnn.GraphConv(
            embed_dim, embed_dim, norm='both',
            activation=None, allow_zero_in_degree=True,
        ))
    return layers


def _build_sage_layers(in_feats, hidden_dim, n_layers, embed_dim, aggregator='mean'):
    """GraphSAGE backbone — inductive, stable on small graphs."""
    layers = thnn.ModuleList()
    if n_layers == 1:
        layers.append(dglnn.SAGEConv(in_feats, embed_dim, aggregator, activation=None))
        return layers
    layers.append(dglnn.SAGEConv(in_feats, hidden_dim, aggregator, activation=F.relu))
    for _ in range(1, n_layers - 1):
        layers.append(dglnn.SAGEConv(hidden_dim, hidden_dim, aggregator, activation=F.relu))
    layers.append(dglnn.SAGEConv(hidden_dim, embed_dim, aggregator, activation=None))
    return layers


class H2GCNLayer(thnn.Module):
    """Single H2GCN layer (Zhu et al., 2020, "Beyond Homophily").
    Key property: ego-neighbor separation via concat, no self-loops in
    aggregation. Output = concat(self-transformed, neighbor-aggregated)."""

    def __init__(self, in_feats, out_feats):
        super().__init__()
        # out_feats is split: half for ego, half for neighbor
        self.lin_ego = thnn.Linear(in_feats, out_feats // 2)
        self.lin_nbr = thnn.Linear(in_feats, out_feats // 2)

    def forward(self, block, feats):
        import dgl.function as fn
        # Aggregate neighbor features (mean-aggregation, no self-loop).
        # block.srcdata receives all src features; block.dstdata is subset.
        with block.local_scope():
            block.srcdata['_h'] = feats
            block.update_all(fn.copy_u('_h', '_m'), fn.mean('_m', '_nbr'))
            h_nbr = block.dstdata['_nbr']
        # Dst features — first N dst nodes in src
        n_dst = block.num_dst_nodes()
        dst_feats = feats[:n_dst]
        return th.cat([self.lin_ego(dst_feats), self.lin_nbr(h_nbr)], dim=-1)


class _H2GCNStack(thnn.Module):
    """Stack of H2GCN layers with intermediate concatenation projection.

    h_0 = input, h_k = H2GCNLayer(h_{k-1}).
    final = Linear(concat(h_1, h_2, ..., h_n)) → embed_dim.
    """

    def __init__(self, in_feats, hidden_dim, n_layers, embed_dim):
        super().__init__()
        # Each H2GCN layer produces hidden_dim (split ego/nbr inside).
        self.layers = thnn.ModuleList()
        if n_layers == 1:
            # Single-layer: output hidden_dim split
            self.layers.append(H2GCNLayer(in_feats, hidden_dim))
            total = hidden_dim
        else:
            self.layers.append(H2GCNLayer(in_feats, hidden_dim))
            for _ in range(1, n_layers):
                self.layers.append(H2GCNLayer(hidden_dim, hidden_dim))
            total = hidden_dim * n_layers  # concat of all intermediate
        self.final = thnn.Linear(total, embed_dim)

    def forward(self, blocks, feats):
        if not isinstance(blocks, (list, tuple)):
            blocks = [blocks]
        h = feats
        hs = []
        for i, layer in enumerate(self.layers):
            block = blocks[i] if i < len(blocks) else blocks[-1]
            # For next layer, we need to slice h to dst_nodes of this block
            h = layer(block, h)
            h = F.relu(h)
            hs.append(h)
        # If multiple layers, concat; else single
        if len(hs) == 1:
            concat = hs[0]
        else:
            # Each h is at dst-node count of its block. For blocks.
            # We need them all at final dst-node count. Simplest: take
            # from each layer, truncate to the smallest (final dst).
            min_n = min(x.shape[0] for x in hs)
            concat = th.cat([x[:min_n] for x in hs], dim=-1)
        return self.final(concat)


def _build_h2gcn_layers(in_feats, hidden_dim, n_layers, embed_dim):
    """H2GCN backbone wrapped as a ModuleList with a single composite module.
    Forward path uses _H2GCNStack which expects the full block list."""
    stack = _H2GCNStack(in_feats, hidden_dim, n_layers, embed_dim)
    layers = thnn.ModuleList()
    layers.append(stack)
    return layers


def _build_backbone(backbone, in_feats, hidden_dim, n_layers, embed_dim,
                    n_heads=4, feat_drop=0.1, attn_drop=0.1):
    """Dispatch to the requested backbone."""
    if backbone == 'gat':
        return _build_gat_layers(in_feats, hidden_dim, n_layers, embed_dim,
                                  n_heads, feat_drop, attn_drop)
    elif backbone == 'gcn':
        return _build_gcn_layers(in_feats, hidden_dim, n_layers, embed_dim)
    elif backbone == 'sgc':
        return _build_sgc_layers(in_feats, hidden_dim, n_layers, embed_dim)
    elif backbone == 'sage':
        return _build_sage_layers(in_feats, hidden_dim, n_layers, embed_dim)
    elif backbone == 'h2gcn':
        return _build_h2gcn_layers(in_feats, hidden_dim, n_layers, embed_dim)
    else:
        raise ValueError(f'Unknown backbone: {backbone}')


def _build_projector(embed_dim: int) -> thnn.Sequential:
    """3-layer MLP projection head with BatchNorm + GELU (SimCLR v2 style).

    BN before activation prevents representation collapse and makes the
    contrastive loss landscape smoother, leading to better downstream
    linear separability (Chen & He, 2021; Chen et al., 2020).
    """
    return thnn.Sequential(
        thnn.Linear(embed_dim, embed_dim * 2),
        thnn.BatchNorm1d(embed_dim * 2),
        thnn.GELU(),
        thnn.Linear(embed_dim * 2, embed_dim),
        thnn.BatchNorm1d(embed_dim),
        thnn.GELU(),
        thnn.Linear(embed_dim, embed_dim),
    )


# ─────────────────────────────────────────────────────────────────────────────
# KAIROS encoder
# ─────────────────────────────────────────────────────────────────────────────

class KairosEncoder(thnn.Module):
    def __init__(
        self,
        in_feats: int,
        hidden_dim: int,
        n_layers: int,
        embed_dim: int,
        norm: str = "both",
        activation=F.relu,
        dropout: float = 0.0,
        n_heads: int = 4,
        backbone: str = 'gat',  # 'gat' | 'gcn' | 'sgc' | 'sage' | 'h2gcn'
    ):
        super().__init__()
        self.n_layers = n_layers
        self.in_feats = in_feats
        self.backbone = backbone
        self.dropout = thnn.Dropout(dropout)
        self.act_orig = thnn.PReLU()
        self.act_diff = thnn.PReLU()

        # ── Original-view backbone ────────────────────────────────────────
        self.orig_layers = _build_backbone(
            backbone, in_feats, hidden_dim, n_layers, embed_dim,
            n_heads=n_heads, feat_drop=0.1, attn_drop=0.1,
        )
        self.orig_linear = thnn.Linear(embed_dim, embed_dim)

        # ── PPR-structural-view backbone (separate weights) ───────────────
        self.diff_layers = _build_backbone(
            backbone, in_feats, hidden_dim, n_layers, embed_dim,
            n_heads=n_heads, feat_drop=0.1, attn_drop=0.1,
        )
        self.diff_linear = thnn.Linear(embed_dim, embed_dim)

        # ── Projection heads (3-layer BN — contrastive space only) ───────
        self.orig_projector = _build_projector(embed_dim)
        self.diff_projector = _build_projector(embed_dim)

        # ── Learnable contrastive temperature ─────────────────────────────
        self.log_tau = thnn.Parameter(th.tensor(math.log(0.07)))

        # ── Koopman temporal operator ─────────────────────────────────────
        self.koopman = KoopmanHead(embed_dim)

    @property
    def tau(self) -> th.Tensor:
        return self.log_tau.exp().clamp(0.01, 1.0)

    def _forward_backbone(self, layers, linear, act, blocks, features,
                           edge_weights=None):
        if self.backbone == 'h2gcn':
            # H2GCN is wrapped as a single composite module that consumes all blocks.
            h = layers[0](blocks, features)
            return act(linear(h))
        h = features
        for i, (layer, block) in enumerate(zip(layers, blocks)):
            # Pass edge_weight if GraphConv and edge_weights provided
            if edge_weights is not None and self.backbone in ('gcn', 'sgc'):
                ew = edge_weights[i] if isinstance(edge_weights, (list, tuple)) else edge_weights
                h = layer(block, h, edge_weight=ew)
            else:
                h = layer(block, h)
            h = h.flatten(1)
        return act(linear(h))

    def encode_orig(self, blocks, features, edge_weights=None):
        """Raw features → pre-projection embedding (used for eval + Koopman)."""
        return self._forward_backbone(
            self.orig_layers, self.orig_linear, self.act_orig, blocks, features,
            edge_weights=edge_weights,
        )

    def encode_diff(self, blocks, features, edge_weights=None):
        """PPR-diffused features / PPR-graph → structural-view embedding.
        If edge_weights provided, GCN/SGC will use them in message passing
        (CLDG++-style PPR-graph mode)."""
        return self._forward_backbone(
            self.diff_layers, self.diff_linear, self.act_diff, blocks, features,
            edge_weights=edge_weights,
        )

    def project_orig(self, h):
        return F.normalize(self.orig_projector(h), p=2, dim=1)

    def project_diff(self, h):
        return F.normalize(self.diff_projector(h), p=2, dim=1)
