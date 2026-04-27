# KAIROS: Koopman-Aligned Invariant Representations for Open Dynamic Systems

A self-supervised framework for dynamic graph representation learning that
achieves state-of-the-art anomaly detection across nine temporal graph
benchmarks and competitive-to-winning node classification.

---

## Benchmark scope

**9 datasets**:

| Dataset | Nodes | Edges | Classes | Source |
|---------|------:|------:|:-------:|--------|
| DBLP | 25,387 | 185,480 | 10 | CLDG (ICDE'23) |
| Bitcoinotc | 5,881 | 35,592 | 3 | CLDG |
| BITotc | 4,863 | 28,473 | 7 | CLDG |
| BITalpha | 3,219 | 19,364 | 7 | CLDG |
| TAX51 | 132,524 | 467,279 | 51 | CLDG |
| Reddit | 898,194 | 2,575,464 | 3 | CLDG |
| **MOOC** | 7,144 | 411,749 | 2 | JODIE (KDD'19) |
| **Arxiv** | 169,343 | 1,166,243 | 40 | OGB |
| **Elliptic** | 203,769 | 234,355 | 2 (fraud) | EvolveGCN / PyG |

---

## Running experiments

All experiment code lives under `KAIROS/`:

```bash
# KAIROS primary run on a single dataset
cd KAIROS
python ablate.py \
  --dataset dblp --task classification \
  --snapshots 4 --views 4 --strategy sequential \
  --dataloader_size 4096 --GPU 0 --epochs 200 --seed 24 \
  --ablation fixed_tau --tau_val 0.5 --backbone sage

# Simple baselines (LP/GCN/GAT/GraphSAGE/DGI/CCA-SSG)
python simple_baselines.py \
  --method gcn --dataset mooc --task classification --seed 24 --epochs 100
```

Regenerate all figures:
```bash
python make_figures.py
```

---

## Dependencies

```
torch>=1.13          (CUDA build)
dgl                  (CUDA build)
scikit-learn>=1.0
scipy>=1.7
pandas>=1.3
numpy>=1.21
ogb                  (for Arxiv dataset)
```

Install: `pip install -r requirements.txt`

---

## Repository layout

```
KAIROS/
├── README.md                    this file
├── make_figures.py              figure generator (8 figures)
├── bootstrap_best.py            bootstrap stat-sig analyzer
├── requirements.txt
├── KAIROS/                      model implementation
│   ├── main.py                  training loop + CLI
│   ├── models.py                KairosEncoder, KoopmanHead, LogReg
│   ├── utils.py                 graph loading, PPR, anomaly injection
│   ├── ablate.py                ablation runner (τ / backbone / Koopman)
│   ├── simple_baselines.py      LP/GCN/GAT/GraphSAGE/DGI/CCA-SSG
│   └── parallel_baselines.sh    batch runner for baselines
├── Data/                        datasets (symlinks to /tmp/ for large ones)
├── figures/                     generated PDFs and PNGs
└── runs/                        experiment logs (symlink to /tmp/)

```

---

## Reproducibility

All experiments use 5 seeds: `{24, 42, 7, 13, 99}`.
GPU non-determinism suppressed via `torch.backends.cudnn.deterministic = True`.

