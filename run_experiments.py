"""
KAIROS — Full Experiment Runner
=================================
Run from repo root:   python run_experiments.py
On Colab:             !python /content/drive/.../run_experiments.py

Set EPOCHS = 1 for a quick sanity check.
Set EPOCHS = None to use the per-dataset paper epoch counts.

Reproduces the final KAIROS configuration of paper Tables 3 (classification)
and 4 (anomaly detection): per-dataset backbone (Table A.tab:config) with
fixed τ = 0.5. To reproduce the canonical (GAT, learnable τ ≈ 0.07) baseline
row of the component ablation, use KAIROS/gcn_multiseed.sh or
`KAIROS/ablate.py --ablation none --backbone gat`.
"""

import csv
import importlib
import os
import random
import subprocess
import sys
import types as _types

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
KAIROS_DIR = os.path.join(SCRIPT_DIR, "KAIROS")
DATA_DIR   = os.path.join(SCRIPT_DIR, "Data")

os.chdir(KAIROS_DIR)
sys.path.insert(0, KAIROS_DIR)


# ── 1. Dependencies ────────────────────────────────────────────────────────────


def _pip(*args):
    subprocess.run([sys.executable, "-m", "pip", "install", *args, "-q"], check=True)


print("=" * 70)
print("Step 1 — Installing / verifying dependencies")
print("=" * 70)

import torch

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA is not available. KAIROS requires a GPU.\n"
        "On Colab: Runtime → Change runtime type → Hardware accelerator: GPU (T4).\n"
        "Then restart the runtime and re-run this script."
    )

cuda_ver = torch.version.cuda
print(f"Detected PyTorch {torch.__version__} with CUDA {cuda_ver}")

# Try CUDA-enabled DGL across a matrix of torch/cu versions. We verify each
# candidate by actually moving a graph to GPU — if that fails, uninstall and
# try the next combination. If none work, raise (do NOT silently fall back
# to CPU DGL, which would force the whole training loop onto CPU).
installed = False
major, minor = [int(x) for x in cuda_ver.split(".")[:2]]
tv = ".".join(torch.__version__.split(".")[:2])
tv_list = [tv]
try:
    tmaj, tmin = int(tv.split(".")[0]), int(tv.split(".")[1])
    for s in range(1, 8):
        p = tmin - s
        if p >= 0:
            tv_list.append(f"{tmaj}.{p}")
except Exception:
    pass

if major >= 12:
    cu_list = [f"cu12{minor}", "cu124", "cu121", "cu118"]
elif major == 11 and minor >= 8:
    cu_list = ["cu118"]
else:
    cu_list = ["cu117", "cu116"]

# de-dup preserving order
seen = set()
cu_list = [c for c in cu_list if not (c in seen or seen.add(c))]

for tv_try in tv_list:
    for cu in cu_list:
        url = f"https://data.dgl.ai/wheels/torch-{tv_try}/{cu}/repo.html"
        ret = subprocess.run(
            [sys.executable, "-m", "pip", "install", "dgl", "-f", url, "-q"],
            capture_output=True,
            text=True,
        )
        if ret.returncode != 0:
            continue
        chk = subprocess.run(
            [
                sys.executable,
                "-c",
                "import dgl,torch; g=dgl.graph(([0],[1])).to('cuda:0'); "
                "assert g.device.type=='cuda'; print('OK')",
            ],
            capture_output=True,
            text=True,
        )
        if "OK" in chk.stdout:
            print(f"DGL (CUDA) installed: torch-{tv_try}/{cu}")
            installed = True
            break
        subprocess.run(
            [sys.executable, "-m", "pip", "uninstall", "dgl", "-y", "-q"],
            capture_output=True,
        )
    if installed:
        break

if not installed:
    raise RuntimeError(
        "Failed to install a CUDA-enabled DGL wheel matching this PyTorch.\n"
        f"  torch: {torch.__version__}   cuda: {cuda_ver}\n"
        "Options:\n"
        "  1. On Colab, pin torch to a version with published DGL wheels, e.g.\n"
        "       !pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121\n"
        "       !pip install dgl -f https://data.dgl.ai/wheels/torch-2.4/cu121/repo.html\n"
        "  2. Check https://www.dgl.ai/pages/start.html for the latest supported combos."
    )

_pip("scikit-learn", "scipy", "pandas", "numpy")
importlib.invalidate_caches()

# ── Stubs for torchdata / graphbolt ───────────────────────────────────────────


class _CAM(_types.ModuleType):
    class _D:
        def __init__(self, *a, **kw):
            pass

        def __iter__(self):
            return iter([])

        def __call__(self, *a, **kw):
            return self

        def __getattr__(self, _):
            return lambda *a, **kw: self

    def __getattr__(self, name):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        return self._D


def _stub(name):
    if name not in sys.modules:
        sys.modules[name] = _CAM(name)
    return sys.modules[name]


for _n in [
    "torchdata",
    "torchdata.datapipes",
    "torchdata.datapipes.iter",
    "torchdata.datapipes.map",
    "torchdata.dataloader2",
    "torchdata.dataloader2.graph",
    "torchdata.dataloader2.adapter",
]:
    _stub(_n)
sys.modules["torchdata"].datapipes = sys.modules["torchdata.datapipes"]
sys.modules["torchdata"].dataloader2 = sys.modules["torchdata.dataloader2"]
sys.modules["torchdata.datapipes"].iter = sys.modules["torchdata.datapipes.iter"]
sys.modules["torchdata.datapipes"].map = sys.modules["torchdata.datapipes.map"]
sys.modules["torchdata.dataloader2"].graph = sys.modules["torchdata.dataloader2.graph"]
sys.modules["torchdata.dataloader2"].adapter = sys.modules[
    "torchdata.dataloader2.adapter"
]

for _gb in [
    "dgl.graphbolt",
    "dgl.graphbolt.base",
    "dgl.graphbolt.dataloader",
    "dgl.graphbolt.feature_fetcher",
    "dgl.graphbolt.minibatch_transformer",
]:
    _stub(_gb)

import dgl
import numpy
import pandas
import scipy
import sklearn

print(
    f"PyTorch {torch.__version__} | DGL {dgl.__version__} | CUDA {torch.cuda.is_available()}"
)
assert torch.cuda.is_available(), "CUDA not available after install — cannot continue."
print(f"GPU: {torch.cuda.get_device_name(0)}")

# Sanity-check that DGL can actually execute on GPU. If this fails, DGL is
# CPU-only and silently running on CPU would defeat the purpose.
_probe = dgl.graph(([0, 1], [1, 0])).to("cuda:0")
assert _probe.device.type == "cuda", "DGL graph did not move to CUDA."
del _probe

print("\n" + "=" * 70)
print("Dependency Versions:")
print(f"Python: {sys.version.split(' ')[0]}")
print(f"PyTorch: {torch.__version__}")
print(f"DGL: {dgl.__version__}")
print(f"NumPy: {numpy.__version__}")
print(f"Pandas: {pandas.__version__}")
print(f"SciPy: {scipy.__version__}")
print(f"Scikit-learn: {sklearn.__version__}")
print("=" * 70 + "\n")

# ── 2. Import KAIROS ──────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("Step 2 — Importing KAIROS model")
print("=" * 70)
import math

import main as kmain
import models as kmodels
from main import train

# ── Final-KAIROS configuration patch ──────────────────────────────────────────
# Paper Table 3/4 numbers come from the final KAIROS configuration:
#   • per-dataset backbone (Table tab:config)
#   • fixed τ = 0.5 (paper Section 5; ablate.py --ablation fixed_tau --tau_val 0.5)
# This wrapper previously called only the canonical (GAT, learnable τ) baseline,
# which reproduces the first row of tab:component_ablation but not Table 3/4.
# Apply the same fixed-τ monkey-patch ablate.py uses, so that running this script
# end-to-end reproduces the headline tables. (See ablate.py:_patch_fixed_tau.)

_FIXED_TAU = 0.5

_orig_encoder_init = kmodels.KairosEncoder.__init__


def _init_fixed_tau(self, *a, **kw):
    _orig_encoder_init(self, *a, **kw)
    del self.log_tau
    self.register_buffer("log_tau", torch.tensor(math.log(_FIXED_TAU)))


kmodels.KairosEncoder.__init__ = _init_fixed_tau
print(f"[run_experiments] τ patched to fixed {_FIXED_TAU} (non-trainable)")

print("KAIROS imported successfully.")

# ── 3. Settings ────────────────────────────────────────────────────────────────

GPU_ID = 0
EPOCHS = None  # set None for full paper runs

# KAIROS hyperparameters
LAMBDA_KOOP     = 1.0   # Koopman temporal-linearization loss weight (default)
LAMBDA_KOOP_REG = 0.01  # Koopman orthogonality regularizer weight
ALPHA = 0.15            # PPR teleport probability (structural view)

# Per-dataset λ_koop override for anomaly detection.
# Reddit violates the linear Koopman assumption (irregular community dynamics),
# so λ_koop=0 is optimal there (Table A5 / paper Section 5.4).
# All other datasets use the default λ_koop=1.0.
ANO_LAMBDA_KOOP = {
    "reddit": 0.0,
}

ORIGINAL_EPOCHS = {
    "dblp": (200, 200),
    "bitcoinotc": (25, 25),
    "bitotc": (50, 50),
    "bitalpha": (200, 100),
    "tax51": (200, 200),
    "reddit": (200, 200),
}

SEP = "-" * 70

# ── 4. Experiment definitions ─────────────────────────────────────────────────

# Per-dataset backbone matches paper Table tab:config (Appendix A).
EXPERIMENTS = [
    dict(
        ds_key="dblp",
        ds_label="DBLP",
        backbone="sage",
        clf_fanout=[20, 20],
        clf_snaps=4,
        clf_views=4,
        clf_strat="sequential",
        clf_dl=4096,
        ano_fanout=[20, 20],
        ano_snaps=4,
        ano_views=4,
        ano_strat="sequential",
        ano_dl=4096,
    ),
    dict(
        ds_key="bitcoinotc",
        ds_label="Bitcoinotc",
        backbone="gcn",
        # Fanouts are a safety fallback; for graphs ≤ FULL_NBR_THRESH nodes
        # the adaptive sampler in main.py uses full-neighbour sampling instead.
        clf_fanout=[20, 20],
        clf_snaps=4,
        clf_views=3,
        clf_strat="sequential",
        clf_dl=64,
        ano_fanout=[20, 20],
        ano_snaps=4,
        ano_views=4,
        ano_strat="sequential",
        ano_dl=64,
    ),
    dict(
        ds_key="bitotc",
        ds_label="BITotc",
        backbone="sgc",
        clf_fanout=[20, 20],
        clf_snaps=4,
        clf_views=4,
        clf_strat="random",
        clf_dl=4096,
        ano_fanout=[20, 20],
        ano_snaps=4,
        ano_views=4,
        ano_strat="sequential",
        ano_dl=4096,
    ),
    dict(
        ds_key="bitalpha",
        ds_label="BITalpha",
        backbone="gcn",
        clf_fanout=[20, 20],
        clf_snaps=6,
        clf_views=4,
        clf_strat="sequential",
        clf_dl=4096,
        ano_fanout=[20, 20],
        ano_snaps=5,
        ano_views=5,
        ano_strat="sequential",
        ano_dl=4096,
    ),
    dict(
        ds_key="tax51",
        ds_label="TAX51",
        backbone="gat",
        clf_fanout=[20, 20],
        clf_snaps=8,
        clf_views=5,
        clf_strat="random",
        clf_dl=4096,
        ano_fanout=[20, 20],
        ano_snaps=4,
        ano_views=4,
        ano_strat="sequential",
        ano_dl=4096,
    ),
    dict(
        ds_key="reddit",
        ds_label="Reddit",
        backbone="gcn",
        clf_fanout=[20, 20],
        clf_snaps=5,
        clf_views=4,
        clf_strat="random",
        clf_dl=4096,
        ano_fanout=[20, 20],
        ano_snaps=4,
        ano_views=4,
        ano_strat="sequential",
        ano_dl=4096,
    ),
]


# ── 5. CSV helpers ────────────────────────────────────────────────────────────

CSV_CLF = os.path.join(SCRIPT_DIR, "table_3_classification.csv")
CSV_ANO = os.path.join(SCRIPT_DIR, "table_4_anomaly.csv")

DS_TO_COL = {
    "dblp": "DBLP",
    "bitcoinotc": "Bitcoinotc",
    "bitotc": "BITotc",
    "bitalpha": "BITalpha",
    "tax51": "TAX51",
    "reddit": "Reddit",
}


def _rw_csv(path, method_name, updates: dict):
    """Read CSV, upsert the KAIROS row, write back."""
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames

    row = next((r for r in rows if r["Method"] == method_name), None)
    if row is None:
        row = {c: "" for c in fieldnames}
        row["Method"] = method_name
        rows.append(row)

    for k, v in updates.items():
        row[k] = v

    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def _update_clf_csv(ds_key, accuracy, weighted_f1):
    col = DS_TO_COL[ds_key]
    _rw_csv(
        CSV_CLF,
        "KAIROS",
        {
            "Input": "X, A, S, T",   # X=raw PE, S=PPR-struct, T=temporal
            f"{col}_Acc": f"{accuracy*100:.2f}",
            f"{col}_Wei": f"{weighted_f1*100:.2f}",
        },
    )
    print(f"  {col} Acc={accuracy*100:.2f}%  Wei={weighted_f1*100:.2f}%")


def _update_ano_csv(ds_key, auc):
    col = DS_TO_COL[ds_key]
    _rw_csv(CSV_ANO, "KAIROS", {f"{col}_AUC": f"{auc*100:.2f}"})
    print(f"  {col} AUC={auc*100:.2f}%")


# ── 6. Run ────────────────────────────────────────────────────────────────────


def run_one(ds_key, ds_label, task, backbone, fanout, snaps, views, strat, dl, epochs,
            lambda_koop=None):
    print(f"\n{SEP}")
    print(f"{ds_label} | {task} | backbone={backbone}")
    print(SEP)
    # Re-seed before every run for per-experiment reproducibility.
    import numpy as _np, torch as _th
    random.seed(24); _np.random.seed(24); _th.manual_seed(24); _th.cuda.manual_seed_all(24)
    lkoop = LAMBDA_KOOP if lambda_koop is None else lambda_koop
    return train(
        dataset=ds_key,
        task=task,
        hidden_dim=128,
        n_classes=64,
        n_layers=2,
        backbone=backbone,
        fanouts=fanout,
        snapshots=snaps,
        views=views,
        strategy=strat,
        readout="max",
        batch_size=256,
        dataloader_size=dl,
        alpha=ALPHA,
        lambda_koop=lkoop,
        lambda_koop_reg=LAMBDA_KOOP_REG,
        num_workers=0,
        epochs=epochs,
        GPU=GPU_ID,
    )


if __name__ == "__main__":
    total = len(EXPERIMENTS) * 2
    done = 0

    for exp in EXPERIMENTS:
        k = exp["ds_key"]
        label = exp["ds_label"]
        orig_clf, orig_ano = ORIGINAL_EPOCHS[k]
        clf_ep = EPOCHS if EPOCHS is not None else orig_clf
        ano_ep = EPOCHS if EPOCHS is not None else orig_ano

        done += 1
        print(f"\n{'='*70}")
        print(f"Exp {done}/{total} — {label} | Classification  (epochs={clf_ep})")
        print("=" * 70)
        r = run_one(
            k,
            label,
            "classification",
            exp["backbone"],
            exp["clf_fanout"],
            exp["clf_snaps"],
            exp["clf_views"],
            exp["clf_strat"],
            exp["clf_dl"],
            clf_ep,
        )
        if r:
            _update_clf_csv(k, r["accuracy"], r["weighted_f1"])

        done += 1
        print(f"\n{'='*70}")
        print(f"Exp {done}/{total} — {label} | Anomaly Detection  (epochs={ano_ep})")
        print("=" * 70)
        ano_lkoop = ANO_LAMBDA_KOOP.get(k, LAMBDA_KOOP)
        r = run_one(
            k,
            label,
            "anomaly_detection",
            exp["backbone"],
            exp["ano_fanout"],
            exp["ano_snaps"],
            exp["ano_views"],
            exp["ano_strat"],
            exp["ano_dl"],
            ano_ep,
            lambda_koop=ano_lkoop,
        )
        if r:
            _update_ano_csv(k, r["auc"])

    print(f"\n{'='*70}")
    print("All experiments done.")
    print(f"Classification  → {CSV_CLF}")
    print(f"Anomaly detect. → {CSV_ANO}")
    print("=" * 70)
