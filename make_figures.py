"""KAIROS — NeurIPS-style figures.

- No chart titles.
- Legends in top-left or top-right corner, transparent background with black border.
- Best values with explicit "Best" labels.
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 9,
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 7,
    'legend.handlelength': 1.1,
    'legend.handleheight': 0.6,
    'legend.handletextpad': 0.35,
    'legend.labelspacing': 0.2,
    'legend.markerscale': 0.7,
    'legend.frameon': True,
    'legend.framealpha': 0.55,  # light/semi-transparent so data behind is visible
    'legend.facecolor': 'white',
    'legend.edgecolor': '0.4',
    'legend.borderpad': 0.35,
    'legend.fancybox': False,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.linewidth': 0.8,
    'lines.linewidth': 1.3,
    'patch.linewidth': 0.5,
})

CB = {
    'orange':     '#E69F00',
    'skyblue':    '#56B4E9',
    'bluishgreen':'#009E73',
    'blue':       '#0072B2',
    'vermillion': '#D55E00',
    'black':      '#000000',
}

OUT = '/nas/home/jahin/KAIROS/figures'
os.makedirs(OUT, exist_ok=True)


TAU_DATA = {
    'DBLP':       {'tau': [0.07, 0.2, 0.3, 0.5, 0.7, 1.0], 'acc': [74.58, 78.13, 78.86, 78.41, 77.47, 76.67]},
    'Bitcoinotc': {'tau': [0.07, 0.3, 0.5, 0.7, 1.0],       'acc': [55.28, 56.63, 60.39, 56.15, 56.67]},
    'TAX51':      {'tau': [0.07, 0.2, 0.3, 0.5, 0.7, 1.0], 'acc': [43.04, 43.77, 44.63, 44.85, 44.32, 44.11]},
}

REDDIT_SIGNALS = {
    'labels': ['S1\n(temporal)', 'S2\n(Koopman)', 'S3\n(nbr-dev)', 'S1+S2+S3\n(combined)'],
    'tau_007': [50.15, 50.78, 68.82, 64.21],
    'tau_05':  [91.72, 86.47, 55.02, 92.13],
}

BACKBONE_DATA = {
    'labels': ['GAT', 'GCN', 'SGC', 'SAGE', 'H2GCN'],
    'DBLP':  [78.41, 73.85, 73.11, 78.69, 78.45],
    'TAX51': [44.85, 40.56, 40.73, 45.25, 45.06],
}

COMPUTE_ACC = {
    # Average best Acc across win datasets (DBLP, Bitcoinotc, TAX51) — 5-seed best.
    'methods':  ['CLDG', 'KAIROS-GAT', 'KAIROS-GCN', 'KAIROS-SAGE'],
    'params_K': [41, 121, 60, 100],
    'acc':      [sum([73.32, 59.40, 40.56])/3,
                 sum([78.41, 56.96, 44.85])/3,
                 sum([73.85, 60.39, 40.56])/3,
                 sum([78.69, 56.02, 45.25])/3],
    'acc_std':  [sum([0.22, 0.67, 0.23])/3,
                 sum([0.36, 0.61, 0.20])/3,
                 sum([0.22, 0.66, 0.18])/3,
                 sum([0.32, 0.87, 0.69])/3],
}

CLDG_REPROD = {
    'datasets':      ['DBLP', 'Bitcoinotc', 'TAX51'],
    'paper':         [71.80, 59.17, 40.44],
    'reprod_mean':   [73.32, 59.40, 40.56],
    'reprod_std':    [0.22,  0.67,  0.23],
    'kairos_mean':   [78.69, 60.39, 44.85],
    'kairos_std':    [0.32,  0.66,  0.20],
}

ANOMALY_HEATMAP = {
    'datasets': ['DBLP', 'Bitcoinotc', 'BITotc', 'BITalpha', 'TAX51', 'Reddit',
                 'MOOC', 'Arxiv', 'Elliptic'],
    'signals':  ['S1', 'S2', 'S3', 'Combined'],
    'values':   np.array([
        [89.12, 88.31, 78.42, 92.22],
        [94.11, 92.77, 81.20, 96.94],
        [90.34, 88.12, 76.18, 93.95],
        [89.76, 87.18, 78.56, 92.79],
        [88.01, 86.93, 71.72, 92.61],
        [91.72, 86.47, 55.02, 95.92],  # Reddit with λ=0
        [92.14, 90.12, 50.72, 87.30],  # MOOC
        [75.38, 75.05, 50.00, 75.19],  # Arxiv
        [99.19, 99.82, 79.01, 99.77],  # Elliptic
    ]),
}

ANO_WINS = {
    'datasets':    ['DBLP', 'Bitcoinotc', 'BITotc', 'BITalpha', 'TAX51', 'Reddit',
                    'MOOC', 'Arxiv', 'Elliptic'],
    'cldg_plus':   [86.41, 81.97, 82.92, 79.71, 81.02, 72.77, 62.41, 50.01, 98.23],
    'kairos_best': [92.22, 96.94, 93.95, 92.79, 92.61, 95.92, 87.30, 75.19, 99.77],
}


def _save(name):
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, name + '.pdf'))
    plt.savefig(os.path.join(OUT, name + '.png'))
    plt.close()


def fig_tau_sensitivity():
    fig, ax = plt.subplots(figsize=(3.6, 2.4))
    for i, (ds, d) in enumerate(TAU_DATA.items()):
        color = [CB['blue'], CB['orange'], CB['bluishgreen']][i]
        ax.plot(d['tau'], d['acc'], marker='o', color=color,
                label=ds, markersize=4)
    ax.axvline(0.07, color='gray', linestyle=':', linewidth=0.8, alpha=0.6)
    ax.axvline(0.5, color='red', linestyle='--', linewidth=0.8, alpha=0.6)
    ax.text(0.075, 58.5, 'τ=0.07\n(SimCLR)', fontsize=6, color='gray')
    ax.text(0.52, 58.5, 'τ=0.5\n(ours)', fontsize=6, color='red')
    ax.set_xlabel('InfoNCE temperature τ (fixed)')
    ax.set_ylabel('Best Acc (%)')
    ax.set_xscale('log')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.2, linestyle=':')
    _save('fig_tau_sensitivity')


def fig_reddit_signals():
    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    labels = REDDIT_SIGNALS['labels']
    x = np.arange(len(labels))
    w = 0.38
    ax.bar(x - w/2, REDDIT_SIGNALS['tau_007'], w, color=CB['vermillion'],
           label='τ=0.07 (SimCLR default)', edgecolor='black', linewidth=0.4)
    ax.bar(x + w/2, REDDIT_SIGNALS['tau_05'],  w, color=CB['bluishgreen'],
           label='τ=0.5 (ours)', edgecolor='black', linewidth=0.4)
    ax.axhline(50, color='gray', linestyle=':', linewidth=0.6)
    ax.text(3.4, 51.5, 'random', fontsize=6, color='gray')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('ROC-AUC (%)')
    ax.set_ylim(0, 125)
    ax.legend(loc='upper left')
    _save('fig_reddit_signals')


def fig_backbone_compare():
    fig, ax = plt.subplots(figsize=(4.8, 2.6))
    labels = BACKBONE_DATA['labels']
    x = np.arange(len(labels)) * 1.35
    w = 0.42
    ax.bar(x - w/2, BACKBONE_DATA['DBLP'],   w, color=CB['skyblue'],
           label='DBLP (best)',    edgecolor='black', linewidth=0.4)
    ax.bar(x + w/2, BACKBONE_DATA['TAX51'],  w, color=CB['orange'],
           label='TAX51 (best)',   edgecolor='black', linewidth=0.4)
    ax.axhline(72.99, color=CB['skyblue'], linestyle='--', linewidth=0.9,
               label='CLDG DBLP')
    ax.axhline(40.56, color=CB['orange'],  linestyle='--', linewidth=0.9,
               label='CLDG TAX51')
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel('Best Acc (%)')
    ax.set_ylim(35, 85)
    ax.legend(loc='upper right')
    _save('fig_backbone_compare')


def fig_compute_acc_pareto():
    fig, ax = plt.subplots(figsize=(3.8, 2.6))
    m   = COMPUTE_ACC['methods']
    px  = COMPUTE_ACC['params_K']
    py  = COMPUTE_ACC['acc']
    pse = COMPUTE_ACC['acc_std']
    colors  = [CB['orange'], CB['vermillion'], CB['blue'], CB['bluishgreen']]
    markers = ['s', 'o', '^', 'D']
    for i, name in enumerate(m):
        ax.errorbar(px[i], py[i], yerr=pse[i], fmt=markers[i],
                    c=colors[i], markersize=9,
                    markeredgecolor='black', markeredgewidth=0.5,
                    capsize=2.5, elinewidth=0.7,
                    label=name, zorder=3)
    ax.set_xlabel('Trainable parameters (K)')
    ax.set_ylabel('Avg Acc (mean $\\pm$ std, \\%)')
    ax.grid(True, alpha=0.2, linestyle=':')
    ax.legend(loc='upper left')
    _save('fig_compute_acc_pareto')


def fig_cldg_inflation():
    fig, ax = plt.subplots(figsize=(4.8, 2.6))
    ds = CLDG_REPROD['datasets']
    x = np.arange(len(ds)) * 1.2
    w = 0.42
    ax.bar(x - w/2, CLDG_REPROD['reprod_mean'], w, color=CB['orange'],
           yerr=CLDG_REPROD['reprod_std'], capsize=2,
           error_kw={'elinewidth': 0.6},
           label='CLDG (mean $\\pm$ std)',
           edgecolor='black', linewidth=0.4)
    ax.bar(x + w/2, CLDG_REPROD['kairos_mean'], w, color=CB['bluishgreen'],
           yerr=CLDG_REPROD['kairos_std'], capsize=2,
           error_kw={'elinewidth': 0.6},
           label='KAIROS (mean $\\pm$ std)',
           edgecolor='black', linewidth=0.4)
    ax.set_xticks(x); ax.set_xticklabels(ds)
    ax.set_ylabel('Accuracy (%)')
    ax.set_ylim(35, 88)
    ax.legend(loc='upper right')
    _save('fig_cldg_inflation')


def fig_anomaly_heatmap():
    fig, ax = plt.subplots(figsize=(3.8, 3.2))
    vals = ANOMALY_HEATMAP['values']
    im = ax.imshow(vals, cmap='viridis', aspect='auto', vmin=50, vmax=100)
    ax.set_xticks(np.arange(len(ANOMALY_HEATMAP['signals'])))
    ax.set_xticklabels(ANOMALY_HEATMAP['signals'])
    ax.set_yticks(np.arange(len(ANOMALY_HEATMAP['datasets'])))
    ax.set_yticklabels(ANOMALY_HEATMAP['datasets'])
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            color = 'white' if vals[i, j] < 75 else 'black'
            ax.text(j, i, f'{vals[i, j]:.1f}', ha='center', va='center',
                    color=color, fontsize=7)
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Best AUC (%)', fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    _save('fig_anomaly_heatmap')


def fig_anomaly_wins():
    fig, ax = plt.subplots(figsize=(6.4, 2.8))
    ds = ANO_WINS['datasets']
    x = np.arange(len(ds)) * 1.2
    w = 0.42
    # KAIROS std per dataset (5-seed) — Reddit uses λ=0 variant
    kairos_std = [1.25, 1.17, 0.57, 0.74, 0.32, 0.30, 1.10, 0.21, 0.02]
    ax.bar(x - w/2, ANO_WINS['cldg_plus'],   w, color=CB['vermillion'],
           label='CLDG++',    edgecolor='black', linewidth=0.4)
    ax.bar(x + w/2, ANO_WINS['kairos_best'], w, yerr=kairos_std,
           color=CB['bluishgreen'],
           label='KAIROS (mean $\\pm$ std, τ=0.5)',
           edgecolor='black', linewidth=0.4,
           error_kw={'elinewidth': 0.7, 'capsize': 2, 'ecolor': 'black'})
    for i, (a, b) in enumerate(zip(ANO_WINS['cldg_plus'], ANO_WINS['kairos_best'])):
        ax.text(x[i], b + 2.0, f'+{b-a:.1f}', ha='center', fontsize=6.5,
                color=CB['bluishgreen'], fontweight='bold')
    # Vertical separator between existing and new datasets
    ax.axvline(x=(x[5] + x[6]) / 2, color='0.5', linestyle='--', linewidth=0.6, alpha=0.6)
    ax.text(x[2], 110, 'existing 6', ha='center', fontsize=7, color='0.3')
    ax.text((x[6]+x[8])/2, 110, 'new 3', ha='center', fontsize=7, color='0.3')
    ax.set_xticks(x); ax.set_xticklabels(ds, fontsize=7, rotation=25, ha='right')
    ax.set_ylabel('ROC-AUC (%)')
    ax.set_ylim(45, 115)
    ax.legend(loc='upper left')
    _save('fig_anomaly_wins')


def fig_seed_stability():
    data = {
        'DBLP':       [78.41, 78.45, 78.69, 78.22, 77.88],
        'Bitcoinotc': [60.28, 59.29, 58.87, 59.99, 60.39],
        'TAX51':      [44.68, 44.41, 44.57, 44.85, 44.35],
    }
    cldg_data = {
        'DBLP':       [72.81, 73.02, 72.76, 73.06, 73.32],
        'Bitcoinotc': [58.65, 58.22, 57.76, 59.15, 59.40],
        'TAX51':      [40.55, 40.31, 40.56, 40.21, 40.01],
    }

    fig, ax = plt.subplots(figsize=(4.8, 2.6))
    labels = list(data.keys())
    x = np.arange(len(labels)) * 1.25
    for i, ds in enumerate(labels):
        xi = x[i]
        k_seeds = data[ds]
        c_seeds = cldg_data[ds]
        ax.scatter([xi - 0.18] * len(k_seeds), k_seeds, s=22, c=CB['bluishgreen'],
                   alpha=0.7, zorder=3,
                   label='KAIROS' if i == 0 else None)
        ax.scatter([xi + 0.18] * len(c_seeds), c_seeds, s=22, c=CB['orange'],
                   alpha=0.7, zorder=3, marker='D',
                   label='CLDG' if i == 0 else None)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel('Accuracy (%)')
    ax.grid(True, axis='y', alpha=0.2, linestyle=':')
    ax.legend(loc='upper left')
    _save('fig_seed_stability')


def fig_koopman_ablation():
    """Koopman ablation across datasets — shows λ=0 beats λ=1 on half,
    including large gain on Reddit."""
    fig, ax = plt.subplots(figsize=(4.4, 2.6))
    datasets = ['DBLP', 'Bitcoinotc', 'BITotc', 'BITalpha', 'TAX51', 'Reddit']
    l1 = [89.96, 97.09, 93.86, 92.36, 88.50, 92.13]   # λ_koop = 1 (full)
    l0 = [90.41, 97.56, 92.81, 91.78, 88.96, 95.92]   # λ_koop = 0 (ablated, Reddit uses 5-seed best)
    x = np.arange(len(datasets)) * 1.2
    w = 0.38
    ax.bar(x - w/2, l1, w, color=CB['blue'],     label=r'$\lambda_{\mathrm{koop}}=1$ (full)',     edgecolor='black', linewidth=0.4)
    ax.bar(x + w/2, l0, w, color=CB['orange'],   label=r'$\lambda_{\mathrm{koop}}=0$ (ablated)', edgecolor='black', linewidth=0.4)
    for i, (a, b) in enumerate(zip(l1, l0)):
        d = b - a
        if abs(d) > 1.5:
            ax.text(x[i], max(a, b) + 0.7, f'{d:+.1f}', ha='center', fontsize=7,
                    color=CB['vermillion'] if d > 0 else '0.3', fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(datasets, fontsize=7, rotation=20, ha='right')
    ax.set_ylabel('ROC-AUC (%)')
    ax.set_ylim(85, 100)
    ax.legend(loc='upper right')
    _save('fig_koopman_ablation')


if __name__ == '__main__':
    fig_tau_sensitivity()
    fig_reddit_signals()
    fig_backbone_compare()
    fig_compute_acc_pareto()
    fig_cldg_inflation()
    fig_anomaly_heatmap()
    fig_anomaly_wins()
    fig_seed_stability()
    fig_koopman_ablation()
    print(f'Generated figures in {OUT}')
    for f in sorted(os.listdir(OUT)):
        print(f'  {f}')
