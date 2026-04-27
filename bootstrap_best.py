"""Bootstrap analysis of best-of-5 KAIROS vs CLDG++ paper numbers.

For each dataset we resample 5 seeds with replacement B=10000 times, take
max each time, and count what fraction beats CLDG++'s reported number.
This answers: "Given our sampling, how likely is our best-seed > baseline?"
"""

import numpy as np

np.random.seed(42)
B = 10000

# KAIROS per-seed Acc (classification, best config per dataset)
kairos = {
    'DBLP':       [77.94, 78.27, 78.41, 78.22, 77.50],
    'Bitcoinotc': [60.28, 59.29, 58.87, 59.99, 60.39],
    'BITotc':     [64.06, 63.83, 63.08, 63.85, 63.85],
    'BITalpha':   [77.92, 77.97, 78.03, 78.03, 78.13],
    'TAX51':      [44.68, 44.41, 44.57, 44.85, 44.35],
}

# KAIROS per-seed AUC (anomaly, τ=0.5) — exact values from logs
kairos_ano = {
    'DBLP':       [92.22, 89.37, 90.39, 89.56, 89.19],
    'Bitcoinotc': [96.32, 95.60, 94.70, 94.05, 96.94],
    'BITotc':     [92.49, 93.45, 93.95, 93.51, 93.79],
    'BITalpha':   [92.02, 92.21, 90.87, 92.79, 91.43],
    'TAX51':      [92.15, 92.61, 91.99, 91.92, 91.77],
}

cldg_plus_clf = {'DBLP': 72.94, 'Bitcoinotc': 59.88, 'BITotc': 65.37,
                 'BITalpha': 80.63, 'TAX51': 41.05}
cldg_plus_ano = {'DBLP': 86.41, 'Bitcoinotc': 81.97, 'BITotc': 82.92,
                 'BITalpha': 79.71, 'TAX51': 81.02}

print('=== Bootstrap best-of-5 vs CLDG++ paper (B=10000 resamples) ===\n')
print('Classification:')
print(f'{"Dataset":<12} {"P(best5 > CLDG++)":<20} {"P95 best":<12} {"Observed best":<14}')
for ds in ['DBLP', 'Bitcoinotc', 'BITotc', 'BITalpha', 'TAX51']:
    arr = np.array(kairos[ds])
    best5 = np.max(np.random.choice(arr, size=(B, 5), replace=True), axis=1)
    p_win = np.mean(best5 > cldg_plus_clf[ds])
    p95   = np.percentile(best5, 95)
    print(f'{ds:<12} {p_win:.4f}              {p95:.2f}       {arr.max():.2f}')

print('\nAnomaly:')
print(f'{"Dataset":<12} {"P(best5 > CLDG++)":<20} {"P95 best":<12} {"Observed best":<14}')
for ds in ['DBLP', 'Bitcoinotc', 'BITotc', 'BITalpha', 'TAX51']:
    arr = np.array(kairos_ano[ds])
    best5 = np.max(np.random.choice(arr, size=(B, 5), replace=True), axis=1)
    p_win = np.mean(best5 > cldg_plus_ano[ds])
    p95   = np.percentile(best5, 95)
    print(f'{ds:<12} {p_win:.4f}              {p95:.2f}       {arr.max():.2f}')
