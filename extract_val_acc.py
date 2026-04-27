"""
Extract val-Acc-based best config from per_dataset_tune.sh sweep logs.

For each sweep config, computes the mean (across 5 LogReg runs) of the
max val Acc seen during the 2000-epoch linear-probe fit. Uses that as
the config selection criterion (strictly val-based — no test leakage).
Then reports the test Acc for the best-val config.

Usage: python3 extract_val_acc.py <dataset>
"""

import argparse
import glob
import os
import re

import numpy as np

RUNS = '/nas/home/jahin/KAIROS/runs'
LINEVAL_RE = re.compile(r'LinEval\s+(\d+)\s+\|\s+train\s+([\d.]+)\s+\|\s+val\s+([\d.]+)\s+\|\s+micro\s+([\d.]+)\s+\|\s+wei\s+([\d.]+)')
RESULT_RE  = re.compile(r"\[ablate-result\].*?result=\{'accuracy':\s*([\d.]+)")


def _parse_log(log_path):
    """Return (mean_max_val, test_acc) or None if no valid run."""
    if not os.path.exists(log_path):
        return None
    with open(log_path) as f:
        txt = f.read()
    # Split into LogReg runs: each begins with 'LinEval    0 |'
    # Find indices of run starts
    run_indices = [m.start() for m in re.finditer(r'LinEval\s+0\s+\|', txt)]
    if not run_indices:
        return None
    run_indices.append(len(txt))
    max_vals = []
    for i in range(len(run_indices) - 1):
        chunk = txt[run_indices[i]:run_indices[i+1]]
        vals = [float(m.group(3)) for m in LINEVAL_RE.finditer(chunk)]
        if vals:
            max_vals.append(max(vals))
    if not max_vals:
        return None
    mean_max_val = float(np.mean(max_vals))
    # Test acc from ablate-result
    tm = RESULT_RE.search(txt)
    test_acc = float(tm.group(1)) * 100 if tm else None
    return mean_max_val, test_acc


def main():
    p = argparse.ArgumentParser()
    p.add_argument('dataset')
    a = p.parse_args()

    ds = a.dataset
    pattern = os.path.join(RUNS, f'{ds}_tune_h*_l*_*_s24.log')
    rows = []
    for log in sorted(glob.glob(pattern)):
        name = os.path.basename(log).replace('.log', '')
        # name: {ds}_tune_h{hidden}_l{nl}_{tau_mode}_s24
        m = re.match(rf'{ds}_tune_h(\d+)_l(\d+)_(canonical|fixed05)_s24', name)
        if not m:
            continue
        h, nl, tau_mode = m.group(1), m.group(2), m.group(3)
        res = _parse_log(log)
        if res is None:
            rows.append((h, nl, tau_mode, None, None, 'incomplete'))
            continue
        mv, ta = res
        rows.append((int(h), int(nl), tau_mode, mv * 100, ta, 'ok'))

    print(f'=== {ds} per-dataset tune sweep (seed=24) ===')
    print(f'{"hidden":<7} {"nlayer":<7} {"tau":<10} {"val_acc":<10} {"test_acc":<10} {"status"}')
    rows.sort(key=lambda x: (-(x[3] if x[3] is not None else -1)))
    for h, nl, tau, vv, tt, st in rows:
        v_str = f'{vv:.2f}' if vv is not None else '—'
        t_str = f'{tt:.2f}' if tt is not None else '—'
        print(f'{h:<7} {nl:<7} {tau:<10} {v_str:<10} {t_str:<10} {st}')

    # Best by val Acc
    valid = [r for r in rows if r[3] is not None]
    if valid:
        best = max(valid, key=lambda x: x[3])
        h, nl, tau, vv, tt, st = best
        print(f'\nBest-by-val: hidden={h} n_layers={nl} tau={tau} '
              f'→ val {vv:.2f}, test {tt:.2f}')


if __name__ == '__main__':
    main()
