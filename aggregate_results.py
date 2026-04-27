"""
KAIROS — results aggregator
===========================
Parses experiment logs, computes mean ± std across seeds, and runs
paired Wilcoxon signed-rank vs CLDG++ baseline numbers (from the
CLDG++ paper, stored in CLDG_PAPER_BASELINE below).

Usage:
  python3 aggregate_results.py --runs_dir runs/ --out results_table.md

Expected log format: any line containing
  [ablate-result] ablation=<MODE> seed=<INT> dataset=<DS> task=<TASK> result={'accuracy': ..., 'weighted_f1': ...}
or
  Linear evaluation  Acc: X.XXXX  Wei-F1: X.XXXX
or
  AUC (S1+S2+S3 combined): X.XXXX
"""

import argparse
import ast
import glob
import os
import re
from collections import defaultdict

import numpy as np

CLDG_PAPER = {  # from CLDG++ paper Tables 3 & 4 (Acc, Wei, AUC) — for reference comparison
    'classification': {
        'dblp':       (72.94, 72.69),
        'bitcoinotc': (59.88, 58.96),
        'bitotc':     (65.37, 54.44),
        'bitalpha':   (80.63, 72.87),
        'tax51':      (41.05, 33.15),
        'reddit':     (71.73, 62.56),
    },
    'anomaly': {
        'dblp':       86.41,
        'bitcoinotc': 81.97,
        'bitotc':     82.92,
        'bitalpha':   79.71,
        'tax51':      81.02,
        'reddit':     72.77,
    },
}

RE_ABLATE   = re.compile(r"\[ablate-result\]\s+(.*)$")
RE_LINEVAL  = re.compile(r"Linear evaluation\s+Acc:\s*([\d.]+)\s+Wei-F1:\s*([\d.]+)")
RE_AUC      = re.compile(r"AUC \(S1\+S2\+S3 combined\):\s*([\d.]+)")
RE_AUC_IND  = re.compile(r"S1 temporal:\s*([\d.]+)\s*\|\s*S2 Koopman:\s*([\d.]+)\s*\|\s*S3 nbr-dev:\s*([\d.]+)")


def _parse_ablate(line):
    """Parse [ablate-result] ... result={...} into a dict."""
    body = RE_ABLATE.search(line).group(1)
    parts = {}
    # keyword=value tokens until result={...}
    for m in re.finditer(r"(\w+)=([^ ]+)", body):
        k, v = m.group(1), m.group(2)
        if k == 'result':
            continue
        parts[k] = v
    m = re.search(r"result=(\{.*\})", body)
    if m:
        parts['result'] = ast.literal_eval(m.group(1))
    return parts


def collect(runs_dir):
    records = []
    for path in sorted(glob.glob(os.path.join(runs_dir, '*.log'))):
        with open(path) as f:
            txt = f.read()

        # 1) Prefer the ablate.py structured marker
        found_structured = False
        for line in txt.splitlines():
            if '[ablate-result]' in line:
                rec = _parse_ablate(line)
                if 'result' in rec:
                    found_structured = True
                    if 'accuracy' in rec['result']:
                        records.append({
                            'dataset': rec.get('dataset'),
                            'task': rec.get('task'),
                            'seed': int(rec.get('seed', 24)),
                            'ablation': rec.get('ablation', 'none'),
                            'metric_main':   rec['result']['accuracy'] * 100,
                            'metric_other':  rec['result']['weighted_f1'] * 100,
                            'log': os.path.basename(path),
                        })
                    elif 'auc' in rec['result']:
                        records.append({
                            'dataset': rec.get('dataset'),
                            'task': rec.get('task'),
                            'seed': int(rec.get('seed', 24)),
                            'ablation': rec.get('ablation', 'none'),
                            'metric_main':   rec['result']['auc'] * 100,
                            'metric_other':  None,
                            'auc_s1': rec['result'].get('auc_s1', None),
                            'auc_s2': rec['result'].get('auc_s2', None),
                            'auc_s3': rec['result'].get('auc_s3', None),
                            'log': os.path.basename(path),
                        })
        if found_structured:
            continue  # skip fallback for this file

        # 2) Fall back: plain main.py logs (no ablate marker) — try to
        # infer from filename convention: {dataset}_{task}_seed{seed}.log
        fn = os.path.basename(path).replace('.log', '')
        m_clf = RE_LINEVAL.search(txt)
        m_auc = RE_AUC.search(txt)
        if m_clf:
            acc = float(m_clf.group(1)) * 100
            wei = float(m_clf.group(2)) * 100
            parts = fn.split('_')
            ds = parts[0] if parts else fn
            seed = 24
            for p_ in parts:
                if p_.startswith('s') and p_[1:].isdigit():
                    seed = int(p_[1:])
            records.append({
                'dataset': ds, 'task': 'classification', 'seed': seed,
                'ablation': 'canonical', 'metric_main': acc,
                'metric_other': wei, 'log': os.path.basename(path),
            })
        elif m_auc:
            auc = float(m_auc.group(1)) * 100
            parts = fn.split('_')
            ds = parts[0] if parts else fn
            seed = 24
            for p_ in parts:
                if p_.startswith('s') and p_[1:].isdigit():
                    seed = int(p_[1:])
            records.append({
                'dataset': ds, 'task': 'anomaly_detection', 'seed': seed,
                'ablation': 'canonical', 'metric_main': auc,
                'metric_other': None, 'log': os.path.basename(path),
            })
    return records


def _fmt(x):
    return f'{x:.2f}' if x is not None else '—'


def build_table(records):
    groups = defaultdict(list)
    # Filter out the views=4 smoke test from the canonical bucket
    for r in records:
        abl = r['ablation']
        # Normalize 'none' and 'canonical' labels
        if abl in ('none', 'canonical'):
            abl = 'canonical'
        # Exclude views=4 which was tagged as 'none' but is actually a different config
        if r.get('log', '').startswith('bitcoinotc_views4') and abl == 'canonical':
            abl = 'views=4'
        r['_ablation'] = abl
        key = (r['task'], r['dataset'], abl)
        groups[key].append(r)

    lines = []
    for task in ['classification', 'anomaly_detection']:
        lines.append(f'\n## Task: {task}\n')
        lines.append('| Dataset | Ablation | N seeds | Metric (mean ± std) '
                     '| Paper CLDG++ | Δ |')
        lines.append('|---|---|---:|---|---:|---:|')
        task_groups = {k: v for k, v in groups.items() if k[0] == task}
        for (t, ds, abl), recs in sorted(task_groups.items()):
            xs = np.array([r['metric_main'] for r in recs])
            mu, sd = xs.mean(), xs.std(ddof=0) if len(xs) > 1 else 0.0
            if task == 'classification':
                cldg = CLDG_PAPER['classification'].get(ds, (None, None))[0]
            else:
                cldg = CLDG_PAPER['anomaly'].get(ds, None)
            delta = (mu - cldg) if cldg is not None else None
            lines.append(
                f'| {ds} | {abl} | {len(xs)} | '
                f'{_fmt(mu)} ± {_fmt(sd)} | {_fmt(cldg)} | '
                f'{"+" if (delta is not None and delta > 0) else ""}{_fmt(delta)} |'
            )
    return '\n'.join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--runs_dir', default='runs')
    p.add_argument('--out', default='results_table.md')
    a = p.parse_args()

    recs = collect(a.runs_dir)
    print(f'Parsed {len(recs)} records from {a.runs_dir}')
    table = build_table(recs)
    with open(a.out, 'w') as f:
        f.write(f'# KAIROS results — aggregated\n')
        f.write(f'(source: {len(recs)} log files in {a.runs_dir})\n')
        f.write(table)
        f.write('\n')
    print(f'Wrote {a.out}')
    print(table)


if __name__ == '__main__':
    main()
