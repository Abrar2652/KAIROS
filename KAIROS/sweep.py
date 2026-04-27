"""
KAIROS — multi-seed / ablation sweep scheduler
==============================================
Assigns runs to free GPUs and launches them in parallel.
Writes one log file per run under /nas/home/jahin/KAIROS/runs/.

Usage:
  python3 sweep.py --grid multiseed_clf --gpus 2,4,5 --max_parallel 3

Grids:
  multiseed_clf        : 5 seeds × 6 datasets × classification
  multiseed_ano        : 5 seeds × 6 datasets × anomaly_detection
  ablate_components    : {no_ppr_view, fixed_tau, no_koop_anomaly} × all datasets
  ppr_alpha_sweep      : α ∈ {0.05, 0.10, 0.15, 0.20, 0.30} × Bitcoin family
  no_ppr_eval_all      : no_ppr_eval on DBLP + TAX51 (sanity — should match or hurt)
"""

import argparse
import os
import queue
import subprocess
import threading
import time

RUNS_DIR = '/nas/home/jahin/KAIROS/runs'
KAIROS_DIR = '/nas/home/jahin/KAIROS/KAIROS'


# ─── Per-dataset canonical configs (match run_experiments.py) ────────────
DATASETS = {
    'dblp':       dict(snapshots=4, views_clf=4, views_ano=4, strat_clf='sequential',
                       strat_ano='sequential', dl_clf=4096, dl_ano=4096,
                       ep_clf=200, ep_ano=200),
    'bitcoinotc': dict(snapshots=4, views_clf=3, views_ano=4, strat_clf='sequential',
                       strat_ano='sequential', dl_clf=64,   dl_ano=64,
                       ep_clf=25,  ep_ano=25),
    'bitotc':     dict(snapshots=4, views_clf=4, views_ano=4, strat_clf='random',
                       strat_ano='sequential', dl_clf=4096, dl_ano=4096,
                       ep_clf=50,  ep_ano=50),
    'bitalpha':   dict(snapshots_clf=6, snapshots_ano=5, views_clf=4, views_ano=5,
                       strat_clf='sequential', strat_ano='sequential',
                       dl_clf=4096, dl_ano=4096, ep_clf=200, ep_ano=100),
    'tax51':      dict(snapshots_clf=8, snapshots_ano=4, views_clf=5, views_ano=4,
                       strat_clf='random', strat_ano='sequential',
                       dl_clf=4096, dl_ano=4096, ep_clf=200, ep_ano=200),
    'reddit':     dict(snapshots_clf=5, snapshots_ano=4, views_clf=4, views_ano=4,
                       strat_clf='random', strat_ano='sequential',
                       dl_clf=4096, dl_ano=4096, ep_clf=200, ep_ano=200),
}

SEEDS = [24, 42, 7, 13, 99]


def _cfg(ds, task):
    d = DATASETS[ds]
    snaps_key = 'snapshots_' + ('clf' if task == 'classification' else 'ano')
    snaps = d.get(snaps_key, d.get('snapshots'))
    views = d['views_clf' if task == 'classification' else 'views_ano']
    strat = d['strat_clf' if task == 'classification' else 'strat_ano']
    dl    = d['dl_clf' if task == 'classification' else 'dl_ano']
    ep    = d['ep_clf' if task == 'classification' else 'ep_ano']
    return snaps, views, strat, dl, ep


def _job(ds, task, seed, ablation='none', alpha=0.15, tau_val=0.1,
         lambda_koop=1.0):
    snaps, views, strat, dl, ep = _cfg(ds, task)
    tag = f'{ds}_{task[:3]}_{ablation}_a{alpha}_lk{lambda_koop}_s{seed}'
    log = os.path.join(RUNS_DIR, f'{tag}.log')
    cmd = [
        'python3', '-u', 'ablate.py',
        '--dataset', ds, '--task', task,
        '--snapshots', str(snaps), '--views', str(views),
        '--strategy', strat, '--dataloader_size', str(dl),
        '--GPU', '0', '--epochs', str(ep), '--seed', str(seed),
        '--alpha', str(alpha), '--lambda_koop', str(lambda_koop),
        '--ablation', ablation, '--tau_val', str(tau_val),
    ]
    return tag, log, cmd


def build_grid(name):
    jobs = []
    if name == 'multiseed_clf':
        for ds in DATASETS:
            for sd in SEEDS:
                if sd == 24:
                    continue  # seed 24 is the canonical; reuse existing run
                jobs.append(_job(ds, 'classification', sd))
    elif name == 'multiseed_ano':
        for ds in DATASETS:
            for sd in SEEDS:
                if sd == 24:
                    continue
                jobs.append(_job(ds, 'anomaly_detection', sd))
    elif name == 'ablate_components':
        for ds in DATASETS:
            # no_ppr_view for classification
            jobs.append(_job(ds, 'classification', 24, ablation='no_ppr_view'))
            # fixed_tau for both tasks
            jobs.append(_job(ds, 'classification', 24, ablation='fixed_tau', tau_val=0.1))
            # no_koop for anomaly
            jobs.append(_job(ds, 'anomaly_detection', 24, ablation='no_koop_anomaly'))
    elif name == 'no_ppr_eval_all':
        for ds in ['dblp', 'tax51']:
            jobs.append(_job(ds, 'classification', 24, ablation='no_ppr_eval'))
    elif name == 'ppr_alpha_sweep':
        for ds in ['bitcoinotc', 'bitotc', 'bitalpha']:
            for a in [0.05, 0.10, 0.20, 0.30]:  # skip 0.15 (canonical)
                jobs.append(_job(ds, 'classification', 24, alpha=a))
    elif name == 'no_ppr_eval_bitcoin_seeds':
        # Multi-seed no_ppr_eval on Bitcoin family (if H1 confirmed)
        for ds in ['bitcoinotc', 'bitotc', 'bitalpha']:
            for sd in SEEDS:
                if sd == 24:
                    continue
                jobs.append(_job(ds, 'classification', sd, ablation='no_ppr_eval'))
    elif name == 'bitcoin_canonical_seeds':
        # Multi-seed CANONICAL (no ablation) on Bitcoin family for variance
        for ds in ['bitcoinotc', 'bitotc', 'bitalpha']:
            for sd in SEEDS:
                if sd == 24:
                    continue
                jobs.append(_job(ds, 'classification', sd))
    else:
        raise SystemExit(f'Unknown grid: {name}')
    return jobs


def run_pool(jobs, gpus, max_parallel):
    q = queue.Queue()
    for j in jobs:
        q.put(j)
    sem = threading.Semaphore(max_parallel)
    free_gpus = queue.Queue()
    for g in gpus:
        free_gpus.put(g)
    failures = []

    def _runner(tag, log, cmd):
        gpu = free_gpus.get()
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(gpu)
        print(f'  [launch] GPU{gpu}  {tag}')
        with open(log, 'w') as f:
            p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT,
                                 env=env, cwd=KAIROS_DIR)
            rc = p.wait()
        free_gpus.put(gpu)
        sem.release()
        if rc != 0:
            failures.append((tag, rc))
            print(f'  [fail ] GPU{gpu}  {tag}  rc={rc}')
        else:
            print(f'  [done ] GPU{gpu}  {tag}')

    threads = []
    while not q.empty():
        sem.acquire()
        tag, log, cmd = q.get()
        t = threading.Thread(target=_runner, args=(tag, log, cmd), daemon=False)
        t.start()
        threads.append(t)
    for t in threads:
        t.join()
    return failures


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--grid', required=True)
    p.add_argument('--gpus', default='2,4,5', help='comma-separated GPU IDs')
    p.add_argument('--max_parallel', type=int, default=3)
    a = p.parse_args()

    gpus = [int(x) for x in a.gpus.split(',')]
    jobs = build_grid(a.grid)
    print(f'Grid: {a.grid}  |  Jobs: {len(jobs)}  |  GPUs: {gpus}')
    for tag, log, _ in jobs:
        print(f'  - {tag}  → {os.path.basename(log)}')
    print()

    t0 = time.time()
    failures = run_pool(jobs, gpus, a.max_parallel)
    dur = time.time() - t0

    print(f'\nDone. {len(jobs)-len(failures)}/{len(jobs)} succeeded '
          f'in {dur/60:.1f} min.')
    for tag, rc in failures:
        print(f'  FAIL  {tag}  rc={rc}')


if __name__ == '__main__':
    main()
