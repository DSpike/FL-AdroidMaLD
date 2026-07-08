"""
K-sensitivity rerun with 3-term fitness function (div+cov+divpair).
gamma=0, delta=0. Only AOA, GWO, PSO need rerunning.
FedAvg-all, Random, Oort, PoCo results reused from original k-sensitivity.

Results: results/ksens_3term/k{K}/{method}/seed{N}/
"""

import subprocess, sys, os, csv
import numpy as np
from scipy import stats
from scipy.stats import norm

BASE         = r"C:\Users\Dspike\Documents\FL-AdroidMaLD"
SCRIPT       = os.path.join(BASE, 'run_experiment_20c_dir01.py')
RESULTS_BASE = os.path.join(BASE, 'results', 'ksens_3term')
OLD_KSENS    = os.path.join(BASE, 'results', 'k_sensitivity')
K_VALUES     = [4, 6, 8, 10, 12]
METHODS_NEW  = ['aoa8', 'gwo8', 'pso8']
SEEDS        = [42, 123, 456, 789, 2024]
MAX_ROUNDS   = 50
THR_82       = 0.82
WEIGHTS      = dict(alpha=0.3, beta=0.3, gamma=0.0, delta=0.0, lam=0.2)

total = len(K_VALUES) * len(METHODS_NEW) * len(SEEDS)
done, failed = 0, []

for k in K_VALUES:
    for method in METHODS_NEW:
        for seed in SEEDS:
            done += 1
            rdir     = os.path.join(RESULTS_BASE, f'k{k}', method, f'seed{seed}')
            csv_path = os.path.join(rdir, f'{method}_seed{seed}_rounds.csv')
            if os.path.exists(csv_path):
                print(f'[{done}/{total}]  k={k}  {method}  seed={seed}  (cached)')
                continue
            print(f'\n{"="*55}')
            print(f' [{done}/{total}]  k={k}  {method}  seed={seed}')
            print(f'{"="*55}')
            result = subprocess.run(
                [sys.executable, SCRIPT,
                 '--method',      method,
                 '--seed',        str(seed),
                 '--k_select',    str(k),
                 '--alpha',       str(WEIGHTS['alpha']),
                 '--beta',        str(WEIGHTS['beta']),
                 '--gamma',       str(WEIGHTS['gamma']),
                 '--delta',       str(WEIGHTS['delta']),
                 '--lam',         str(WEIGHTS['lam']),
                 '--results_dir', rdir],
                cwd=BASE,
            )
            if result.returncode != 0:
                failed.append((k, method, seed))

# ── Summary ────────────────────────────────────────────────────────────────────
def load(base, k, method, seeds):
    f1s, rds = [], []
    for s in seeds:
        p = os.path.join(base, f'k{k}', method, f'seed{s}', f'{method}_seed{s}_rounds.csv')
        if not os.path.exists(p): continue
        best, hit = 0.0, None
        with open(p) as f:
            for row in csv.DictReader(f):
                v = float(row['f1_macro'])
                if v > best: best = v
                if row['phase']=='phase2' and v>=THR_82 and hit is None:
                    hit = int(row['round'])
        f1s.append(best*100); rds.append(hit if hit else MAX_ROUNDS)
    return np.array(f1s), np.array(rds)

def sig(p): return '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else 'n.s.'

ALL_METHODS = ['fedavg_all','random8','oort8','poco8','gwo8','pso8','aoa8']

print(f'\n{"="*90}')
print(f' K-SENSITIVITY (3-term: div+cov+divpair), 5 seeds, Dir(0.1)')
print(f'{"="*90}')

from math import comb
for k in K_VALUES:
    print(f'\n--- k={k}  (C(20,{k})={comb(20,k):,}) ---')
    # AOA is proposed
    aoa_f1, aoa_rd = load(RESULTS_BASE, k, 'aoa8', SEEDS)
    aoa_rm = aoa_rd[aoa_rd<MAX_ROUNDS].mean() if (aoa_rd<MAX_ROUNDS).any() else MAX_ROUNDS
    print(f'  {"Method":12s}  {"PeakF1":>8}  {"Rds82%":>7}  {"Never":>6}  {"p(vs AOA)":>10}')
    print(f'  {"aoa8":12s}  {aoa_f1.mean():7.2f}%  {aoa_rm:6.1f}   {int((aoa_rd==MAX_ROUNDS).sum()):3d}/{len(aoa_f1)}   {"---":>10}  <- PROPOSED')
    for method in ['gwo8','pso8','fedavg_all','random8','oort8','poco8']:
        base = RESULTS_BASE if method in METHODS_NEW else OLD_KSENS
        f1, rd = load(base, k, method, SEEDS)
        if len(rd)==0: continue
        nev = int((rd==MAX_ROUNDS).sum())
        rm  = rd[rd<MAX_ROUNDS].mean() if (rd<MAX_ROUNDS).any() else MAX_ROUNDS
        try:
            stat, p = stats.wilcoxon(aoa_rd, rd, alternative='less')
            r = 1 - 2*stat/(len(aoa_rd)*(len(aoa_rd)+1)/2)
        except: p, r = 1.0, 0.0
        print(f'  {method:12s}  {f1.mean():7.2f}%  {rm:6.1f}   {nev:3d}/{len(f1)}   {p:8.4f}{sig(p):3s}')

if failed:
    print(f'\nFailed: {failed}')
else:
    print(f'\nAll {total} k-sensitivity runs completed.')
