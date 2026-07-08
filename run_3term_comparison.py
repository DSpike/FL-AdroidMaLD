"""
Rerun main SOTA comparison with 3-term fitness function.

3-term function: div(S) + cov(S) + divpair(S)
  alpha=0.3  (divergence)    -- KEPT
  beta=0.3   (class coverage) -- KEPT
  gamma=0.0  (volume)         -- REMOVED (ablation showed no contribution)
  delta=0.0  (local f1val)    -- REMOVED (ablation showed no contribution)
  lam=0.2    (pairwise div)   -- KEPT

Only AOA, GWO, PSO need rerunning (they use the fitness function).
FedAvg-all, Random, Oort, PoCo results are unchanged.

Results: results/sota_3term/{method}/seed{N}/
"""

import subprocess, sys, os, csv
import numpy as np
from scipy import stats
from scipy.stats import norm

BASE         = r"C:\Users\Dspike\Documents\FL-AdroidMaLD"
SCRIPT       = os.path.join(BASE, 'run_experiment_20c_dir01.py')
RESULTS_BASE = os.path.join(BASE, 'results', 'sota_3term')
SEEDS        = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 42, 123, 456, 789, 2024]
MAX_ROUNDS   = 50
THR_82       = 0.82
METHODS      = ['aoa8', 'gwo8', 'pso8']

WEIGHTS = dict(alpha=0.3, beta=0.3, gamma=0.0, delta=0.0, lam=0.2)

total = len(METHODS) * len(SEEDS)
done, failed = 0, []

for method in METHODS:
    for seed in SEEDS:
        done += 1
        rdir     = os.path.join(RESULTS_BASE, method, f'seed{seed}')
        csv_path = os.path.join(rdir, f'{method}_seed{seed}_rounds.csv')
        if os.path.exists(csv_path):
            print(f'[{done}/{total}]  {method}  seed={seed}  (cached)')
            continue
        print(f'\n{"="*60}')
        print(f' [{done}/{total}]  {method}  seed={seed}  (3-term: div+cov+divpair)')
        print(f'{"="*60}')
        result = subprocess.run(
            [sys.executable, SCRIPT,
             '--method',      method,
             '--seed',        str(seed),
             '--k_select',    '8',
             '--alpha',       str(WEIGHTS['alpha']),
             '--beta',        str(WEIGHTS['beta']),
             '--gamma',       str(WEIGHTS['gamma']),
             '--delta',       str(WEIGHTS['delta']),
             '--lam',         str(WEIGHTS['lam']),
             '--results_dir', rdir],
            cwd=BASE,
        )
        if result.returncode != 0:
            print(f'  *** FAILED ***')
            failed.append((method, seed))

# ── Summary ────────────────────────────────────────────────────────────────────
SOTA_OLD = os.path.join(BASE, 'results', 'sota_comparison')

def load(base, method, seeds):
    f1s, rds = [], []
    for s in seeds:
        p = os.path.join(base, method, f'seed{s}', f'{method}_seed{s}_rounds.csv')
        if not os.path.exists(p): continue
        best, hit = 0.0, None
        with open(p) as f:
            for row in csv.DictReader(f):
                v = float(row['f1_macro'])
                if v > best: best = v
                if row['phase'] == 'phase2' and v >= THR_82 and hit is None:
                    hit = int(row['round'])
        f1s.append(best * 100)
        rds.append(hit if hit else MAX_ROUNDS)
    return np.array(f1s), np.array(rds)

def sig(p):
    return '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else 'n.s.'

print(f'\n{"="*80}')
print(f' 3-TERM COMPARISON RESULTS (div + cov + divpair, k=8, Dir(0.1), n=15)')
print(f'{"="*80}')

# Load unchanged baselines from old sota_comparison
BASELINES = ['fedavg_all', 'random8', 'oort8', 'poco8']
aoa_f1, aoa_rd = load(RESULTS_BASE, 'aoa8', SEEDS)
aoa_rm  = aoa_rd[aoa_rd < MAX_ROUNDS].mean() if (aoa_rd < MAX_ROUNDS).any() else MAX_ROUNDS

print(f'\n  {"Method":12s}  {"PeakF1":>8}  {"Rds82%":>7}  {"Never":>6}  {"p(vs AOA)":>10}  {"r":>6}  {"d":>6}')
print(f'  {"-"*12}  {"-"*8}  {"-"*7}  {"-"*6}  {"-"*10}  {"-"*6}  {"-"*6}')

for method in ['aoa8', 'gwo8', 'pso8']:
    f1, rd = load(RESULTS_BASE, method, SEEDS)
    nev = int((rd == MAX_ROUNDS).sum())
    rm  = rd[rd < MAX_ROUNDS].mean() if (rd < MAX_ROUNDS).any() else MAX_ROUNDS
    if method == 'aoa8':
        print(f'  {"aoa8":12s}  {f1.mean():7.2f}%  {rm:6.1f}   {nev:3d}/15   {"---":>10}  {"---":>6}  {"---":>6}  <- PROPOSED')
        continue
    try:
        stat, p = stats.wilcoxon(aoa_rd, rd, alternative='less')
        r = 1 - 2*stat/(len(aoa_rd)*(len(aoa_rd)+1)/2)
        z = norm.ppf(1-p) if p < 1 else 0.0
        d = z / np.sqrt(len(aoa_rd))
    except: p, r, d = 1.0, 0.0, 0.0
    print(f'  {method:12s}  {f1.mean():7.2f}%  {rm:6.1f}   {nev:3d}/15   {p:8.4f}{sig(p):3s}  {r:6.3f}  {d:6.3f}')

for method in BASELINES:
    f1, rd = load(SOTA_OLD, method, SEEDS)
    if len(rd) == 0: continue
    nev = int((rd == MAX_ROUNDS).sum())
    rm  = rd[rd < MAX_ROUNDS].mean() if (rd < MAX_ROUNDS).any() else MAX_ROUNDS
    try:
        stat, p = stats.wilcoxon(aoa_rd, rd, alternative='less')
        r = 1 - 2*stat/(len(aoa_rd)*(len(aoa_rd)+1)/2)
        z = norm.ppf(1-p) if p < 1 else 0.0
        d = z / np.sqrt(len(aoa_rd))
    except: p, r, d = 1.0, 0.0, 0.0
    print(f'  {method:12s}  {f1.mean():7.2f}%  {rm:6.1f}   {nev:3d}/15   {p:8.4f}{sig(p):3s}  {r:6.3f}  {d:6.3f}')

if failed:
    print(f'\nFailed: {failed}')
else:
    print(f'\nAll {total} runs completed.')
