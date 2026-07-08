"""
Non-IID sensitivity sweep: Dir(0.05), Dir(0.1), Dir(0.5) × all methods, 5 seeds.

Evaluates how well each method performs across three levels of data heterogeneity:
  Dir_0.05  — extreme heterogeneity (harder than our primary setting)
  Dir_0.1   — severe (primary evaluation setting in the paper)
  Dir_0.5   — moderate (standard in FL literature, easier)

5 seeds per cell (independent from the 15-seed main comparison) for tractability.

Results land in: results/noniid_sensitivity/{distribution}/{method}/seed{N}/

Usage:
    python run_noniid_sensitivity.py
    python run_noniid_sensitivity.py --methods aoa8 oort8 poco8 divfl8
    python run_noniid_sensitivity.py --dry_run
"""

import argparse
import subprocess, sys, os, csv
import numpy as np
from scipy import stats
from scipy.stats import norm

parser = argparse.ArgumentParser()
parser.add_argument('--methods', nargs='+',
                    default=['aoa8', 'oort8', 'poco8', 'random8', 'gwo8', 'pso8', 'divfl8'],
                    help='Methods to evaluate (default: all seven)')
parser.add_argument('--distributions', nargs='+',
                    default=['Dir_0.05', 'Dir_0.1', 'Dir_0.5'],
                    help='Distributions to sweep (default: Dir_0.05 Dir_0.1 Dir_0.5)')
parser.add_argument('--seeds', nargs='+', type=int,
                    default=[1, 2, 3, 42, 123],
                    help='Seeds per cell (default: 5)')
parser.add_argument('--dry_run', action='store_true',
                    help='Print commands without executing')
args = parser.parse_args()

BASE         = r"C:\Users\Dspike\Documents\FL-AdroidMaLD"
SCRIPT       = os.path.join(BASE, 'run_experiment_20c_dir01.py')
RESULTS_BASE = os.path.join(BASE, 'results', 'noniid_sensitivity')
MAX_ROUNDS   = 50
THR_82       = 0.82

METHODS       = args.methods
DISTRIBUTIONS = args.distributions
SEEDS         = args.seeds

# AOA fitness weights — same as the main comparison
WEIGHTS = dict(alpha=0.3, beta=0.3, gamma=0.0, delta=0.0, lam=0.2)

total = len(METHODS) * len(DISTRIBUTIONS) * len(SEEDS)
done, failed = 0, []

print(f"\n{'='*70}")
print(f" Non-IID Sensitivity Sweep")
print(f" Methods : {METHODS}")
print(f" Dists   : {DISTRIBUTIONS}")
print(f" Seeds   : {SEEDS}")
print(f" Total   : {total} runs")
print(f"{'='*70}\n")

for dist in DISTRIBUTIONS:
    for method in METHODS:
        for seed in SEEDS:
            done += 1
            rdir     = os.path.join(RESULTS_BASE, dist, method, f'seed{seed}')
            csv_path = os.path.join(rdir, f'{method}_seed{seed}_rounds.csv')

            if os.path.exists(csv_path):
                print(f'[{done}/{total}]  {dist}  {method}  seed={seed}  (cached)')
                continue

            print(f'\n{"="*60}')
            print(f' [{done}/{total}]  {dist}  {method}  seed={seed}')
            print(f'{"="*60}')

            cmd = [sys.executable, SCRIPT,
                   '--method',       method,
                   '--seed',         str(seed),
                   '--k_select',     '8',
                   '--distribution', dist,
                   '--alpha',        str(WEIGHTS['alpha']),
                   '--beta',         str(WEIGHTS['beta']),
                   '--gamma',        str(WEIGHTS['gamma']),
                   '--delta',        str(WEIGHTS['delta']),
                   '--lam',          str(WEIGHTS['lam']),
                   '--results_dir',  rdir]

            if args.dry_run:
                print('  DRY RUN:', ' '.join(cmd))
                continue

            result = subprocess.run(cmd, cwd=BASE)
            if result.returncode != 0:
                print(f'  *** FAILED ***')
                failed.append((dist, method, seed))


# ── Summary table ────────────────────────────────────────────────────────────

def load(base, dist, method, seeds):
    f1s, rds = [], []
    for s in seeds:
        p = os.path.join(base, dist, method, f'seed{s}',
                         f'{method}_seed{s}_rounds.csv')
        if not os.path.exists(p):
            continue
        best, hit = 0.0, None
        with open(p) as f:
            for row in csv.DictReader(f):
                v = float(row['f1_macro'])
                if v > best:
                    best = v
                if row['phase'] == 'phase2' and v >= THR_82 and hit is None:
                    hit = int(row['round'])
        f1s.append(best * 100)
        rds.append(hit if hit else MAX_ROUNDS)
    return np.array(f1s), np.array(rds)


print(f'\n\n{"="*90}')
print(f' NON-IID SENSITIVITY RESULTS — Peak F1-macro (mean ± std), Never/5')
print(f'{"="*90}')
print(f'  {"Method":12s}', end='')
for dist in DISTRIBUTIONS:
    print(f'  {dist:>26s}', end='')
print()
print('  ' + '-'*86)

for method in METHODS:
    print(f'  {method:12s}', end='')
    for dist in DISTRIBUTIONS:
        f1, rd = load(RESULTS_BASE, dist, method, SEEDS)
        if len(f1) == 0:
            print(f'  {"---":>26s}', end='')
        else:
            nev = int((rd == MAX_ROUNDS).sum())
            cell = f'{f1.mean():.1f}±{f1.std():.1f}%  ({nev}/{len(SEEDS)} fail)'
            print(f'  {cell:>26s}', end='')
    print()

print()

# Per-distribution: AOA vs DivFL significance test
if 'aoa8' in METHODS and 'divfl8' in METHODS:
    print(f'\n  AOA vs DivFL (one-sided Wilcoxon on rounds-to-82%, n={len(SEEDS)}):')
    for dist in DISTRIBUTIONS:
        _, aoa_rd   = load(RESULTS_BASE, dist, 'aoa8',   SEEDS)
        _, divfl_rd = load(RESULTS_BASE, dist, 'divfl8', SEEDS)
        if len(aoa_rd) == 0 or len(divfl_rd) == 0:
            print(f'    {dist}: no data')
            continue
        try:
            stat, p = stats.wilcoxon(aoa_rd, divfl_rd, alternative='less')
            r = 1 - 2*stat/(len(aoa_rd)*(len(aoa_rd)+1)/2)
            z = norm.ppf(1-p) if p < 1 else 0.0
            d = z / np.sqrt(len(aoa_rd))
            sig = '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else 'n.s.'
            print(f'    {dist}: p={p:.4f}{sig}  r={r:.3f}  d={d:.3f}')
        except Exception as e:
            print(f'    {dist}: test failed ({e})')

if failed:
    print(f'\nFailed runs ({len(failed)}): {failed}')
else:
    print(f'\nAll {total} runs completed or cached.')
