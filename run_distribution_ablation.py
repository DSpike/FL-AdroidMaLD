"""
Distribution heterogeneity ablation.
Tests AOA vs Random-8 vs FedAvg-all across all 5 distributions.

Hypothesis: AOA convergence speedup GROWS with non-IID severity —
  more heterogeneous data = random selection finds worse subsets
  = larger benefit from guided combinatorial search.

Setup: 20 clients, select 8, optimised AOA (λ=0.0, δ=0.3), 5 seeds.
Total: 5 distributions × 3 methods × 5 seeds = 75 experiments (~10 hrs).

Heterogeneity scores (from federated_data_distribution.py):
  IID          → 0.004
  Dir_1.0      → 0.058
  Dir_0.5      → 0.073
  Pathological → 0.094
  Dir_0.1      → 0.107  ← most severe
"""

import subprocess, sys, os, csv
import numpy as np
from scipy import stats

BASE         = r"C:\Users\Dspike\Documents\FL-AdroidMaLD"
SCRIPT       = os.path.join(BASE, 'run_experiment_20c_dir01.py')
RESULTS_BASE = os.path.join(BASE, 'results', 'dist_ablation')
SEEDS        = [42, 123, 456, 789, 2024]
METHODS      = ['fedavg_all', 'random8', 'aoa8']
MAX_ROUNDS   = 50
THR_82       = 0.82

DISTRIBUTIONS = ['IID', 'Dir_1.0', 'Dir_0.5', 'Dir_0.1', 'Pathological']
HETERO        = {'IID': 0.004, 'Dir_1.0': 0.058, 'Dir_0.5': 0.073,
                 'Dir_0.1': 0.107, 'Pathological': 0.094}
DIST_LABEL    = {'IID': 'iid', 'Dir_1.0': 'dir1.0', 'Dir_0.5': 'dir0.5',
                 'Dir_0.1': 'dir0.1', 'Pathological': 'patho'}

total = len(DISTRIBUTIONS) * len(METHODS) * len(SEEDS)
done, failed = 0, []

for dist in DISTRIBUTIONS:
    for method in METHODS:
        for seed in SEEDS:
            done += 1
            label = DIST_LABEL[dist]
            rdir  = os.path.join(RESULTS_BASE, label, method, f'seed{seed}')
            print(f"\n{'='*60}")
            print(f" [{done}/{total}]  {dist}  {method}  seed={seed}")
            print(f"{'='*60}")
            result = subprocess.run(
                [sys.executable, SCRIPT,
                 '--method',       method,
                 '--seed',         str(seed),
                 '--distribution', dist,
                 '--lam',          '0.0',
                 '--delta',        '0.3',
                 '--results_dir',  rdir],
                cwd=BASE,
            )
            if result.returncode != 0:
                print(f"  *** FAILED ***")
                failed.append((dist, method, seed))

# ── Summary table ─────────────────────────────────────────────────────────────
def load(dirpath, method, seeds):
    f1s, rds = [], []
    for s in seeds:
        p = os.path.join(dirpath, method, f'seed{s}',
                         f'{method}_seed{s}_rounds.csv')
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
    return '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'n.s.'

print(f"\n{'='*80}")
print(f" DISTRIBUTION ABLATION — 20c select 8, λ=0.0 δ=0.3, 5 seeds")
print(f"{'='*80}")
print(f"\n{'Distribution':14s}  {'Het':>5s}  {'FedAvg-all':>11s}  "
      f"{'Random-8':>10s}  {'AOA-8':>10s}  {'Speedup→82%':>12s}  {'p':>8s}")

for dist in DISTRIBUTIONS:
    label = DIST_LABEL[dist]
    d     = os.path.join(RESULTS_BASE, label)
    rows  = {m: load(d, m, SEEDS) for m in METHODS}

    if not all(len(rows[m][0]) > 0 for m in METHODS):
        print(f"{dist:14s}  {'incomplete':>5s}")
        continue

    fa_f1 = rows['fedavg_all'][0]
    rn_f1 = rows['random8'][0];   rn_r = rows['random8'][1]
    ao_f1 = rows['aoa8'][0];      ao_r = rows['aoa8'][1]

    try:
        _, p_rds = stats.wilcoxon(ao_r, rn_r, alternative='less')
    except Exception:
        p_rds = 1.0

    ao_rm = ao_r[ao_r < MAX_ROUNDS]
    rn_rm = rn_r[rn_r < MAX_ROUNDS]
    spd   = ((rn_rm.mean() - ao_rm.mean()) / rn_rm.mean() * 100
             if len(rn_rm) and len(ao_rm) else float('nan'))

    print(f"{dist:14s}  {HETERO[dist]:>5.3f}  "
          f"{fa_f1.mean():>9.2f}%  "
          f"{rn_f1.mean():>9.2f}%  "
          f"{ao_f1.mean():>9.2f}%  "
          f"{spd:>10.1f}%   "
          f"{p_rds:>6.4f}{sig(p_rds):3s}")

if failed:
    print(f"\nFailed: {failed}")
else:
    print(f"\nAll {total} distribution ablation runs completed.")
