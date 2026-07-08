"""
Run greedy-divergence baseline: top-k on composite linear score (λ=0, no metaheuristic).

This tests the null hypothesis that greedy selection on the per-client scores is
sufficient — i.e., whether the metaheuristic search over the non-linear fitness
landscape adds any value beyond deterministic top-k.

Same 15 seeds and weights as sota_3term (alpha=0.3, beta=0.3, gamma=0.0, delta=0.0).
Results: results/greedy_baseline/greedy8/seed{N}/
"""

import subprocess, sys, os, csv
import numpy as np
from scipy.stats import wilcoxon
from itertools import combinations

BASE         = r"C:\Users\Dspike\Documents\FL-AdroidMaLD"
SCRIPT       = os.path.join(BASE, 'run_experiment_20c_dir01.py')
RESULTS_BASE = os.path.join(BASE, 'results', 'greedy_baseline')
AO_BASE      = os.path.join(BASE, 'results', 'sota_3term', 'aoa8')
SEEDS        = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 42, 123, 456, 789, 2024]
MAX_ROUNDS   = 50
THR_82       = 0.82

WEIGHTS = dict(alpha=0.3, beta=0.3, gamma=0.0, delta=0.0, lam=0.0)

# ── Run greedy8 seeds ──────────────────────────────────────────────────────────
total = len(SEEDS)
done, failed = 0, []

for seed in SEEDS:
    done += 1
    rdir     = os.path.join(RESULTS_BASE, 'greedy8', f'seed{seed}')
    csv_path = os.path.join(rdir, f'greedy8_seed{seed}_rounds.csv')
    if os.path.exists(csv_path):
        print(f'[{done}/{total}]  greedy8  seed={seed}  (cached)')
        continue
    print(f'\n{"="*60}')
    print(f' [{done}/{total}]  greedy8  seed={seed}')
    print(f'{"="*60}')
    result = subprocess.run(
        [sys.executable, SCRIPT,
         '--method',      'greedy8',
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
        failed.append(seed)

if failed:
    print(f'\nFailed seeds: {failed}')

# ── Analyse results ────────────────────────────────────────────────────────────
def load_results(base, method, seeds, thr=THR_82):
    f1s, rds, fails = [], [], 0
    for s in seeds:
        p = os.path.join(base, method, f'seed{s}', f'{method}_seed{s}_rounds.csv')
        if not os.path.exists(p):
            continue
        best_f1, hit = 0.0, None
        with open(p) as f:
            for row in csv.DictReader(f):
                v = float(row['f1_macro'])
                if v > best_f1:
                    best_f1 = v
                if row['phase'] == 'phase2' and v >= thr and hit is None:
                    hit = int(row['round'])
        f1s.append(best_f1 * 100)
        if hit is None:
            fails += 1
            rds.append(MAX_ROUNDS)
        else:
            rds.append(hit)
    return np.array(f1s), np.array(rds), fails

print('\n' + '='*60)
print(' RESULTS SUMMARY')
print('='*60)

greedy_f1, greedy_rds, greedy_fails = load_results(RESULTS_BASE, 'greedy8', SEEDS)
ao_f1,     ao_rds,     ao_fails     = load_results(AO_BASE, '.',     SEEDS)

# Reload AO directly from sota_3term
ao_f1_list, ao_rds_list = [], []
for s in SEEDS:
    p = os.path.join(AO_BASE, f'seed{s}', f'aoa8_seed{s}_rounds.csv')
    if not os.path.exists(p):
        continue
    best_f1, hit = 0.0, None
    with open(p) as f:
        for row in csv.DictReader(f):
            v = float(row['f1_macro'])
            if v > best_f1:
                best_f1 = v
            if row['phase'] == 'phase2' and v >= THR_82 and hit is None:
                hit = int(row['round'])
    ao_f1_list.append(best_f1 * 100)
    ao_rds_list.append(hit if hit else MAX_ROUNDS)
ao_f1  = np.array(ao_f1_list)
ao_rds = np.array(ao_rds_list)

print(f"\nGreedy-8 : F1={greedy_f1.mean():.2f}±{greedy_f1.std():.2f}%  "
      f"Rds={greedy_rds[greedy_rds<MAX_ROUNDS].mean():.1f}  Fails={greedy_fails}/15")
print(f"AO-8     : F1={ao_f1.mean():.2f}±{ao_f1.std():.2f}%  "
      f"Rds={ao_rds[ao_rds<MAX_ROUNDS].mean():.1f}  Fails={ao_fails}/15")

# Paired Wilcoxon on rounds-to-82% (converged seeds only)
min_len = min(len(greedy_rds), len(ao_rds))
if min_len >= 5:
    try:
        stat, p = wilcoxon(ao_rds[:min_len], greedy_rds[:min_len], alternative='less')
        print(f"\nWilcoxon (AO < Greedy, rounds-to-82%): p={p:.4f}")
    except Exception as e:
        print(f"Wilcoxon failed: {e}")

print('\nDone.')
