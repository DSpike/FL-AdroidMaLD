"""
Run proper Oort baseline (training-loss utility) across 15 seeds.
Compares against existing AO results from results/sota_3term/aoa8/.

Usage: python run_oort_proper.py
"""

import subprocess, sys, os, csv
import numpy as np
from scipy.stats import wilcoxon

BASE         = r"C:\Users\Dspike\Documents\FL-AdroidMaLD"
SCRIPT       = os.path.join(BASE, 'run_experiment_20c_dir01.py')
RESULTS_BASE = os.path.join(BASE, 'results', 'oort_proper')
AO_BASE      = os.path.join(BASE, 'results', 'sota_3term', 'aoa8')
SEEDS        = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 42, 123, 456, 789, 2024]
MAX_ROUNDS   = 50
THR_82       = 0.82

total = len(SEEDS)
done, failed = 0, []

for seed in SEEDS:
    done += 1
    rdir     = os.path.join(RESULTS_BASE, f'seed{seed}')
    csv_path = os.path.join(rdir, f'oort8_seed{seed}_rounds.csv')
    if os.path.exists(csv_path):
        print(f'[{done}/{total}]  oort8_proper  seed={seed}  (cached)')
        continue
    print(f'\n{"="*60}')
    print(f' [{done}/{total}]  oort8_proper  seed={seed}')
    print(f'{"="*60}')
    result = subprocess.run(
        [sys.executable, SCRIPT,
         '--method',      'oort8',
         '--seed',        str(seed),
         '--k_select',    '8',
         '--results_dir', rdir],
        cwd=BASE,
    )
    if result.returncode != 0:
        print(f'  *** FAILED (exit {result.returncode}) ***')
        failed.append(seed)

if failed:
    print(f'\nFailed seeds: {failed}')

# ── Analysis ──────────────────────────────────────────────────────────────────
def load_results(base, seeds, prefix, thr=THR_82):
    f1s, rds, fails = [], [], 0
    for s in seeds:
        p = os.path.join(base, f'seed{s}', f'{prefix}_seed{s}_rounds.csv')
        if not os.path.exists(p):
            print(f'  [MISSING] {p}')
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

oort_f1, oort_rds, oort_fails = load_results(RESULTS_BASE, SEEDS, 'oort8')

# Load AO results
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

converged_oort = oort_rds[oort_rds < MAX_ROUNDS]
converged_ao   = ao_rds[ao_rds < MAX_ROUNDS]

print(f"\nOort (proper): F1={oort_f1.mean():.2f}±{oort_f1.std():.2f}%  "
      f"Rds={converged_oort.mean():.1f if len(converged_oort) else float('nan'):.1f}  "
      f"Fails={oort_fails}/15")
print(f"AO (ours)    : F1={ao_f1.mean():.2f}±{ao_f1.std():.2f}%  "
      f"Rds={converged_ao.mean():.1f}  Fails={15-len(ao_f1_list)}/15")

min_len = min(len(oort_rds), len(ao_rds))
if min_len >= 5:
    try:
        stat, p = wilcoxon(ao_rds[:min_len], oort_rds[:min_len], alternative='less')
        r = 1 - 2 * stat / (min_len * (min_len + 1))
        print(f"\nWilcoxon (AO < Oort_proper, rounds-to-82%): stat={stat:.1f}  p={p:.4f}  r={r:.3f}")
    except Exception as e:
        print(f"Wilcoxon: {e}")

print('\nDone.')
