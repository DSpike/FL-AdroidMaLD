"""
Oort sanity check: verify the Oort implementation is correct by running on IID data.

Hypothesis: if Oort's failure at Dir(0.1) is a regime mismatch (not a code bug),
it should perform competitively with AO under IID distribution where its utility
function assumptions hold (near-uniform participation, stable loss estimates).

Runs 5 seeds each of: Oort, AO, Random, FedAvg-all — all on IID distribution.
Expected result: Oort converges reliably and reaches similar F1 to other methods.

Results: results/oort_sanity/{method}/seed{N}/
"""

import subprocess, sys, os, csv
import numpy as np

BASE         = r"C:\Users\Dspike\Documents\FL-AdroidMaLD"
SCRIPT       = os.path.join(BASE, 'run_experiment_20c_dir01.py')
RESULTS_BASE = os.path.join(BASE, 'results', 'oort_sanity')
SEEDS        = [1, 2, 3, 4, 5]
MAX_ROUNDS   = 50
THR_82       = 0.82
METHODS      = ['oort8', 'aoa8', 'random8', 'fedavg_all']

# ── Run all methods ────────────────────────────────────────────────────────────
total_runs = len(METHODS) * len(SEEDS)
done = 0

for method in METHODS:
    for seed in SEEDS:
        done += 1
        rdir     = os.path.join(RESULTS_BASE, method, f'seed{seed}')
        csv_name = f'{method}_seed{seed}_rounds.csv'
        csv_path = os.path.join(rdir, csv_name)

        if os.path.exists(csv_path):
            print(f'[{done}/{total_runs}]  {method}  seed={seed}  (cached)')
            continue

        print(f'\n{"="*60}')
        print(f' [{done}/{total_runs}]  {method}  seed={seed}  [IID]')
        print(f'{"="*60}')

        result = subprocess.run(
            [sys.executable, SCRIPT,
             '--method',      method,
             '--seed',        str(seed),
             '--k_select',    '8',
             '--distribution', 'IID',
             '--results_dir', rdir],
            cwd=BASE,
        )
        if result.returncode != 0:
            print(f'  *** FAILED (exit {result.returncode}) ***')

# ── Analyse results ────────────────────────────────────────────────────────────
def load_results(base, method, seeds, thr=THR_82):
    f1s, rds, fails = [], [], 0
    for s in seeds:
        p = os.path.join(base, method, f'seed{s}', f'{method}_seed{s}_rounds.csv')
        if not os.path.exists(p):
            print(f'  [warn] missing: {p}')
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
print(' OORT SANITY CHECK — IID DISTRIBUTION (5 seeds)')
print('='*60)
print(f'{"Method":<14}  {"Peak F1":>10}  {"Rds→82%":>8}  {"Fails":>6}')
print('-'*46)

for method in METHODS:
    f1s, rds, fails = load_results(RESULTS_BASE, method, SEEDS)
    if len(f1s) == 0:
        print(f'{method:<14}  {"no data":>10}')
        continue
    converged_rds = rds[rds < MAX_ROUNDS]
    rds_str = f'{converged_rds.mean():.1f}' if len(converged_rds) else 'N/A'
    print(f'{method:<14}  {f1s.mean():.2f}±{f1s.std():.2f}%  {rds_str:>8}  {fails}/{len(SEEDS)}')

print('='*60)
print('\nInterpretation:')
print('  If Oort converges reliably on IID with F1 near AO/Random,')
print('  its Dir(0.1) failure is regime mismatch, NOT misconfiguration.')
print('  If Oort also fails on IID, the implementation should be checked.')
