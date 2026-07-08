"""
Re-run Oort k-sensitivity with proper training-loss utility.
Overwrites existing cached k-sensitivity Oort results.
5 seeds × 5 k-values = 25 runs.
"""
import subprocess, sys, os, csv
import numpy as np
from scipy.stats import wilcoxon, norm

BASE         = r"C:\Users\Dspike\Documents\FL-AdroidMaLD"
SCRIPT       = os.path.join(BASE, 'run_experiment_20c_dir01.py')
KSENS_BASE   = os.path.join(BASE, 'results', 'k_sensitivity')
K_VALUES     = [4, 6, 8, 10, 12]
SEEDS        = [42, 123, 456, 789, 2024]
MAX_ROUNDS   = 50
THR_82       = 0.82

total = len(K_VALUES) * len(SEEDS)
done  = 0

for k in K_VALUES:
    for seed in SEEDS:
        done += 1
        rdir     = os.path.join(KSENS_BASE, f'k{k}', 'oort8', f'seed{seed}')
        csv_path = os.path.join(rdir, f'oort8_seed{seed}_rounds.csv')
        print(f"\n{'='*60}")
        print(f" [{done}/{total}]  k={k}  oort8_proper  seed={seed}")
        print(f"{'='*60}")
        os.makedirs(rdir, exist_ok=True)
        result = subprocess.run(
            [sys.executable, SCRIPT,
             '--method',      'oort8',
             '--seed',        str(seed),
             '--k_select',    str(k),
             '--results_dir', rdir],
            cwd=BASE,
        )
        if result.returncode != 0:
            print(f"  *** FAILED (exit {result.returncode}) ***")

# ── Analysis ──────────────────────────────────────────────────────────────────
print('\n' + '='*60)
print(' K-SENSITIVITY OORT (proper) SUMMARY')
print('='*60)

def load_k(k, method, seeds):
    rds, fails = [], 0
    for s in seeds:
        p = os.path.join(KSENS_BASE, f'k{k}', method, f'seed{s}', f'{method}_seed{s}_rounds.csv')
        if not os.path.exists(p):
            print(f'  MISSING: {p}')
            fails += 1
            rds.append(MAX_ROUNDS)
            continue
        hit = None
        with open(p) as f:
            for row in csv.DictReader(f):
                v = float(row['f1_macro'])
                if row['phase'] == 'phase2' and v >= THR_82 and hit is None:
                    hit = int(row['round'])
        if hit is None:
            fails += 1
            rds.append(MAX_ROUNDS)
        else:
            rds.append(hit)
    return np.array(rds), fails

print(f"\n{'k':>3} | {'Oort rds':>10} | {'Oort fails':>10} | {'AO rds':>8} | {'AO fails':>8} | p")
print("-" * 65)
for k in K_VALUES:
    oort_rds, oort_fails = load_k(k, 'oort8', SEEDS)
    ao_rds,   ao_fails   = load_k(k, 'aoa8',  SEEDS)
    n = min(len(oort_rds), len(ao_rds))
    try:
        stat, p = wilcoxon(ao_rds[:n], oort_rds[:n], alternative='less')
        sig = "*" if p <= 0.031 else ""
    except Exception:
        p, sig = float('nan'), ""
    print(f"{k:3d} | {oort_rds.mean():10.1f} | {oort_fails:10d} | {ao_rds.mean():8.1f} | {ao_fails:8d} | {p:.4f}{sig}")

print("\nDone.")
