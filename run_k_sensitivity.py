"""
k-sensitivity ablation: does AOA's relative position vs other metaheuristics
(GWO, PSO) and naive baselines hold across different selection fractions?

Setup: 20 clients, Dir(0.1), k ∈ {4, 6, 8, 10, 12}, 5 seeds, λ=0.2 δ=0.3
Methods: fedavg_all, random8, oort8, poco8, gwo8, pso8, aoa8 (proposed)
Total: 5 k-values × 7 methods × 5 seeds = 175 experiments (50 new Oort/PoCo runs)

Motivation: at k=8 (20-client SOTA comparison, 15 seeds), AOA significantly
beat Oort/PoCo but was statistically tied with PSO (p=0.659) and only
borderline ahead of GWO (p=0.046). This ablation checks whether AOA's
relative standing vs PSO/GWO specifically shifts at other k values, rather
than assuming k=8 is representative.

Results: results/k_sensitivity/k{K}/{method}/seed{N}/
"""

import subprocess, sys, os, csv
import numpy as np
from scipy import stats
from scipy.stats import norm
from math import comb

BASE         = r"C:\Users\Dspike\Documents\FL-AdroidMaLD"
SCRIPT       = os.path.join(BASE, 'run_experiment_20c_dir01.py')
RESULTS_BASE = os.path.join(BASE, 'results', 'k_sensitivity')
K_VALUES     = [4, 6, 8, 10, 12]
METHODS      = ['fedavg_all', 'random8', 'oort8', 'poco8', 'gwo8', 'pso8', 'aoa8']
SEEDS        = [42, 123, 456, 789, 2024]
MAX_ROUNDS   = 50
THR_82       = 0.82
LAM, DELTA   = 0.2, 0.3
N_CLIENTS    = 20

total = len(K_VALUES) * len(METHODS) * len(SEEDS)
done, failed = 0, []

for k in K_VALUES:
    for method in METHODS:
        for seed in SEEDS:
            done += 1
            rdir     = os.path.join(RESULTS_BASE, f'k{k}', method, f'seed{seed}')
            csv_path = os.path.join(rdir, f'{method}_seed{seed}_rounds.csv')
            if os.path.exists(csv_path):
                print(f"[{done}/{total}]  k={k}  {method}  seed={seed}  (cached)")
                continue
            print(f"\n{'='*60}")
            print(f" [{done}/{total}]  k={k}  {method}  seed={seed}")
            print(f"{'='*60}")
            result = subprocess.run(
                [sys.executable, SCRIPT,
                 '--method',      method,
                 '--seed',        str(seed),
                 '--k_select',    str(k),
                 '--lam',         str(LAM),
                 '--delta',       str(DELTA),
                 '--results_dir', rdir],
                cwd=BASE,
            )
            if result.returncode != 0:
                print(f"  *** FAILED ***")
                failed.append((k, method, seed))

# ── Summary ────────────────────────────────────────────────────────────────────
def load(k, method, seeds):
    f1s, rds = [], []
    for s in seeds:
        p = os.path.join(RESULTS_BASE, f'k{k}', method, f'seed{s}',
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

def rank_biserial(w, n):
    return 1 - 2 * w / (n * (n + 1) / 2)

print(f"\n{'='*100}")
print(f" k-SENSITIVITY — 20 clients, Dir(0.1), λ={LAM} δ={DELTA}, {len(SEEDS)} seeds")
print(f"{'='*100}")

for k in K_VALUES:
    search_space = comb(N_CLIENTS, k)
    print(f"\n--- k={k}  (C({N_CLIENTS},{k}) = {search_space:,} candidate subsets) ---")

    loaded = {m: load(k, m, SEEDS) for m in METHODS}
    if not all(len(loaded[m][0]) > 0 for m in METHODS):
        print("  incomplete")
        continue

    ao_f1, ao_rd = loaded['aoa8']
    ao_rm = ao_rd[ao_rd < MAX_ROUNDS].mean() if (ao_rd < MAX_ROUNDS).any() else MAX_ROUNDS

    print(f"  {'Method':12s}  {'PeakF1':>7s}  {'Rds82%':>7s}  {'Never':>6s}  {'p(vs AOA)':>10s}  {'r':>6s}  {'d':>6s}")
    for m in METHODS:
        f1, rd = loaded[m]
        nev = int((rd == MAX_ROUNDS).sum())
        rm  = rd[rd < MAX_ROUNDS].mean() if (rd < MAX_ROUNDS).any() else MAX_ROUNDS
        if m == 'aoa8':
            print(f"  {m:12s}  {f1.mean():6.2f}%  {rm:6.1f}   {nev:3d}/{len(f1)}   {'---':>10s}  {'---':>6s}  {'---':>6s}  <- PROPOSED")
            continue
        try:
            stat, p = stats.wilcoxon(ao_rd, rd, alternative='less')
            r = rank_biserial(stat, len(ao_rd))
            z = norm.ppf(1 - p) if p < 1 else 0.0
            d = z / np.sqrt(len(ao_rd))
        except Exception:
            p, r, d = 1.0, 0.0, 0.0
        print(f"  {m:12s}  {f1.mean():6.2f}%  {rm:6.1f}   {nev:3d}/{len(f1)}   {p:8.4f}{sig(p):3s}  {r:6.3f}  {d:6.3f}")

if failed:
    print(f"\nFailed: {failed}")
else:
    print(f"\nAll {total} k-sensitivity runs completed.")
