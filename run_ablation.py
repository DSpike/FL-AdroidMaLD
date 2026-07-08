"""
Fitness-function term ablation study.

Each condition removes exactly one term from F(S) by setting its weight to 0.
Remaining weights are kept at their default values (no renormalization: the
fitness is used for within-round subset ranking, so absolute scale is irrelevant).

Conditions (6 total):
  full       : α=0.3  β=0.3  γ=0.1  δ=0.3  λ=0.2  (baseline)
  no_div     : α=0    β=0.3  γ=0.1  δ=0.3  λ=0.2  (remove divergence)
  no_cov     : α=0.3  β=0    γ=0.1  δ=0.3  λ=0.2  (remove class coverage)
  no_vol     : α=0.3  β=0.3  γ=0    δ=0.3  λ=0.2  (remove data volume)
  no_val     : α=0.3  β=0.3  γ=0.1  δ=0    λ=0.2  (remove local validation F1)
  no_divpair : α=0.3  β=0.3  γ=0.1  δ=0.3  λ=0    (remove pairwise diversity)

15 seeds × 6 conditions = 90 runs total.
Results: results/ablation_terms/{condition}/seed{N}/aoa8_seed{N}_rounds.csv
"""

import subprocess, sys, os, csv
import numpy as np
from scipy import stats
from scipy.stats import norm

BASE         = r"C:\Users\Dspike\Documents\FL-AdroidMaLD"
SCRIPT       = os.path.join(BASE, 'run_experiment_20c_dir01.py')
RESULTS_BASE = os.path.join(BASE, 'results', 'ablation_terms')
SEEDS        = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 42, 123, 456, 789, 2024]
MAX_ROUNDS   = 50
THR_82       = 0.82

CONDITIONS = {
    'no_div':     dict(alpha=0.0, beta=0.3, gamma=0.1, delta=0.3, lam=0.2),
    'no_cov':     dict(alpha=0.3, beta=0.0, gamma=0.1, delta=0.3, lam=0.2),
    'no_vol':     dict(alpha=0.3, beta=0.3, gamma=0.0, delta=0.3, lam=0.2),
    'no_val':     dict(alpha=0.3, beta=0.3, gamma=0.1, delta=0.0, lam=0.2),
    'no_divpair': dict(alpha=0.3, beta=0.3, gamma=0.1, delta=0.3, lam=0.0),
}

total = len(CONDITIONS) * len(SEEDS)
done, failed = 0, []

for cond, weights in CONDITIONS.items():
    for seed in SEEDS:
        done += 1
        rdir     = os.path.join(RESULTS_BASE, cond, f'seed{seed}')
        csv_path = os.path.join(rdir, f'aoa8_seed{seed}_rounds.csv')
        if os.path.exists(csv_path):
            print(f"[{done}/{total}]  {cond}  seed={seed}  (cached)")
            continue
        print(f"\n{'='*60}")
        print(f" [{done}/{total}]  condition={cond}  seed={seed}")
        print(f" weights: α={weights['alpha']} β={weights['beta']} "
              f"γ={weights['gamma']} δ={weights['delta']} λ={weights['lam']}")
        print(f"{'='*60}")
        result = subprocess.run(
            [sys.executable, SCRIPT,
             '--method',      'aoa8',
             '--seed',        str(seed),
             '--k_select',    '8',
             '--alpha',       str(weights['alpha']),
             '--beta',        str(weights['beta']),
             '--gamma',       str(weights['gamma']),
             '--delta',       str(weights['delta']),
             '--lam',         str(weights['lam']),
             '--results_dir', rdir],
            cwd=BASE,
        )
        if result.returncode != 0:
            print(f"  *** FAILED ***")
            failed.append((cond, seed))

# ── Summary ────────────────────────────────────────────────────────────────────
def load(cond, seeds):
    f1s, rds = [], []
    for s in seeds:
        p = os.path.join(RESULTS_BASE, cond, f'seed{s}', f'aoa8_seed{s}_rounds.csv')
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

def sig(p):
    return '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'n.s.'

# For "full" condition, load from SOTA comparison results (same seeds, same weights)
SOTA_AOA_DIR = os.path.join(BASE, '..', 'results', 'sota_comparison', 'aoa8')

def load_sota_full(seeds):
    f1s, rds = [], []
    for s in seeds:
        p = os.path.join(BASE, '..', 'results', 'sota_comparison', 'aoa8',
                         f'seed{s}', f'aoa8_seed{s}_rounds.csv')
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

print(f"\n{'='*80}")
print(f" FITNESS TERM ABLATION — AOA, k=8, Dir(0.1), {len(SEEDS)} seeds")
print(f" Full baseline loaded from SOTA comparison results (identical run)")
print(f" Wilcoxon: one-sided, H1 = full converges faster than ablated condition")
print(f"{'='*80}")

TERM_LABELS = {
    'full':       'Full F(S)',
    'no_div':     'F - div(S)      [a=0]',
    'no_cov':     'F - cov(S)      [b=0]',
    'no_vol':     'F - vol(S)      [g=0]',
    'no_val':     'F - f1val(S)    [d=0]',
    'no_divpair': 'F - divpair(S)  [l=0]',
}

full_f1, full_rd = load_sota_full(SEEDS)
full_rm  = full_rd[full_rd < MAX_ROUNDS].mean() if (full_rd < MAX_ROUNDS).any() else MAX_ROUNDS
nev_full = int((full_rd == MAX_ROUNDS).sum())

print(f"\n  {'Condition':<26}  {'PeakF1':>8}  {'Rds82%':>7}  {'Never':>6}  "
      f"{'p(full<ablated)':>16}  {'r':>6}  {'d':>6}")
print(f"  {'-'*26}  {'-'*8}  {'-'*7}  {'-'*6}  {'-'*16}  {'-'*6}  {'-'*6}")
print(f"  {TERM_LABELS['full']:<26}  {full_f1.mean():7.2f}%  {full_rm:6.1f}   "
      f"{nev_full:3d}/{len(full_f1)}   {'--- (baseline)':>16}")

for cond in ['no_div', 'no_cov', 'no_vol', 'no_val', 'no_divpair']:
    f1, rd = load(cond, SEEDS)
    if len(rd) == 0:
        print(f"  {TERM_LABELS[cond]:<26}  (no data)")
        continue
    nev = int((rd == MAX_ROUNDS).sum())
    rm  = rd[rd < MAX_ROUNDS].mean() if (rd < MAX_ROUNDS).any() else MAX_ROUNDS
    try:
        stat, p = stats.wilcoxon(full_rd, rd, alternative='less')
        r = 1 - 2 * stat / (len(full_rd) * (len(full_rd) + 1) / 2)
        z = norm.ppf(1 - p) if p < 1 else 0.0
        d = z / np.sqrt(len(full_rd))
    except Exception:
        p, r, d = 1.0, 0.0, 0.0
    print(f"  {TERM_LABELS[cond]:<26}  {f1.mean():7.2f}%  {rm:6.1f}   "
          f"{nev:3d}/{len(f1)}   {p:14.4f}{sig(p):3s}  {r:6.3f}  {d:6.3f}")

if failed:
    print(f"\nFailed: {failed}")
else:
    print(f"\nAll {total} ablation runs completed successfully.")
