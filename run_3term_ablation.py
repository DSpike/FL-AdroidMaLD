"""
Fitness-function term ablation against 3-term baseline (div+cov+divpair).

vol and f1val are already excluded from the 3-term function, so only 3
ablation conditions are needed:
  full_3term  : alpha=0.3  beta=0.3  gamma=0.0  delta=0.0  lam=0.2  (BASELINE)
  no_div      : alpha=0.0  beta=0.3  gamma=0.0  delta=0.0  lam=0.2
  no_cov      : alpha=0.3  beta=0.0  gamma=0.0  delta=0.0  lam=0.2
  no_divpair  : alpha=0.3  beta=0.3  gamma=0.0  delta=0.0  lam=0.0

"full_3term" is loaded directly from results/sota_3term/aoa8/ (no rerun).
3 conditions × 15 seeds = 45 new runs.

Results: results/ablation_3term/{condition}/seed{N}/aoa8_seed{N}_rounds.csv
"""

import subprocess, sys, os, csv
import numpy as np
from scipy import stats
from scipy.stats import norm

BASE         = r"C:\Users\Dspike\Documents\FL-AdroidMaLD"
SCRIPT       = os.path.join(BASE, 'run_experiment_20c_dir01.py')
RESULTS_BASE = os.path.join(BASE, 'results', 'ablation_3term')
SOTA_3TERM   = os.path.join(BASE, 'results', 'sota_3term')
SEEDS        = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 42, 123, 456, 789, 2024]
MAX_ROUNDS   = 50
THR_82       = 0.82

CONDITIONS = {
    'no_div':    dict(alpha=0.0, beta=0.3, gamma=0.0, delta=0.0, lam=0.2),
    'no_cov':    dict(alpha=0.3, beta=0.0, gamma=0.0, delta=0.0, lam=0.2),
    'no_divpair':dict(alpha=0.3, beta=0.3, gamma=0.0, delta=0.0, lam=0.0),
}

total = len(CONDITIONS) * len(SEEDS)
done, failed = 0, []

for cond, weights in CONDITIONS.items():
    for seed in SEEDS:
        done += 1
        rdir     = os.path.join(RESULTS_BASE, cond, f'seed{seed}')
        csv_path = os.path.join(rdir, f'aoa8_seed{seed}_rounds.csv')
        if os.path.exists(csv_path):
            print(f'[{done}/{total}]  {cond}  seed={seed}  (cached)')
            continue
        print(f'\n{"="*60}')
        print(f' [{done}/{total}]  condition={cond}  seed={seed}')
        print(f' weights: a={weights["alpha"]} b={weights["beta"]} '
              f'g={weights["gamma"]} d={weights["delta"]} l={weights["lam"]}')
        print(f'{"="*60}')
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
            print(f'  *** FAILED ***')
            failed.append((cond, seed))

# ── Load helper ────────────────────────────────────────────────────────────────
def load_dir(base_dir, method, seeds):
    f1s, rds = [], []
    for s in seeds:
        p = os.path.join(base_dir, method, f'seed{s}', f'{method}_seed{s}_rounds.csv')
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

def load_abl(cond, seeds):
    f1s, rds = [], []
    for s in seeds:
        p = os.path.join(RESULTS_BASE, cond, f'seed{s}', f'aoa8_seed{s}_rounds.csv')
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

def sig(p): return '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else 'n.s.'

# ── Summary ────────────────────────────────────────────────────────────────────
print(f'\n{"="*80}')
print(f' 3-TERM ABLATION STUDY — AOA, k=8, Dir(0.1), n={len(SEEDS)} seeds')
print(f' Baseline: full_3term loaded from sota_3term/aoa8 (no rerun)')
print(f' Wilcoxon: one-sided, H1 = full converges faster than ablated')
print(f'{"="*80}')

LABELS = {
    'full_3term': 'Full 3-term F(S)',
    'no_div':     'F - div(S)      [a=0]',
    'no_cov':     'F - cov(S)      [b=0]',
    'no_divpair': 'F - divpair(S)  [l=0]',
}

full_f1, full_rd = load_dir(SOTA_3TERM, 'aoa8', SEEDS)
full_rm  = full_rd[full_rd < MAX_ROUNDS].mean() if (full_rd < MAX_ROUNDS).any() else MAX_ROUNDS
nev_full = int((full_rd == MAX_ROUNDS).sum())

print(f'\n  {"Condition":<26}  {"PeakF1":>8}  {"Rds82%":>7}  {"Never":>6}  '
      f'{"p(full<ablated)":>16}  {"r":>6}  {"d":>6}')
print(f'  {"-"*26}  {"-"*8}  {"-"*7}  {"-"*6}  {"-"*16}  {"-"*6}  {"-"*6}')
print(f'  {LABELS["full_3term"]:<26}  {full_f1.mean():7.2f}%  {full_rm:6.1f}   '
      f'{nev_full:3d}/{len(full_f1)}   {"--- (baseline)":>16}')

for cond in ['no_div', 'no_cov', 'no_divpair']:
    f1, rd = load_abl(cond, SEEDS)
    if len(rd) == 0:
        print(f'  {LABELS[cond]:<26}  (no data)')
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
    print(f'  {LABELS[cond]:<26}  {f1.mean():7.2f}%  {rm:6.1f}   '
          f'{nev:3d}/{len(f1)}   {p:14.4f}{sig(p):3s}  {r:6.3f}  {d:6.3f}')

if failed:
    print(f'\nFailed: {failed}')
else:
    print(f'\nAll {total} ablation runs completed successfully.')
