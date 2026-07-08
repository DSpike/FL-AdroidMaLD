"""
SOTA comparison: AOA-8 vs metaheuristic and utility-based client selectors.

Methods (all select k=8 from 20 clients, Dir(0.1)):
  fedavg_all  — lower bound (no selection)
  random8     — standard baseline
  oort8       — Oort (Lai et al., OSDI 2021): utility-based
  poco8       — Power-of-Choice (Cho et al., ICML 2022): loss-biased
  gwo8        — GWO selector (same fitness as AOA)
  pso8        — Binary PSO selector (same fitness as AOA)
  aoa8        — PROPOSED

15 seeds. Significance: paired Wilcoxon (AOA vs each baseline), rounds-to-82%.
Results: results/sota_comparison/{method}/seed{N}/
"""

import subprocess, sys, os, csv
import numpy as np
from scipy import stats

BASE         = r"C:\Users\Dspike\Documents\FL-AdroidMaLD"
SCRIPT       = os.path.join(BASE, 'run_experiment_20c_dir01.py')
RESULTS_BASE = os.path.join(BASE, 'results', 'sota_comparison')
SEEDS        = [42, 123, 456, 789, 2024, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
METHODS      = ['fedavg_all', 'random8', 'oort8', 'poco8', 'gwo8', 'pso8', 'aoa8']
MAX_ROUNDS   = 50
THR_82       = 0.82
LAM, DELTA   = 0.2, 0.3   # original AOA hyperparams (strongest result)

total = len(METHODS) * len(SEEDS)
done, failed = 0, []

for method in METHODS:
    for seed in SEEDS:
        done += 1
        rdir = os.path.join(RESULTS_BASE, method, f'seed{seed}')
        csv_path = os.path.join(rdir, f'{method}_seed{seed}_rounds.csv')
        if os.path.exists(csv_path):
            print(f"[{done}/{total}]  {method}  seed={seed}  (cached)")
            continue
        print(f"\n{'='*60}")
        print(f" [{done}/{total}]  {method}  seed={seed}")
        print(f"{'='*60}")
        result = subprocess.run(
            [sys.executable, SCRIPT,
             '--method',      method,
             '--seed',        str(seed),
             '--lam',         str(LAM),
             '--delta',       str(DELTA),
             '--results_dir', rdir],
            cwd=BASE,
        )
        if result.returncode != 0:
            print(f"  *** FAILED ***")
            failed.append((method, seed))

# ── Summary ────────────────────────────────────────────────────────────────────
def load(method, seeds):
    f1s, rds = [], []
    for s in seeds:
        p = os.path.join(RESULTS_BASE, method, f'seed{s}',
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

aoa_f1, aoa_rd = load('aoa8', SEEDS)

print(f"\n{'='*90}")
print(f" SOTA COMPARISON — 20 clients, k=8, Dir(0.1), 15 seeds, λ={LAM} δ={DELTA}")
print(f"{'='*90}")
print(f"{'Method':12s}  {'Peak F1':>9s}  {'±':>5s}  {'Rds→82%':>8s}  {'Never':>7s}  "
      f"{'p (rds)':>9s}  {'r':>6s}  {'d':>6s}")
print('-' * 90)

for method in METHODS:
    f1, rd = load(method, SEEDS)
    if len(f1) == 0:
        print(f'{method:12s}  no data'); continue

    nev = int((rd == MAX_ROUNDS).sum())
    rm  = rd[rd < MAX_ROUNDS].mean() if (rd < MAX_ROUNDS).any() else MAX_ROUNDS

    if method == 'aoa8':
        print(f'{method:12s}  {f1.mean():>8.2f}%  {f1.std():>4.2f}%  {rm:>7.1f}   '
              f'{nev:>4d}/{len(f1)}   {"—":>9s}  {"—":>6s}  {"—":>6s}  ← PROPOSED')
        continue

    try:
        stat, p = stats.wilcoxon(aoa_rd, rd, alternative='less')
        r = rank_biserial(stat, len(aoa_rd))
        from scipy.stats import norm as _norm
        z = _norm.ppf(1 - p)
        d = z / np.sqrt(len(aoa_rd))
    except Exception:
        p, r, d = 1.0, 0.0, 0.0

    print(f'{method:12s}  {f1.mean():>8.2f}%  {f1.std():>4.2f}%  {rm:>7.1f}   '
          f'{nev:>4d}/{len(f1)}   {p:>7.4f}{sig(p):3s}  {r:>6.3f}  {d:>6.3f}')

if failed:
    print(f"\nFailed runs: {failed}")
else:
    print(f"\nAll {total} SOTA comparison runs completed.")
