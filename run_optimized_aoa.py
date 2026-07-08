"""
15-seed main comparison using optimised AOA hyperparameters (λ=0.0, δ=0.3).
Ablation showed λ=0.0 reaches 82% F1 in 31.2 rounds vs 36.8 with λ=0.2.

Methods: fedavg_all, random8, aoa8 (λ=0.0, δ=0.3)
Seeds: 15  |  20 clients, Dir(0.1), select 8
Results: results/20c_dir01_optimized/
"""

import subprocess, sys, os, csv
import numpy as np
from scipy import stats

BASE         = r"C:\Users\Dspike\Documents\FL-AdroidMaLD"
SCRIPT       = os.path.join(BASE, 'run_experiment_20c_dir01.py')
RESULTS_BASE = os.path.join(BASE, 'results', '20c_dir01_optimized')
ALL_SEEDS    = [42, 123, 456, 789, 2024, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
METHODS      = ['fedavg_all', 'random8', 'aoa8']
MAX_ROUNDS   = 50
THR_82       = 0.82

total = len(METHODS) * len(ALL_SEEDS)
done, failed = 0, []

for method in METHODS:
    for seed in ALL_SEEDS:
        done += 1
        rdir = os.path.join(RESULTS_BASE, method, f'seed{seed}')
        print(f"\n{'='*60}")
        print(f" [{done}/{total}]  {method}  seed={seed}  (optimised λ=0.0 δ=0.3)")
        print(f"{'='*60}")

        # AOA uses optimised lam=0.0, delta=0.3; others ignore these args
        result = subprocess.run(
            [sys.executable, SCRIPT,
             '--method',      method,
             '--seed',        str(seed),
             '--lam',         '0.0',
             '--delta',       '0.3',
             '--results_dir', rdir],
            cwd=BASE,
        )
        if result.returncode != 0:
            print(f"  *** FAILED ***")
            failed.append((method, seed))

# ── Summary ───────────────────────────────────────────────────────────────────
def load_f1(method, seeds):
    bests = []
    for s in seeds:
        p = os.path.join(RESULTS_BASE, method, f'seed{s}',
                         f'{method}_seed{s}_rounds.csv')
        if os.path.exists(p):
            with open(p) as f:
                bests.append(max(float(r['f1_macro']) for r in csv.DictReader(f)) * 100)
    return np.array(bests)

def load_rds(method, seeds, thr):
    rds = []
    for s in seeds:
        p = os.path.join(RESULTS_BASE, method, f'seed{s}',
                         f'{method}_seed{s}_rounds.csv')
        if not os.path.exists(p): continue
        hit = None
        with open(p) as f:
            for row in csv.DictReader(f):
                if row['phase'] == 'phase2' and float(row['f1_macro']) >= thr and hit is None:
                    hit = int(row['round'])
        rds.append(hit if hit else MAX_ROUNDS)
    return np.array(rds)

print(f"\n{'='*65}")
print(f" OPTIMISED AOA (λ=0.0, δ=0.3) — 15 seeds, 20 clients, Dir(0.1)")
print(f"{'='*65}")

print(f"\n{'Method':12s}  {'Mean F1':>10s}  {'Std':>7s}  {'Rds→82%':>9s}  {'Never':>6s}")
data = {}
for method in METHODS:
    f1  = load_f1(method, ALL_SEEDS)
    rds = load_rds(method, ALL_SEEDS, THR_82)
    data[method] = (f1, rds)
    if len(f1):
        nev = int((rds == MAX_ROUNDS).sum())
        rm  = rds[rds < MAX_ROUNDS].mean() if (rds < MAX_ROUNDS).any() else 50.0
        print(f"{method:12s}  {f1.mean():>9.2f}%  {f1.std():>6.2f}%  {rm:>8.1f}   {nev:>4d}/15")

# Significance
print(f"\nPaired Wilcoxon — AOA vs Random-8:")
aoa_f1, aoa_r = data['aoa8']
rnd_f1, rnd_r = data['random8']
n = min(len(aoa_f1), len(rnd_f1))

_, p_f1  = stats.wilcoxon(aoa_f1[:n], rnd_f1[:n], alternative='greater')
_, p_rds = stats.wilcoxon(aoa_r[:n],  rnd_r[:n],  alternative='less')

def sig(p): return '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else 'n.s.'

rnd_reached = rnd_r[rnd_r < MAX_ROUNDS]
aoa_reached = aoa_r[aoa_r < MAX_ROUNDS]
speedup_pct = (rnd_reached.mean() - aoa_reached.mean()) / rnd_reached.mean() * 100

print(f"  F1:  AOA={aoa_f1.mean():.2f}% vs Random={rnd_f1.mean():.2f}%  "
      f"(Δ={aoa_f1.mean()-rnd_f1.mean():+.2f}%)  p={p_f1:.4f} {sig(p_f1)}")
print(f"  Rds: AOA={aoa_reached.mean():.1f} vs Random={rnd_reached.mean():.1f}  "
      f"(speedup={speedup_pct:.1f}%)  p={p_rds:.4f} {sig(p_rds)}")

if failed:
    print(f"\nFailed: {failed}")
else:
    print(f"\nAll {total} runs completed.")
