"""
Scalability experiment: 50 clients, Dir(0.1), select 10.
5 seeds × 3 methods = 15 experiments.
Optimised AOA: λ=0.0, δ=0.3.

Part of federation-size ablation:
  20c → C(20,8)=125,970       [done]
  50c → C(50,10)=10,272,278,170  [this script]
"""

import subprocess, sys, os, csv
import numpy as np
from scipy import stats

BASE         = r"C:\Users\Dspike\Documents\FL-AdroidMaLD"
SCRIPT       = os.path.join(BASE, 'run_experiment_50c_dir01.py')
RESULTS_BASE = os.path.join(BASE, 'results', '50c_dir01')
SEEDS        = [42, 123, 456, 789, 2024]
METHODS      = ['fedavg_all', 'random10', 'aoa10']
MAX_ROUNDS   = 50
THR_82       = 0.82

total = len(METHODS) * len(SEEDS)
done, failed = 0, []

for method in METHODS:
    for seed in SEEDS:
        done += 1
        rdir = os.path.join(RESULTS_BASE, method, f'seed{seed}')
        print(f"\n{'='*60}")
        print(f" [{done}/{total}]  {method}  seed={seed}  (50c Dir0.1)")
        print(f"{'='*60}")
        result = subprocess.run(
            [sys.executable, SCRIPT,
             '--method',      method,
             '--seed',        str(seed),
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
print(f" 50-CLIENT RESULTS — λ=0.0 δ=0.3, 5 seeds, Dir(0.1), select 10")
print(f" C(50,10) = 10,272,278,170 search space")
print(f"{'='*65}")

print(f"\n{'Method':12s}  {'Mean F1':>10s}  {'Std':>7s}  {'Rds→82%':>9s}  {'Never':>6s}")
data = {}
for method in METHODS:
    f1  = load_f1(method, SEEDS)
    rds = load_rds(method, SEEDS, THR_82)
    data[method] = (f1, rds)
    if len(f1):
        nev = int((rds == MAX_ROUNDS).sum())
        rm  = rds[rds < MAX_ROUNDS].mean() if (rds < MAX_ROUNDS).any() else 50.0
        print(f"{method:12s}  {f1.mean():>9.2f}%  {f1.std():>6.2f}%  {rm:>8.1f}   {nev:>3d}/5")

# Significance: AOA vs Random
if 'aoa10' in data and 'random10' in data:
    af1, ar = data['aoa10']
    rf1, rr = data['random10']
    n = min(len(af1), len(rf1))
    if n >= 5:
        _, p_f1  = stats.wilcoxon(af1[:n], rf1[:n], alternative='greater')
        _, p_rds = stats.wilcoxon(ar[:n],  rr[:n],  alternative='less')
        def sig(p): return '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else 'n.s.'
        ar_r = ar[ar < MAX_ROUNDS]; rr_r = rr[rr < MAX_ROUNDS]
        spd = (rr_r.mean() - ar_r.mean()) / rr_r.mean() * 100 if len(rr_r) else 0
        print(f"\nAOA vs Random-10:")
        print(f"  F1:  {af1.mean():.2f}% vs {rf1.mean():.2f}%  p={p_f1:.4f} {sig(p_f1)}")
        print(f"  Rds: {ar_r.mean():.1f} vs {rr_r.mean():.1f}  speedup={spd:.1f}%  p={p_rds:.4f} {sig(p_rds)}")

if failed:
    print(f"\nFailed: {failed}")
else:
    print(f"\nAll {total} runs completed.")
