"""
Extension run: 10 additional seeds for the 20-client Dir(0.1) experiment.
Seeds [42,123,456,789,2024] already done — this adds [1,2,3,4,5,6,7,8,9,10].
Combined with prior 5 seeds gives 15 total for statistical significance.
"""

import subprocess
import sys
import os
import csv
import numpy as np

# 10 new seeds (do NOT overlap with the first 5 already done)
NEW_SEEDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
ALL_SEEDS = [42, 123, 456, 789, 2024] + NEW_SEEDS   # 15 total

METHODS = ['fedavg_all', 'random8', 'aoa8']
BASE    = r"C:\Users\Dspike\Documents\FL-AdroidMaLD"
SCRIPT  = os.path.join(BASE, 'run_experiment_20c_dir01.py')

# ── Run new seeds only ────────────────────────────────────────────────────────
total  = len(METHODS) * len(NEW_SEEDS)
done   = 0
failed = []

for method in METHODS:
    for seed in NEW_SEEDS:
        done += 1
        print(f"\n{'='*60}")
        print(f" [{done}/{total}]  {method}  seed={seed}  (20c Dir0.1 ext)")
        print(f"{'='*60}")
        result = subprocess.run(
            [sys.executable, SCRIPT, '--method', method, '--seed', str(seed)],
            cwd=BASE,
        )
        if result.returncode != 0:
            print(f"  *** FAILED: {method} seed={seed} ***")
            failed.append((method, seed))

# ── Combined 15-seed summary ──────────────────────────────────────────────────
RESULTS_BASE = os.path.join(BASE, 'results', '20c_dir01_5seed')
THRESHOLDS   = [0.75, 0.78, 0.80, 0.82]

print(f"\n{'='*60}")
print(f" COMBINED RESULTS — 20 clients, Dir(0.1), {len(ALL_SEEDS)} seeds")
print(f"{'='*60}")

# Best F1 summary
print(f"\n{'Method':12s}  {'Mean F1':>10s}  {'Std':>8s}  {'n':>4s}")
for method in METHODS:
    bests = []
    for seed in ALL_SEEDS:
        path = os.path.join(RESULTS_BASE, method, f'seed{seed}',
                            f'{method}_seed{seed}_rounds.csv')
        if os.path.exists(path):
            with open(path) as f:
                reader = csv.reader(f)
                next(reader)
                bests.append(max(float(row[3]) for row in reader) * 100)
    if bests:
        print(f"{method:12s}  {np.mean(bests):>9.2f}%  {np.std(bests):>7.2f}%  {len(bests):>4d}")

# Convergence speed
print(f"\nRounds to threshold (phase 2):")
print(f"{'Threshold':>12s}  {'fedavg_all':>22s}  {'random8':>22s}  {'aoa8':>22s}")

for thr in THRESHOLDS:
    row_vals = []
    for method in METHODS:
        rounds_list = []
        for seed in ALL_SEEDS:
            path = os.path.join(RESULTS_BASE, method, f'seed{seed}',
                                f'{method}_seed{seed}_rounds.csv')
            if not os.path.exists(path):
                continue
            hit = None
            with open(path) as f:
                reader = csv.reader(f)
                next(reader)
                for r in reader:
                    if r[1] == 'phase2' and float(r[3]) >= thr:
                        hit = int(r[0])
                        break
            rounds_list.append(hit if hit is not None else 999)
        reached = [r for r in rounds_list if r < 999]
        never   = len(rounds_list) - len(reached)
        if reached:
            tag = f'[{never}x-]' if never else ''
            row_vals.append(f"{np.mean(reached):.1f}±{np.std(reached):.1f} {tag}")
        else:
            row_vals.append('never')
    print(f"  {thr*100:.0f}%        {'':>5}  {row_vals[0]:>22s}  {row_vals[1]:>22s}  {row_vals[2]:>22s}")

if failed:
    print(f"\nFailed: {failed}")
else:
    print(f"\nAll {total} extension runs completed.")
