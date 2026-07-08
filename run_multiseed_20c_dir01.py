"""
5-seed experiment: FedAvg-all vs Random-8 vs AOA-8
20 clients, Dir(0.1), select 8 per round.

C(20,8) = 125,970 — gives AOA a genuine combinatorial search task.
Runs 15 experiments total (3 methods × 5 seeds) sequentially.
"""

import subprocess
import sys
import os
import csv
import numpy as np

SEEDS   = [42, 123, 456, 789, 2024]
METHODS = ['fedavg_all', 'random8', 'aoa8']
BASE    = r"C:\Users\Dspike\Documents\FL-AdroidMaLD"
SCRIPT  = os.path.join(BASE, 'run_experiment_20c_dir01.py')

total  = len(METHODS) * len(SEEDS)
done   = 0
failed = []

for method in METHODS:
    for seed in SEEDS:
        done += 1
        print(f"\n{'='*60}")
        print(f" [{done}/{total}]  {method}  seed={seed}  (20 clients, Dir0.1)")
        print(f"{'='*60}")

        result = subprocess.run(
            [sys.executable, SCRIPT, '--method', method, '--seed', str(seed)],
            cwd=BASE,
        )
        if result.returncode != 0:
            print(f"  *** FAILED: {method} seed={seed} ***")
            failed.append((method, seed))

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(" RESULTS SUMMARY  —  20 clients, Dir(0.1), 5 seeds")
print(f"{'='*60}")
print(f"{'Method':12s}  {'Mean best F1':>14s}  {'Std':>8s}  Per-seed bests")

for method in METHODS:
    bests = []
    for seed in SEEDS:
        csv_path = os.path.join(
            BASE, 'results', '20c_dir01_5seed', method, f'seed{seed}',
            f'{method}_seed{seed}_rounds.csv'
        )
        if os.path.exists(csv_path):
            with open(csv_path) as f:
                reader = csv.reader(f)
                next(reader)
                f1s = [float(row[3]) for row in reader]
            bests.append(max(f1s) * 100)
        else:
            print(f"  WARNING: CSV not found for {method} seed={seed}")

    if bests:
        seeds_str = '  '.join(f"{b:.2f}%" for b in bests)
        print(f"{method:12s}  {np.mean(bests):>13.2f}%  {np.std(bests):>7.2f}%  [{seeds_str}]")

if failed:
    print(f"\nFailed runs: {failed}")
else:
    print("\nAll runs completed successfully.")
