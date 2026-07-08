"""
k-sensitivity plot (results/k_sensitivity, 5 seeds per k x method).

Generates in plots/k_sensitivity/:
  1. k_sensitivity_rounds.png — rounds-to-82% vs k, one line per method
  2. k_sensitivity_peakf1.png — peak F1-macro vs k, one line per method
"""

import os, csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE         = r'C:\Users\Dspike\Documents\FL-AdroidMaLD'
RESULTS_BASE = os.path.join(BASE, 'results', 'k_sensitivity')
OUT_DIR      = os.path.join(BASE, 'plots', 'k_sensitivity')
os.makedirs(OUT_DIR, exist_ok=True)

K_VALUES = [4, 6, 8, 10, 12]
SEEDS    = [42, 123, 456, 789, 2024]
METHODS  = ['fedavg_all', 'random8', 'gwo8', 'pso8', 'aoa8']
LABELS   = {
    'fedavg_all': 'FedAvg-all', 'random8': 'Random', 'gwo8': 'GWO',
    'pso8': 'PSO', 'aoa8': 'AOA (ours)',
}
COLORS = {
    'fedavg_all': '#888888', 'random8': '#2196F3', 'gwo8': '#4CAF50',
    'pso8': '#00BCD4', 'aoa8': '#E53935',
}
MARKERS = {'fedavg_all': 's', 'random8': '^', 'gwo8': 'P', 'pso8': 'X', 'aoa8': 'o'}
MAX_ROUNDS = 50
THR_82     = 0.82

def load(k, method, seeds):
    f1s, rds = [], []
    for s in seeds:
        p = os.path.join(RESULTS_BASE, f'k{k}', method, f'seed{s}', f'{method}_seed{s}_rounds.csv')
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

# ── Plot 1: Rounds-to-82% vs k (raw mean, failures capped at 50 — fair comparison) ──
fig, ax = plt.subplots(figsize=(8, 5.5))
for method in METHODS:
    means, stds = [], []
    for k in K_VALUES:
        f1, rd = load(k, method, SEEDS)
        means.append(rd.mean())
        stds.append(rd.std())
    means, stds = np.array(means), np.array(stds)
    lw = 2.5 if method == 'aoa8' else 1.6
    ax.errorbar(K_VALUES, means, yerr=stds, color=COLORS[method], marker=MARKERS[method],
                markersize=7, linewidth=lw, capsize=4, label=LABELS[method],
                zorder=10 if method == 'aoa8' else 5)

ax.set_xlabel('k (clients selected per round)', fontsize=12)
ax.set_ylabel('Rounds to reach 82% F1-macro\n(failures capped at 50)', fontsize=11)
ax.set_title('k-Sensitivity — 20 clients, Dir(0.1), 5 seeds\n(raw mean incl. non-converged seeds, matches Wilcoxon test)', fontsize=10)
ax.set_xticks(K_VALUES)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
out = os.path.join(OUT_DIR, 'k_sensitivity_rounds.png')
plt.savefig(out, dpi=150)
plt.close()
print(f"Saved: {out}")

# ── Plot 2: Peak F1 vs k ───────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5.5))
for method in METHODS:
    means, stds = [], []
    for k in K_VALUES:
        f1, rd = load(k, method, SEEDS)
        means.append(f1.mean())
        stds.append(f1.std())
    means, stds = np.array(means), np.array(stds)
    lw = 2.5 if method == 'aoa8' else 1.6
    ax.errorbar(K_VALUES, means, yerr=stds, color=COLORS[method], marker=MARKERS[method],
                markersize=7, linewidth=lw, capsize=4, label=LABELS[method],
                zorder=10 if method == 'aoa8' else 5)

ax.set_xlabel('k (clients selected per round)', fontsize=12)
ax.set_ylabel('Peak F1-Macro (%)', fontsize=12)
ax.set_title('Peak F1 vs k — 20 clients, Dir(0.1), 5 seeds', fontsize=11)
ax.set_xticks(K_VALUES)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
out = os.path.join(OUT_DIR, 'k_sensitivity_peakf1.png')
plt.savefig(out, dpi=150)
plt.close()
print(f"Saved: {out}")

print("\nAll k-sensitivity plots generated.")
