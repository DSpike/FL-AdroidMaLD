"""
Summary plots for 3-term SOTA comparison.
AO/GWO/PSO loaded from results/sota_3term/ (15 seeds, 3-term weights).
FedAvg-all/Random/Oort/PoCo loaded from results/sota_comparison/ (unchanged).

Generates in plots/sota_3term/:
  1. convergence_curves.png
  2. bar_peak_f1.png
  3. bar_convergence.png
  4. accuracy_vs_f1_dual_panel.png
"""

import os, csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE         = r'C:\Users\Dspike\Documents\FL-AdroidMaLD'
DIR_3TERM    = os.path.join(BASE, 'results', 'sota_3term')
DIR_OLD      = os.path.join(BASE, 'results', 'sota_comparison')
OUT_DIR      = os.path.join(BASE, 'plots', 'sota_3term')
os.makedirs(OUT_DIR, exist_ok=True)

SEEDS      = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 42, 123, 456, 789, 2024]
METHODS    = ['fedavg_all', 'random8', 'oort8', 'poco8', 'gwo8', 'pso8', 'aoa8']
METHODS_3T = {'aoa8', 'gwo8', 'pso8'}

LABELS = {
    'fedavg_all': 'FedAvg-all', 'random8': 'Random-8', 'oort8': 'Oort-8',
    'poco8': 'PoCo-8', 'gwo8': 'GWO-8', 'pso8': 'PSO-8', 'aoa8': 'AO-8 (ours)',
}
COLORS = {
    'fedavg_all': '#888888', 'random8': '#2196F3', 'oort8': '#9C27B0',
    'poco8': '#FF9800', 'gwo8': '#4CAF50', 'pso8': '#00BCD4', 'aoa8': '#E53935',
}
MARKERS = {
    'fedavg_all': 's', 'random8': '^', 'oort8': 'D',
    'poco8': 'v', 'gwo8': 'P', 'pso8': 'X', 'aoa8': 'o',
}
N_ROUNDS   = 50
MAX_ROUNDS = 50
THRESHOLDS = [0.75, 0.78, 0.80, 0.82]

def result_dir(method):
    return DIR_3TERM if method in METHODS_3T else DIR_OLD

def load_curves(method, seeds, metric='f1_macro'):
    curves = []
    rdir = result_dir(method)
    for s in seeds:
        p = os.path.join(rdir, method, f'seed{s}', f'{method}_seed{s}_rounds.csv')
        if not os.path.exists(p): continue
        rounds_v = {}
        with open(p) as f:
            for row in csv.DictReader(f):
                rounds_v[int(row['round'])] = float(row[metric]) * 100
        curves.append([rounds_v.get(r, np.nan) for r in range(1, N_ROUNDS + 1)])
    return np.array(curves) if curves else np.empty((0, N_ROUNDS))

def load_best_f1(method, seeds):
    bests, rdir = [], result_dir(method)
    for s in seeds:
        p = os.path.join(rdir, method, f'seed{s}', f'{method}_seed{s}_rounds.csv')
        if os.path.exists(p):
            with open(p) as f:
                bests.append(max(float(r['f1_macro']) for r in csv.DictReader(f)) * 100)
    return np.array(bests)

def load_rounds_to(method, seeds, thr):
    rds, rdir = [], result_dir(method)
    for s in seeds:
        p = os.path.join(rdir, method, f'seed{s}', f'{method}_seed{s}_rounds.csv')
        if not os.path.exists(p): continue
        hit = None
        with open(p) as f:
            for row in csv.DictReader(f):
                if row['phase'] == 'phase2' and float(row['f1_macro']) >= thr and hit is None:
                    hit = int(row['round'])
        rds.append(hit if hit else MAX_ROUNDS)
    return np.array(rds)

curves_data = {m: load_curves(m, SEEDS) for m in METHODS}
acc_data    = {m: load_curves(m, SEEDS, 'accuracy') for m in METHODS}
f1_data     = {m: load_best_f1(m, SEEDS) for m in METHODS}
available   = [m for m in METHODS if len(curves_data[m]) > 0]
print(f"Methods with data: {available}")
for m in available:
    print(f"  {m}: n={len(curves_data[m])} seeds")

rounds = np.arange(1, N_ROUNDS + 1)

# ── Plot 1: Convergence curves ────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5.5))
for method in available:
    c = curves_data[method]
    mean = np.nanmean(c, axis=0)
    std  = np.nanstd(c, axis=0)
    lw = 2.5 if method == 'aoa8' else 1.6
    ax.plot(rounds, mean, color=COLORS[method], marker=MARKERS[method],
            markevery=5, markersize=5, linewidth=lw, label=LABELS[method],
            zorder=10 if method == 'aoa8' else 5)
    ax.fill_between(rounds, mean - std, mean + std, color=COLORS[method], alpha=0.10)

ax.axvline(x=20, color='black', linestyle='--', linewidth=1, alpha=0.5)
ax.text(20.5, 4, 'Phase 2\nstart', fontsize=8, color='black')
ax.axhline(y=82, color='gray', linestyle=':', linewidth=1, alpha=0.7)
ax.text(1, 82.8, '82% threshold', fontsize=8, color='gray')
ax.set_xlabel('Communication Round', fontsize=12)
ax.set_ylabel('F1-Macro (%)', fontsize=12)
ax.set_title('SOTA Comparison — 20 clients, k=8, Dir(0.1), 15 seeds\n'
             '(AO/GWO/PSO: 3-term fitness: div+cov+divpair)', fontsize=11)
ax.legend(fontsize=9, ncol=2, loc='lower right')
ax.set_xlim(1, N_ROUNDS); ax.set_ylim(0, 100)
ax.grid(True, alpha=0.3)
plt.tight_layout()
out = os.path.join(OUT_DIR, 'convergence_curves.png')
plt.savefig(out, dpi=150); plt.close()
print(f"Saved: {out}")

# ── Plot 2: Peak F1 bar chart ─────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(len(available))
for i, method in enumerate(available):
    f1   = f1_data[method]
    edge = '#7a0000' if method == 'aoa8' else 'black'
    lw_e = 2 if method == 'aoa8' else 1
    ax.bar(i, f1.mean(), yerr=f1.std(), color=COLORS[method], capsize=5, width=0.6,
           label=LABELS[method], alpha=0.88, edgecolor=edge, linewidth=lw_e)
    ax.text(i, f1.mean() + f1.std() + 0.4, f'{f1.mean():.2f}%',
            ha='center', va='bottom', fontsize=9, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([LABELS[m] for m in available], fontsize=9, rotation=20, ha='right')
ax.set_ylabel('Peak F1-Macro (%)', fontsize=11)
ax.set_title('Peak F1-Macro — 3-term SOTA Comparison, 20c Dir(0.1), 15 seeds', fontsize=11)
ax.set_ylim(75, 92); ax.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
out = os.path.join(OUT_DIR, 'bar_peak_f1.png')
plt.savefig(out, dpi=150); plt.close()
print(f"Saved: {out}")

# ── Plot 3: Rounds-to-threshold grouped bar ───────────────────────────────────
fig, ax = plt.subplots(figsize=(16, 7.5))
n_thr = len(THRESHOLDS); width = 0.11
x = np.arange(n_thr)
for i, method in enumerate(available):
    means, stds = [], []
    for thr in THRESHOLDS:
        rd = load_rounds_to(method, SEEDS, thr)
        reached = rd[rd < MAX_ROUNDS]
        means.append(reached.mean() if len(reached) else MAX_ROUNDS)
        stds.append(reached.std()  if len(reached) > 1 else 0)
    offset = (i - len(available) / 2 + 0.5) * width
    edge = '#7a0000' if method == 'aoa8' else 'black'
    bars = ax.bar(x + offset, means, width, yerr=stds, label=LABELS[method],
                  color=COLORS[method], capsize=3, alpha=0.88, edgecolor=edge)
    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2, m + s + 1.0, f'{m:.1f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold', rotation=90)
ax.set_xticks(x)
ax.set_xticklabels([f'{int(t*100)}%' for t in THRESHOLDS], fontsize=15)
ax.set_xlabel('F1-Macro Threshold', fontsize=16)
ax.set_ylabel('Rounds to Reach Threshold (converged seeds only)', fontsize=15)
ax.set_title('Convergence Speed — 3-term SOTA Comparison, 20c Dir(0.1), 15 seeds', fontsize=16)
ax.tick_params(axis='y', labelsize=13)
ax.set_ylim(0, 62)
ax.legend(fontsize=13, ncol=4, loc='upper left')
ax.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
out = os.path.join(OUT_DIR, 'bar_convergence.png')
plt.savefig(out, dpi=150); plt.close()
print(f"Saved: {out}")

# ── Plot 4: Accuracy + F1 dual panel ─────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
for method in available:
    lw = 2.5 if method == 'aoa8' else 1.6
    zo = 10 if method == 'aoa8' else 5
    a  = acc_data[method]
    ma = np.nanmean(a, axis=0); sa = np.nanstd(a, axis=0)
    ax1.plot(rounds, ma, color=COLORS[method], marker=MARKERS[method],
             markevery=5, markersize=5, linewidth=lw, label=LABELS[method], zorder=zo)
    ax1.fill_between(rounds, ma - sa, ma + sa, color=COLORS[method], alpha=0.10)
    c  = curves_data[method]
    mc = np.nanmean(c, axis=0); sc = np.nanstd(c, axis=0)
    ax2.plot(rounds, mc, color=COLORS[method], marker=MARKERS[method],
             markevery=5, markersize=5, linewidth=lw, label=LABELS[method], zorder=zo)
    ax2.fill_between(rounds, mc - sc, mc + sc, color=COLORS[method], alpha=0.10)

for ax, ylabel, title in [
    (ax1, 'Accuracy (%)', 'Accuracy'),
    (ax2, 'F1-Macro (%)', 'F1-Macro (rare-class sensitive)'),
]:
    ax.axvline(x=20, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xlabel('Communication Round', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=12)
    ax.set_xlim(1, N_ROUNDS); ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2, loc='lower right')
ax2.axhline(y=82, color='gray', linestyle=':', linewidth=1, alpha=0.7)

plt.suptitle('SOTA Comparison — Accuracy vs. F1-Macro, 20c Dir(0.1), 15 seeds\n'
             '(AO/GWO/PSO: 3-term fitness)', fontsize=12)
plt.tight_layout()
out = os.path.join(OUT_DIR, 'accuracy_vs_f1_dual_panel.png')
plt.savefig(out, dpi=150); plt.close()
print(f"Saved: {out}")

print("\nAll 3-term SOTA comparison plots generated.")
