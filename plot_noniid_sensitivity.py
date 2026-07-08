"""
Non-IID sensitivity figure: two-panel.
  Left  — robustness lines: Peak F1-macro vs heterogeneity level (Dir_0.5→Dir_0.1→Dir_0.05)
  Right — convergence failure heatmap: fail/5 seeds per (method × distribution) cell

Output: paper/figures/noniid_sensitivity.png  (double-column, figure*)
"""

import os, csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE         = r'C:\Users\Dspike\Documents\FL-AdroidMaLD'
RESULTS_BASE = os.path.join(BASE, 'results', 'noniid_sensitivity')
OUT          = os.path.join(BASE, 'paper', 'figures', 'noniid_sensitivity.png')

# ── Data config ───────────────────────────────────────────────────────────────
SEEDS  = [1, 2, 3, 42, 123]
MAX_R  = 50
THR    = 0.82

# x-axis: left = easier (moderate), right = harder (extreme)
DISTS       = ['Dir_0.5',          'Dir_0.1',                'Dir_0.05']
DIST_LABELS = [r'Dir($\alpha$=0.5)', r'Dir($\alpha$=0.1)', r'Dir($\alpha$=0.05)']
DIST_SUB    = ['Moderate',          'Severe\n(primary)',     'Extreme']

# Methods — same order as tab:sota for consistency; divfl8 appended
METHODS = ['aoa8', 'gwo8', 'pso8', 'random8', 'poco8', 'divfl8', 'oort8']
LABELS  = {
    'aoa8':    'AO-8 (ours)',
    'gwo8':    'GWO-8',
    'pso8':    'PSO-8',
    'random8': 'Random-8',
    'poco8':   'PoCo-8',
    'divfl8':  'DivFL-8',
    'oort8':   'Oort-8',
}

# Colors: reuse established palette; add divfl8
COLORS = {
    'aoa8':    '#E53935',
    'gwo8':    '#4CAF50',
    'pso8':    '#00BCD4',
    'random8': '#2196F3',
    'poco8':   '#FF9800',
    'divfl8':  '#795548',
    'oort8':   '#9C27B0',
}
MARKERS = {
    'aoa8':    'o',
    'gwo8':    'P',
    'pso8':    'X',
    'random8': '^',
    'poco8':   'v',
    'divfl8':  's',
    'oort8':   'D',
}


# ── Loader ────────────────────────────────────────────────────────────────────
def load(method, dist):
    f1s, rds = [], []
    for s in SEEDS:
        p = os.path.join(RESULTS_BASE, dist, method, f'seed{s}',
                         f'{method}_seed{s}_rounds.csv')
        if not os.path.exists(p):
            continue
        best, hit = 0.0, None
        with open(p) as f:
            for row in csv.DictReader(f):
                v = float(row['f1_macro'])
                if v > best:
                    best = v
                if row['phase'] == 'phase2' and v >= THR and hit is None:
                    hit = int(row['round'])
        f1s.append(best * 100)
        rds.append(hit if hit else MAX_R)
    return np.array(f1s), np.array(rds)


# ── Collect ───────────────────────────────────────────────────────────────────
data = {m: {d: load(m, d) for d in DISTS} for m in METHODS}

# ── Figure ────────────────────────────────────────────────────────────────────
fig, ax1 = plt.subplots(figsize=(9, 5))

x = np.arange(len(DISTS))

# ── Robustness line plot ──────────────────────────────────────────────────────
for m in METHODS:
    means, stds = [], []
    for d in DISTS:
        f1, _ = data[m][d]
        means.append(f1.mean() if len(f1) else np.nan)
        stds.append(f1.std()  if len(f1) else np.nan)
    means, stds = np.array(means), np.array(stds)
    lw = 2.5 if m == 'aoa8' else 1.6
    zo = 10  if m == 'aoa8' else 5
    ax1.plot(x, means, color=COLORS[m], label=LABELS[m],
             lw=lw, marker=MARKERS[m], markersize=7, zorder=zo)
    ax1.fill_between(x, means - stds, means + stds,
                     color=COLORS[m], alpha=0.10, zorder=zo - 1)

# Mark primary evaluation setting
ax1.axvline(x=1, color='black', ls='--', lw=1.0, alpha=0.5)
ax1.text(1.05, 88.5, 'primary\nsetting', fontsize=8, color='black',
         alpha=0.65, va='top')

ax1.set_xticks(x)
ax1.set_xticklabels(
    [f'{DIST_LABELS[i]}\n{DIST_SUB[i]}' for i in range(len(DISTS))],
    fontsize=10
)
ax1.set_ylabel('Peak F1-macro (%)', fontsize=12)
ax1.set_title('Non-IID Sensitivity — Peak F1-macro across heterogeneity levels\n'
              '20 clients, k=8, 5 seeds per cell', fontsize=11)
ax1.set_ylim(77, 90)
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=9, loc='lower left', ncol=1, framealpha=0.88)

plt.savefig(OUT, dpi=150, bbox_inches='tight')
print(f'Saved: {OUT}')
