"""
Regenerate bar_peak_f1.png for the paper using authoritative results directories.

Method → directory mapping:
  FedAvg-all, Random  → results/sota_comparison/
  Oort                → results/oort_proper/
  PoCo                → results/poco_proper/
  GWO, PSO, AO        → results/sota_3term/

Output: paper/figures/bar_peak_f1.png  (overwrites stale version)
"""

import os, csv, shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = r'C:\Users\Dspike\Documents\FL-AdroidMaLD'
SEEDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 42, 123, 456, 789, 2024]

# (method_key, results_subdir, csv_prefix)
METHOD_CONF = [
    ('fedavg_all', 'sota_comparison/fedavg_all', 'fedavg_all'),
    ('random8',    'sota_comparison/random8',    'random8'),
    ('oort8',      'oort_proper',                'oort8'),
    ('poco8',      'sota_comparison/poco8',       'poco8'),
    ('gwo8',       'sota_3term/gwo8',            'gwo8'),
    ('pso8',       'sota_3term/pso8',            'pso8'),
    ('aoa8',       'sota_3term/aoa8',            'aoa8'),
]

LABELS = {
    'fedavg_all': 'FedAvg-all', 'random8': 'Random',
    'oort8': 'Oort',            'poco8': 'PoCo',
    'gwo8': 'GWO',              'pso8': 'PSO',
    'aoa8': 'AO (ours)',
}
COLORS = {
    'fedavg_all': '#888888', 'random8': '#2196F3', 'oort8': '#9C27B0',
    'poco8': '#FF9800',      'gwo8': '#4CAF50',    'pso8': '#00BCD4',
    'aoa8': '#E53935',
}


def load_peak_f1(results_subdir, csv_prefix, seeds):
    bests = []
    for s in seeds:
        p = os.path.join(BASE, 'results', results_subdir,
                         f'seed{s}', f'{csv_prefix}_seed{s}_rounds.csv')
        if not os.path.exists(p):
            print(f'  [missing] {p}')
            continue
        with open(p) as f:
            vals = [float(r['f1_macro']) for r in csv.DictReader(f)]
        bests.append(max(vals) * 100)
    return np.array(bests)


print('Loading per-method peak F1 ...')
f1_data = {}
for key, subdir, prefix in METHOD_CONF:
    arr = load_peak_f1(subdir, prefix, SEEDS)
    f1_data[key] = arr
    n = len(arr)
    print(f'  {key:<12}  n={n}  mean={arr.mean():.2f}%  std={arr.std():.2f}%')

# ── Bar chart ─────────────────────────────────────────────────────────────────
methods = [k for k, _, _ in METHOD_CONF]
fig, ax = plt.subplots(figsize=(9, 5))

for i, key in enumerate(methods):
    f1 = f1_data[key]
    if len(f1) == 0:
        continue
    edge = 'black' if key != 'aoa8' else '#7a0000'
    lw   = 1       if key != 'aoa8' else 2
    ax.bar(i, f1.mean(), yerr=f1.std(), color=COLORS[key], capsize=5,
           width=0.6, label=LABELS[key], alpha=0.88, edgecolor=edge, linewidth=lw)
    ax.text(i, f1.mean() + f1.std() + 0.4,
            f'{f1.mean():.2f}%', ha='center', va='bottom',
            fontsize=9, fontweight='bold')

ax.set_xticks(range(len(methods)))
ax.set_xticklabels([LABELS[k] for k in methods], fontsize=9, rotation=20, ha='right')
ax.set_ylabel('Peak F1-Macro (%)', fontsize=11)
ax.set_title('Peak F1-Macro — 20 clients, k=8, Dir(0.1), 15 seeds', fontsize=11)
ax.set_ylim(75, 92)
ax.grid(True, axis='y', alpha=0.3)
plt.tight_layout()

# Save to plots/sota_3term/ and copy to paper/figures/
out_dir = os.path.join(BASE, 'plots', 'sota_3term')
os.makedirs(out_dir, exist_ok=True)
out_plot = os.path.join(out_dir, 'bar_peak_f1.png')
plt.savefig(out_plot, dpi=150)
plt.close()
print(f'\nSaved: {out_plot}')

fig_dst = os.path.join(BASE, 'paper', 'figures', 'bar_peak_f1.png')
shutil.copy2(out_plot, fig_dst)
print(f'Copied to: {fig_dst}')
print('\nDone — recompile main.tex to pick up the updated figure.')
