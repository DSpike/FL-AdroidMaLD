"""
Summary comparison plots for the FL-AdroidMaLD paper.

Generates:
  1. convergence_curves.png  — mean ± std F1-macro per round across 15 seeds
  2. bar_peak_f1.png         — peak F1 bar chart with error bars
  3. bar_convergence.png     — rounds-to-threshold bar chart (75/78/80/82%)
  4. ablation_lam.png        — λ sweep effect on convergence
  5. ablation_delta.png      — δ sweep effect on peak F1

Usage:
  python plot_comparison.py              # uses optimised results
  python plot_comparison.py --dir old    # uses original (λ=0.2) results
"""

import os, csv, argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

parser = argparse.ArgumentParser()
parser.add_argument('--dir', choices=['optimized', 'original'], default='optimized')
args = parser.parse_args()

BASE = r'C:\Users\Dspike\Documents\FL-AdroidMaLD'
if args.dir == 'optimized':
    RESULTS_BASE = os.path.join(BASE, 'results', '20c_dir01_optimized')
    OUT_DIR      = os.path.join(BASE, 'plots', 'optimized')
    TITLE_TAG    = 'Optimised AOA (λ=0.0, δ=0.3)'
else:
    RESULTS_BASE = os.path.join(BASE, 'results', '20c_dir01_5seed')
    OUT_DIR      = os.path.join(BASE, 'plots', 'original')
    TITLE_TAG    = 'Original AOA (λ=0.2, δ=0.3)'

os.makedirs(OUT_DIR, exist_ok=True)

ALL_SEEDS  = [42, 123, 456, 789, 2024, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
METHODS    = ['fedavg_all', 'random8', 'aoa8']
LABELS     = {'fedavg_all': 'FedAvg-all', 'random8': 'Random-8', 'aoa8': 'AOA-8 (ours)'}
COLORS     = {'fedavg_all': '#888888',    'random8': '#2196F3',   'aoa8': '#E53935'}
MARKERS    = {'fedavg_all': 's',          'random8': '^',          'aoa8': 'o'}
N_ROUNDS   = 50
MAX_ROUNDS = 50
THRESHOLDS = [0.75, 0.78, 0.80, 0.82]

# ── Load data ─────────────────────────────────────────────────────────────────
def load_curves(method, seeds):
    """Returns (n_seeds, n_rounds) F1 array aligned by round."""
    curves = []
    for s in seeds:
        p = os.path.join(RESULTS_BASE, method, f'seed{s}',
                         f'{method}_seed{s}_rounds.csv')
        if not os.path.exists(p):
            continue
        rounds_f1 = {}
        with open(p) as f:
            for row in csv.DictReader(f):
                rounds_f1[int(row['round'])] = float(row['f1_macro']) * 100
        curve = [rounds_f1.get(r, np.nan) for r in range(1, N_ROUNDS + 1)]
        curves.append(curve)
    return np.array(curves) if curves else np.empty((0, N_ROUNDS))

def load_best_f1(method, seeds):
    bests = []
    for s in seeds:
        p = os.path.join(RESULTS_BASE, method, f'seed{s}',
                         f'{method}_seed{s}_rounds.csv')
        if os.path.exists(p):
            with open(p) as f:
                bests.append(max(float(r['f1_macro']) for r in csv.DictReader(f)) * 100)
    return np.array(bests)

def load_rounds_to(method, seeds, thr):
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

curves_data = {m: load_curves(m, ALL_SEEDS) for m in METHODS}
f1_data     = {m: load_best_f1(m, ALL_SEEDS) for m in METHODS}

# Check we have data
available = [m for m in METHODS if len(curves_data[m]) > 0]
if not available:
    print("No data found — run experiments first.")
    exit(0)
print(f"Methods with data: {available}")

# ── Plot 1: Convergence curves ────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
rounds = np.arange(1, N_ROUNDS + 1)

for method in available:
    c = curves_data[method]
    mean = np.nanmean(c, axis=0)
    std  = np.nanstd(c, axis=0)
    ax.plot(rounds, mean, color=COLORS[method], marker=MARKERS[method],
            markevery=5, linewidth=2, label=LABELS[method])
    ax.fill_between(rounds, mean - std, mean + std,
                    color=COLORS[method], alpha=0.15)

ax.axvline(x=20, color='black', linestyle='--', linewidth=1, alpha=0.5)
ax.text(20.5, ax.get_ylim()[0] + 2, 'Phase 2\nstart', fontsize=8, color='black')
ax.axhline(y=82, color='gray', linestyle=':', linewidth=1, alpha=0.7)
ax.text(1, 82.5, '82% threshold', fontsize=8, color='gray')

ax.set_xlabel('Communication Round', fontsize=12)
ax.set_ylabel('F1-Macro (%)', fontsize=12)
ax.set_title(f'Convergence — 20 clients, Dir(0.1), 15 seeds\n{TITLE_TAG}', fontsize=11)
ax.legend(fontsize=10)
ax.set_xlim(1, N_ROUNDS)
ax.set_ylim(0, 100)
ax.grid(True, alpha=0.3)
plt.tight_layout()
out = os.path.join(OUT_DIR, 'convergence_curves.png')
plt.savefig(out, dpi=150)
plt.close()
print(f"Saved: {out}")

# ── Plot 2: Peak F1 bar chart ─────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 4))
x = np.arange(len(available))
for i, method in enumerate(available):
    f1 = f1_data[method]
    ax.bar(i, f1.mean(), yerr=f1.std(), color=COLORS[method],
           capsize=5, width=0.5, label=LABELS[method], alpha=0.85, edgecolor='black')
    ax.text(i, f1.mean() + f1.std() + 0.3, f'{f1.mean():.2f}%',
            ha='center', va='bottom', fontsize=9, fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels([LABELS[m] for m in available], fontsize=10)
ax.set_ylabel('Peak F1-Macro (%)', fontsize=11)
ax.set_title(f'Peak F1-Macro — 20c Dir(0.1), 15 seeds\n{TITLE_TAG}', fontsize=10)
ax.set_ylim(75, 92)
ax.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
out = os.path.join(OUT_DIR, 'bar_peak_f1.png')
plt.savefig(out, dpi=150)
plt.close()
print(f"Saved: {out}")

# ── Plot 3: Rounds-to-threshold grouped bar chart ─────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
n_thr = len(THRESHOLDS)
width = 0.22
x     = np.arange(n_thr)

for i, method in enumerate(available):
    means, stds = [], []
    for thr in THRESHOLDS:
        rd = load_rounds_to(method, ALL_SEEDS, thr)
        reached = rd[rd < MAX_ROUNDS]
        means.append(reached.mean() if len(reached) else MAX_ROUNDS)
        stds.append(reached.std() if len(reached) > 1 else 0)
    offset = (i - len(available) / 2 + 0.5) * width
    ax.bar(x + offset, means, width, yerr=stds, label=LABELS[method],
           color=COLORS[method], capsize=4, alpha=0.85, edgecolor='black')

ax.set_xticks(x)
ax.set_xticklabels([f'{int(t*100)}%' for t in THRESHOLDS], fontsize=11)
ax.set_xlabel('F1-Macro Threshold', fontsize=11)
ax.set_ylabel('Rounds to Reach Threshold', fontsize=11)
ax.set_title(f'Convergence Speed — 20c Dir(0.1), 15 seeds\n{TITLE_TAG}', fontsize=10)
ax.legend(fontsize=10)
ax.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
out = os.path.join(OUT_DIR, 'bar_convergence.png')
plt.savefig(out, dpi=150)
plt.close()
print(f"Saved: {out}")

# ── Plot 4: λ ablation ────────────────────────────────────────────────────────
ABL_BASE  = os.path.join(BASE, 'results', 'ablation_aoa')
ABL_SEEDS = [42, 123, 456, 789, 2024]
LAM_CONFIGS = [
    (0.0, 0.3, 'lam0.0_del0.3'),
    (0.1, 0.3, 'lam0.1_del0.3'),
    (0.2, 0.3, None),            # baseline from original results
    (0.4, 0.3, 'lam0.4_del0.3'),
]
BL_BASE = os.path.join(BASE, 'results', '20c_dir01_5seed', 'aoa8')

lam_vals, lam_rds, lam_f1s = [], [], []
for lam, delta, cfg in LAM_CONFIGS:
    rds, f1s = [], []
    seeds_to_use = ABL_SEEDS
    if cfg is None:
        d = BL_BASE; prefix = 'aoa8'
    else:
        d = os.path.join(ABL_BASE, cfg); prefix = 'aoa8'
    for s in seeds_to_use:
        p = os.path.join(d, f'seed{s}', f'{prefix}_seed{s}_rounds.csv')
        if not os.path.exists(p): continue
        best, hit = 0.0, None
        with open(p) as f:
            for row in csv.DictReader(f):
                v = float(row['f1_macro'])
                if v > best: best = v
                if row['phase'] == 'phase2' and v >= 0.82 and hit is None:
                    hit = int(row['round'])
        f1s.append(best * 100); rds.append(hit if hit else MAX_ROUNDS)
    lam_vals.append(lam)
    lam_rds.append(np.array(rds))
    lam_f1s.append(np.array(f1s))

if any(len(r) > 0 for r in lam_rds):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    x = np.arange(len(lam_vals))
    labels = [str(l) for l in lam_vals]

    means_r = [r[r < MAX_ROUNDS].mean() if (r < MAX_ROUNDS).any() else MAX_ROUNDS for r in lam_rds]
    stds_r  = [r[r < MAX_ROUNDS].std()  if (r < MAX_ROUNDS).sum() > 1 else 0 for r in lam_rds]
    means_f = [f.mean() if len(f) else 0 for f in lam_f1s]
    stds_f  = [f.std()  if len(f) > 1 else 0 for f in lam_f1s]

    ax1.bar(x, means_r, yerr=stds_r, capsize=5, color='#5C6BC0', alpha=0.85, edgecolor='black')
    ax1.set_xticks(x); ax1.set_xticklabels(labels)
    ax1.set_xlabel('λ (diversity weight)'); ax1.set_ylabel('Rounds → 82%')
    ax1.set_title('Effect of λ on Convergence Speed'); ax1.grid(True, axis='y', alpha=0.3)

    ax2.bar(x, means_f, yerr=stds_f, capsize=5, color='#26A69A', alpha=0.85, edgecolor='black')
    ax2.set_xticks(x); ax2.set_xticklabels(labels)
    ax2.set_xlabel('λ (diversity weight)'); ax2.set_ylabel('Peak F1-Macro (%)')
    ax2.set_title('Effect of λ on Peak F1'); ax2.grid(True, axis='y', alpha=0.3)
    ax2.set_ylim(80, 88)

    plt.suptitle('λ Ablation (δ=0.3 fixed)', fontsize=11)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, 'ablation_lam.png')
    plt.savefig(out, dpi=150); plt.close()
    print(f"Saved: {out}")

# ── Plot 5: δ ablation ────────────────────────────────────────────────────────
DELTA_CONFIGS = [
    (0.2, 0.0, 'lam0.2_del0.0'),
    (0.2, 0.1, 'lam0.2_del0.1'),
    (0.2, 0.3, None),
    (0.2, 0.5, 'lam0.2_del0.5'),
]

del_vals, del_rds, del_f1s = [], [], []
for lam, delta, cfg in DELTA_CONFIGS:
    rds, f1s = [], []
    if cfg is None:
        d = BL_BASE; prefix = 'aoa8'
    else:
        d = os.path.join(ABL_BASE, cfg); prefix = 'aoa8'
    for s in ABL_SEEDS:
        p = os.path.join(d, f'seed{s}', f'{prefix}_seed{s}_rounds.csv')
        if not os.path.exists(p): continue
        best, hit = 0.0, None
        with open(p) as f:
            for row in csv.DictReader(f):
                v = float(row['f1_macro'])
                if v > best: best = v
                if row['phase'] == 'phase2' and v >= 0.82 and hit is None:
                    hit = int(row['round'])
        f1s.append(best * 100); rds.append(hit if hit else MAX_ROUNDS)
    del_vals.append(delta)
    del_rds.append(np.array(rds))
    del_f1s.append(np.array(f1s))

if any(len(r) > 0 for r in del_rds):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    x = np.arange(len(del_vals))
    labels = [str(d) for d in del_vals]

    means_r = [r[r < MAX_ROUNDS].mean() if (r < MAX_ROUNDS).any() else MAX_ROUNDS for r in del_rds]
    stds_r  = [r[r < MAX_ROUNDS].std()  if (r < MAX_ROUNDS).sum() > 1 else 0 for r in del_rds]
    means_f = [f.mean() if len(f) else 0 for f in del_f1s]
    stds_f  = [f.std()  if len(f) > 1 else 0 for f in del_f1s]

    ax1.bar(x, means_r, yerr=stds_r, capsize=5, color='#5C6BC0', alpha=0.85, edgecolor='black')
    ax1.set_xticks(x); ax1.set_xticklabels(labels)
    ax1.set_xlabel('δ (error-rate weight)'); ax1.set_ylabel('Rounds → 82%')
    ax1.set_title('Effect of δ on Convergence Speed'); ax1.grid(True, axis='y', alpha=0.3)

    ax2.bar(x, means_f, yerr=stds_f, capsize=5, color='#26A69A', alpha=0.85, edgecolor='black')
    ax2.set_xticks(x); ax2.set_xticklabels(labels)
    ax2.set_xlabel('δ (error-rate weight)'); ax2.set_ylabel('Peak F1-Macro (%)')
    ax2.set_title('Effect of δ on Peak F1'); ax2.grid(True, axis='y', alpha=0.3)
    ax2.set_ylim(80, 88)

    plt.suptitle('δ Ablation (λ=0.2 fixed)', fontsize=11)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, 'ablation_delta.png')
    plt.savefig(out, dpi=150); plt.close()
    print(f"Saved: {out}")

print("\nAll plots generated.")
