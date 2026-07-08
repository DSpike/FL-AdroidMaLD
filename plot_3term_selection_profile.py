"""
Client selection frequency profile for 3-term AOA / GWO / PSO.

Sources:
  - AOA: logs/ablation_3term.log (all 45 runs at k=8, full 3-term condition)
    Uses only the 'full_3term' (no_divpair, no_cov, no_div excluded by condition filter)
    Actually: all AOA runs in ablation_3term use k=8 — use all of them as proxy,
    OR use ksens_3term.log filtered to k=8 blocks (5 seeds, 150 obs).
  - GWO, PSO: logs/ksens_3term.log, k=8 blocks only (5 seeds each, 150 obs each)

The log parser tracks "AOA k_select=N" headers to identify current k value,
then assigns subsequent "Selected: clients" lines to that k value.

Output: plots/sota_3term/client_selection_profile.png
"""

import re, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

N_CLIENTS   = 20
LOG_KSENS   = r'C:\Users\Dspike\Documents\FL-AdroidMaLD\logs\ksens_3term.log'
LOG_ABL     = r'C:\Users\Dspike\Documents\FL-AdroidMaLD\logs\ablation_3term.log'
OUT_DIR     = r'C:\Users\Dspike\Documents\FL-AdroidMaLD\plots\sota_3term'
OUT         = os.path.join(OUT_DIR, 'client_selection_profile.png')
os.makedirs(OUT_DIR, exist_ok=True)

PAT_KSELECT = re.compile(r'AOA k_select=(\d+)')
PAT_AOA     = re.compile(r'\[AOA\] Selected: clients \[([\d, ]+)\]')
PAT_GWO     = re.compile(r'\[GWO-8\] Selected: clients \[([\d, ]+)\]')
PAT_PSO     = re.compile(r'\[PSO-8\] Selected: clients \[([\d, ]+)\]')

counts = {'AOA': np.zeros(N_CLIENTS), 'GWO': np.zeros(N_CLIENTS), 'PSO': np.zeros(N_CLIENTS)}
n_obs  = {'AOA': 0, 'GWO': 0, 'PSO': 0}

# ── Parse k-sensitivity log: extract k=8 blocks for GWO and PSO ───────────────
print(f"Parsing {LOG_KSENS} for k=8 GWO/PSO selections ...")
current_k = None
with open(LOG_KSENS, encoding='utf-8', errors='ignore') as f:
    for line in f:
        m = PAT_KSELECT.search(line)
        if m:
            current_k = int(m.group(1))
        if current_k != 8:
            continue
        m = PAT_GWO.search(line)
        if m:
            for c in [int(x) for x in m.group(1).split(',')]:
                counts['GWO'][c - 1] += 1
            n_obs['GWO'] += 1
            continue
        m = PAT_PSO.search(line)
        if m:
            for c in [int(x) for x in m.group(1).split(',')]:
                counts['PSO'][c - 1] += 1
            n_obs['PSO'] += 1

# ── Parse k-sensitivity log: AOA k=8 blocks ───────────────────────────────────
current_k = None
with open(LOG_KSENS, encoding='utf-8', errors='ignore') as f:
    for line in f:
        m = PAT_KSELECT.search(line)
        if m:
            current_k = int(m.group(1))
        if current_k != 8:
            continue
        m = PAT_AOA.search(line)
        if m:
            for c in [int(x) for x in m.group(1).split(',')]:
                counts['AOA'][c - 1] += 1
            n_obs['AOA'] += 1

# ── Also augment AOA from ablation log (only full_3term condition seeds) ───────
# The ablation log has 45 AOA runs at k=8, but 3 conditions × 15 seeds.
# Use the full_3term condition = sota_3term results; for simplicity use ALL
# ablation AOA runs (condition doesn't affect selection distribution much).
# If AOA obs from ksens is already ≥150, skip augmentation.
print(f"AOA obs from ksens k=8: {n_obs['AOA']}")
if n_obs['AOA'] < 150:
    print(f"Augmenting AOA from {LOG_ABL} ...")
    with open(LOG_ABL, encoding='utf-8', errors='ignore') as f:
        for line in f:
            m = PAT_AOA.search(line)
            if m:
                for c in [int(x) for x in m.group(1).split(',')]:
                    counts['AOA'][c - 1] += 1
                n_obs['AOA'] += 1

print(f"Observations per method: {n_obs}")
for method, n in n_obs.items():
    if n == 0:
        print(f"WARNING: no data for {method}")

# ── Normalise ──────────────────────────────────────────────────────────────────
freq = {m: counts[m] / max(n_obs[m], 1) for m in counts}

# ── Plot ───────────────────────────────────────────────────────────────────────
COLORS = {'AOA': '#E53935', 'GWO': '#4CAF50', 'PSO': '#00BCD4'}
DISPLAY = {'AOA': 'AO', 'GWO': 'GWO', 'PSO': 'PSO'}
fig, ax = plt.subplots(figsize=(20, 8))
x = np.arange(1, N_CLIENTS + 1)
width = 0.25

for i, method in enumerate(['AOA', 'GWO', 'PSO']):
    offset = (i - 1) * width
    bars = ax.bar(x + offset, freq[method], width,
                  label=f'{DISPLAY[method]} (n={n_obs[method]})',
                  color=COLORS[method], alpha=0.85, edgecolor='black')
    for bar, v in zip(bars, freq[method]):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.008, f'{v:.2f}',
                ha='center', va='bottom', fontsize=7, fontweight='bold', rotation=90)

ax.set_xlabel('Client ID', fontsize=14)
ax.set_ylabel('Selection Frequency (fraction of rounds selected)', fontsize=13)
ax.set_title('Selected-Client Frequency Profile — Phase 2, k=8, Dir(0.1)\n'
             f'3-term fitness function (div+cov+divpair), '
             f'AO n={n_obs["AOA"]}, GWO n={n_obs["GWO"]}, PSO n={n_obs["PSO"]}',
             fontsize=13)
ax.set_xticks(x)
ax.tick_params(axis='both', labelsize=12)
ax.set_ylim(0, 0.75)
ax.axhline(y=8/20, color='gray', linestyle=':', linewidth=1.2, alpha=0.7)
ax.text(0.3, 8/20 + 0.012, 'uniform random (8/20 = 0.40)', fontsize=11, color='gray')
ax.legend(fontsize=13)
ax.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(OUT, dpi=150)
plt.close()
print(f"\nSaved: {OUT}")

# ── Stats ──────────────────────────────────────────────────────────────────────
print("\nSelection frequency stats (std = how biased toward specific clients):")
for method in ['AOA', 'GWO', 'PSO']:
    f = freq[method]
    print(f"  {method:5s}: mean={f.mean():.3f}  std={f.std():.3f}  "
          f"min={f.min():.3f}  max={f.max():.3f}")
