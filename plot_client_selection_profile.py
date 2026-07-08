"""
Selected-client-frequency profile for AOA / GWO / PSO (sota_resume.log).

Parses "[METHOD] Selected: clients [...]" lines logged during phase 2 of the
SOTA comparison run and plots how often each of the 20 clients was picked,
normalised by the number of (seed x round) observations per method.

Generates: plots/sota_comparison/client_selection_profile.png
"""

import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

LOGS = [r'C:\Users\Dspike\Documents\FL-AdroidMaLD\sota_resume.log',
        r'C:\Users\Dspike\Documents\FL-AdroidMaLD\gwo_missing_seeds.log']
OUT = r'C:\Users\Dspike\Documents\FL-AdroidMaLD\plots\sota_comparison\client_selection_profile.png'
N_CLIENTS = 20

PATTERNS = {
    'AOA':  re.compile(r'\[AOA\] Selected: clients \[([\d, ]+)\]'),
    'GWO':  re.compile(r'\[GWO-8\] Selected: clients \[([\d, ]+)\]'),
    'PSO':  re.compile(r'\[PSO-8\] Selected: clients \[([\d, ]+)\]'),
}
COLORS = {'AOA': '#E53935', 'GWO': '#4CAF50', 'PSO': '#00BCD4'}

counts = {m: np.zeros(N_CLIENTS) for m in PATTERNS}
n_obs  = {m: 0 for m in PATTERNS}

for log_path in LOGS:
    try:
        with open(log_path, encoding='utf-8', errors='ignore') as f:
            text = f.read()
    except FileNotFoundError:
        print(f'Warning: {log_path} not found, skipping')
        continue
    for method, pat in PATTERNS.items():
        for match in pat.finditer(text):
            clients = [int(x) for x in match.group(1).split(',')]
            for c in clients:
                counts[method][c - 1] += 1
            n_obs[method] += 1

print("Observations per method:", n_obs)

freq = {m: counts[m] / n_obs[m] for m in PATTERNS}

fig, ax = plt.subplots(figsize=(20, 8))
x = np.arange(1, N_CLIENTS + 1)
width = 0.25

for i, method in enumerate(PATTERNS):
    offset = (i - 1) * width
    bars = ax.bar(x + offset, freq[method], width, label=f'{method} (n={n_obs[method]})',
                   color=COLORS[method], alpha=0.85, edgecolor='black')
    for bar, v in zip(bars, freq[method]):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.012, f'{v:.2f}',
                ha='center', va='bottom', fontsize=7.5, fontweight='bold', rotation=90)

ax.set_xlabel('Client ID', fontsize=14)
ax.set_ylabel('Selection Frequency (fraction of rounds selected)', fontsize=13)
ax.set_title('Selected-Client Frequency Profile — Phase 2, k=8, Dir(0.1)\n'
             '(AOA, GWO, PSO: 15 seeds each)', fontsize=14)
ax.set_xticks(x)
ax.tick_params(axis='both', labelsize=12)
ax.set_ylim(0, 0.72)
ax.axhline(y=8/20, color='gray', linestyle=':', linewidth=1.2, alpha=0.7)
ax.text(0.3, 8/20 + 0.015, 'uniform random expectation (8/20=0.40)', fontsize=11, color='gray')
ax.legend(fontsize=13)
ax.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(OUT, dpi=150)
plt.close()
print(f"Saved: {OUT}")

# ── Quantify how "peaked" vs "flat" each method's selection is ───────────────
print("\nSelection frequency stats (std dev across clients — higher = more biased toward specific clients):")
for method in PATTERNS:
    f = freq[method]
    print(f"  {method:5s}: mean={f.mean():.3f}  std={f.std():.3f}  min={f.min():.3f}  max={f.max():.3f}")
