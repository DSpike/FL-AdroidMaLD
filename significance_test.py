"""
Statistical significance tests on 15-seed convergence results.
Paired Wilcoxon signed-rank test (same seeds across methods → paired).
Effect size: rank-biserial correlation r.
"""

import os, csv
import numpy as np
from scipy import stats

ALL_SEEDS = [42, 123, 456, 789, 2024, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
METHODS   = ['fedavg_all', 'random8', 'aoa8']
BASE      = r'C:\Users\Dspike\Documents\FL-AdroidMaLD\results\20c_dir01_5seed'
THRESHOLDS = [0.75, 0.78, 0.80, 0.82]
MAX_ROUNDS = 50   # treat "never reached" as 50 (conservative for AOA)


def load_rounds_to_threshold(method, seeds, thr):
    """Returns list of rounds-to-thr for each seed (MAX_ROUNDS if never reached)."""
    values = []
    for seed in seeds:
        path = os.path.join(BASE, method, f'seed{seed}',
                            f'{method}_seed{seed}_rounds.csv')
        hit = None
        if os.path.exists(path):
            with open(path) as f:
                reader = csv.reader(f)
                next(reader)
                for row in reader:
                    if row[1] == 'phase2' and float(row[3]) >= thr:
                        hit = int(row[0])
                        break
        values.append(hit if hit is not None else MAX_ROUNDS)
    return np.array(values)


def rank_biserial(x, y):
    """Rank-biserial correlation for Wilcoxon (effect size, range -1 to +1)."""
    diff = x - y
    diff = diff[diff != 0]
    if len(diff) == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(diff))
    r_plus  = ranks[diff > 0].sum()
    r_minus = ranks[diff < 0].sum()
    n = len(diff)
    return (r_plus - r_minus) / (n * (n + 1) / 2)


def cohens_d(x, y):
    pooled_std = np.sqrt((np.std(x, ddof=1)**2 + np.std(y, ddof=1)**2) / 2)
    return (np.mean(x) - np.mean(y)) / pooled_std if pooled_std > 0 else 0.0


print("=" * 70)
print(" STATISTICAL SIGNIFICANCE — 15 seeds, 20 clients, Dir(0.1)")
print(" Paired Wilcoxon signed-rank test (same seeds → paired)")
print(" Never-reached seeds treated as MAX_ROUNDS=50 (conservative)")
print("=" * 70)

for thr in THRESHOLDS:
    fa  = load_rounds_to_threshold('fedavg_all', ALL_SEEDS, thr)
    rnd = load_rounds_to_threshold('random8',    ALL_SEEDS, thr)
    aoa = load_rounds_to_threshold('aoa8',       ALL_SEEDS, thr)

    print(f"\n── Threshold {thr*100:.0f}% ──────────────────────────────────────────────")
    print(f"  fedavg_all : {fa.mean():.1f} ± {fa.std():.1f}  "
          f"(never={int((fa==MAX_ROUNDS).sum())}/15)")
    print(f"  random8    : {rnd.mean():.1f} ± {rnd.std():.1f}  "
          f"(never={int((rnd==MAX_ROUNDS).sum())}/15)")
    print(f"  aoa8       : {aoa.mean():.1f} ± {aoa.std():.1f}  "
          f"(never={int((aoa==MAX_ROUNDS).sum())}/15)")

    # AOA vs Random (primary comparison)
    if np.all(aoa == rnd):
        print("  AOA vs Random: identical — no test needed")
    else:
        stat, p = stats.wilcoxon(aoa, rnd, alternative='less')  # AOA < Random?
        r       = rank_biserial(aoa, rnd)
        d       = cohens_d(rnd, aoa)  # positive d = random slower
        sig     = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
        print(f"\n  AOA vs Random-8:")
        print(f"    Δ rounds = {rnd.mean()-aoa.mean():.1f} fewer for AOA")
        print(f"    Wilcoxon W={stat:.0f}  p={p:.4f}  {sig}")
        print(f"    Effect size: r={r:.3f}  Cohen's d={d:.3f}")

    # AOA vs FedAvg-all (secondary)
    if not np.all(aoa == fa):
        stat2, p2 = stats.wilcoxon(aoa, fa, alternative='less')
        r2        = rank_biserial(aoa, fa)
        sig2      = "***" if p2 < 0.001 else "**" if p2 < 0.01 else "*" if p2 < 0.05 else "n.s."
        print(f"\n  AOA vs FedAvg-all:")
        print(f"    Δ rounds = {fa.mean()-aoa.mean():.1f} fewer for AOA")
        print(f"    Wilcoxon W={stat2:.0f}  p={p2:.4f}  {sig2}")
        print(f"    Effect size: r={r2:.3f}")

# ── Peak F1 significance ───────────────────────────────────────────────────────
print("\n" + "=" * 70)
print(" PEAK F1 SIGNIFICANCE")
print("=" * 70)

def load_best_f1(method, seeds):
    bests = []
    for seed in seeds:
        path = os.path.join(BASE, method, f'seed{seed}',
                            f'{method}_seed{seed}_rounds.csv')
        if os.path.exists(path):
            with open(path) as f:
                reader = csv.reader(f)
                next(reader)
                bests.append(max(float(row[3]) for row in reader) * 100)
    return np.array(bests)

fa_f1  = load_best_f1('fedavg_all', ALL_SEEDS)
rnd_f1 = load_best_f1('random8',    ALL_SEEDS)
aoa_f1 = load_best_f1('aoa8',       ALL_SEEDS)

for label, x, y in [('AOA vs Random-8',   aoa_f1, rnd_f1),
                     ('AOA vs FedAvg-all', aoa_f1, fa_f1)]:
    stat, p = stats.wilcoxon(x, y, alternative='greater')
    r       = rank_biserial(x, y)
    d       = cohens_d(x, y)
    sig     = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
    print(f"\n  {label}:")
    print(f"    {x.mean():.2f}% vs {y.mean():.2f}%  (Δ={x.mean()-y.mean():.2f}%)")
    print(f"    Wilcoxon W={stat:.0f}  p={p:.4f}  {sig}")
    print(f"    Effect size: r={r:.3f}  Cohen's d={d:.3f}")

print()
