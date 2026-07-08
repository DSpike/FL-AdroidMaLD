"""
Run DivFL baseline head-to-head against AOA at k=8, 15 seeds.

DivFL selects a diverse subset via greedy facility location over gradient
proxy vectors (model weight differences), matching Balakrishnan et al. (2022).

Results land in: results/sota_3term/divfl8/seed{N}/
Baselines (AOA, Oort, PoCo) are loaded from the existing sota_3term cache.

Usage:
    python run_divfl_comparison.py
    python run_divfl_comparison.py --dry_run    # print commands only
"""

import argparse
import subprocess, sys, os, csv
import numpy as np
from scipy import stats
from scipy.stats import norm

parser = argparse.ArgumentParser()
parser.add_argument('--dry_run', action='store_true',
                    help='Print commands without executing')
args = parser.parse_args()

BASE         = r"C:\Users\Dspike\Documents\FL-AdroidMaLD"
SCRIPT       = os.path.join(BASE, 'run_experiment_20c_dir01.py')
RESULTS_BASE = os.path.join(BASE, 'results', 'sota_3term')
SEEDS        = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 42, 123, 456, 789, 2024]
MAX_ROUNDS   = 50
THR_82       = 0.82

total = len(SEEDS)
done, failed = 0, []

print(f"\n{'='*70}")
print(f" DivFL Baseline — k=8, Dir(0.1), 15 seeds")
print(f" Greedy facility location over gradient proxy vectors")
print(f"{'='*70}\n")

for seed in SEEDS:
    done += 1
    rdir     = os.path.join(RESULTS_BASE, 'divfl8', f'seed{seed}')
    csv_path = os.path.join(rdir, f'divfl8_seed{seed}_rounds.csv')

    if os.path.exists(csv_path):
        print(f'[{done}/{total}]  divfl8  seed={seed}  (cached)')
        continue

    print(f'\n{"="*60}')
    print(f' [{done}/{total}]  divfl8  seed={seed}')
    print(f'{"="*60}')

    cmd = [sys.executable, SCRIPT,
           '--method',      'divfl8',
           '--seed',        str(seed),
           '--k_select',    '8',
           '--results_dir', rdir]

    if args.dry_run:
        print('  DRY RUN:', ' '.join(cmd))
        continue

    result = subprocess.run(cmd, cwd=BASE)
    if result.returncode != 0:
        print(f'  *** FAILED seed={seed} ***')
        failed.append(seed)


# ── Analysis ────────────────────────────────────────────────────────────────

def load(base, method, seeds):
    f1s, rds = [], []
    for s in seeds:
        p = os.path.join(base, method, f'seed{s}', f'{method}_seed{s}_rounds.csv')
        if not os.path.exists(p):
            continue
        best, hit = 0.0, None
        with open(p) as f:
            for row in csv.DictReader(f):
                v = float(row['f1_macro'])
                if v > best:
                    best = v
                if row['phase'] == 'phase2' and v >= THR_82 and hit is None:
                    hit = int(row['round'])
        f1s.append(best * 100)
        rds.append(hit if hit else MAX_ROUNDS)
    return np.array(f1s), np.array(rds)


def wilcoxon_vs(aoa_rd, other_rd):
    """One-sided Wilcoxon: AOA rounds < other rounds (AOA faster)."""
    try:
        stat, p = stats.wilcoxon(aoa_rd, other_rd, alternative='less')
        r = 1.0 - 2.0 * stat / (len(aoa_rd) * (len(aoa_rd) + 1) / 2)
        z = norm.ppf(1 - p) if p < 1 else 0.0
        d = z / np.sqrt(len(aoa_rd))
        return p, r, d
    except Exception:
        return 1.0, 0.0, 0.0


def sig(p):
    return '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'n.s.'


print(f'\n\n{"="*80}')
print(f' HEAD-TO-HEAD: DivFL vs AOA (k=8, Dir(0.1), up to 15 seeds)')
print(f'{"="*80}\n')

# Load results
aoa_f1,    aoa_rd    = load(RESULTS_BASE, 'aoa8',    SEEDS)
divfl_f1,  divfl_rd  = load(RESULTS_BASE, 'divfl8',  SEEDS)
oort_f1,   oort_rd   = load(RESULTS_BASE, 'oort8',   SEEDS)
poco_f1,   poco_rd   = load(RESULTS_BASE, 'poco8',   SEEDS)

header = f'  {"Method":12s}  {"n":>3}  {"PeakF1":>8}  {"Rds82%":>7}  {"Never":>6}  {"p(vs AOA)":>10}  {"r":>6}  {"d":>6}'
print(header)
print('  ' + '-'*78)

for method, f1, rd in [
    ('AOA (ours)',  aoa_f1,   aoa_rd),
    ('DivFL',      divfl_f1, divfl_rd),
    ('Oort',       oort_f1,  oort_rd),
    ('PoCo',       poco_f1,  poco_rd),
]:
    n_res = len(rd)
    if n_res == 0:
        print(f'  {method:12s}  {"no data":>3}')
        continue
    nev = int((rd == MAX_ROUNDS).sum())
    rm  = rd[rd < MAX_ROUNDS].mean() if (rd < MAX_ROUNDS).any() else MAX_ROUNDS

    if method == 'AOA (ours)':
        print(f'  {method:12s}  {n_res:>3}  {f1.mean():7.2f}%  {rm:6.1f}   {nev:3d}/{n_res:<3}  '
              f'{"---":>10}  {"---":>6}  {"---":>6}  <- proposed')
    else:
        n_common = min(len(aoa_rd), len(rd))
        p, r, d = wilcoxon_vs(aoa_rd[:n_common], rd[:n_common])
        print(f'  {method:12s}  {n_res:>3}  {f1.mean():7.2f}%  {rm:6.1f}   {nev:3d}/{n_res:<3}  '
              f'{p:8.4f}{sig(p):3s}  {r:6.3f}  {d:6.3f}')

if failed:
    print(f'\nFailed seeds: {failed}')
else:
    print(f'\nAll {total} runs completed or cached.')
