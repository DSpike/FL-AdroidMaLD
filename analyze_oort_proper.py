import os, csv
import numpy as np
from scipy.stats import wilcoxon

SEEDS    = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 42, 123, 456, 789, 2024]
MAX_R    = 50
THR      = 0.82
BASE     = r"C:\Users\Dspike\Documents\FL-AdroidMaLD"

def load(base, prefix):
    f1s, rds, fails = [], [], 0
    for s in SEEDS:
        p = os.path.join(base, f'seed{s}', f'{prefix}_seed{s}_rounds.csv')
        if not os.path.exists(p):
            print(f'  MISSING: {p}')
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
        if hit is None:
            fails += 1
            rds.append(MAX_R)
        else:
            rds.append(hit)
    return np.array(f1s), np.array(rds), fails

oort_f1, oort_rds, oort_fails = load(os.path.join(BASE, 'results', 'oort_proper'), 'oort8')
ao_f1,   ao_rds,   ao_fails   = load(os.path.join(BASE, 'results', 'sota_3term', 'aoa8'), 'aoa8')

conv_oort = oort_rds[oort_rds < MAX_R]
conv_ao   = ao_rds[ao_rds < MAX_R]
conv_oort_mean = conv_oort.mean() if len(conv_oort) else float('nan')

print(f"Oort(proper): n={len(oort_f1)}  F1={oort_f1.mean():.2f}+/-{oort_f1.std():.2f}%  "
      f"Fails={oort_fails}/15  ConvergedRds={conv_oort_mean:.1f}")
print(f"AO(ours):     n={len(ao_f1)}  F1={ao_f1.mean():.2f}+/-{ao_f1.std():.2f}%  "
      f"Fails={ao_fails}/15  ConvergedRds={conv_ao.mean():.1f}")
print()

print("Seed | Oort F1 | Oort rds | AO F1  | AO rds | Oort converged?")
for i, s in enumerate(SEEDS):
    of  = oort_f1[i]  if i < len(oort_f1)  else float('nan')
    or_ = oort_rds[i] if i < len(oort_rds) else MAX_R
    af  = ao_f1[i]    if i < len(ao_f1)    else float('nan')
    ar  = ao_rds[i]   if i < len(ao_rds)   else MAX_R
    conv = "YES" if or_ < MAX_R else "FAIL"
    print(f"{s:4d} | {of:6.2f}% | {or_:3d}      | {af:6.2f}% | {ar:3d}    | {conv}")

n = min(len(oort_rds), len(ao_rds))
if n >= 5:
    try:
        stat, p = wilcoxon(ao_rds[:n], oort_rds[:n], alternative='less')
        r = 1 - 2 * stat / (n * (n + 1))
        print(f"\nWilcoxon(AO < Oort_proper, rounds-to-82%): stat={stat:.1f}  p={p:.4f}  r={r:.3f}")
    except Exception as e:
        print(f"Wilcoxon: {e}")

print("\nDone.")
