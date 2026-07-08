import os, csv
import numpy as np

BASE    = r'C:\Users\Dspike\Documents\FL-AdroidMaLD\results\ablation_aoa'
BL_BASE = r'C:\Users\Dspike\Documents\FL-AdroidMaLD\results\20c_dir01_5seed\aoa8'
SEEDS   = [42, 123, 456, 789, 2024]
CONFIGS = ['lam0.0_del0.3','lam0.1_del0.3','lam0.4_del0.3',
           'lam0.2_del0.0','lam0.2_del0.1','lam0.2_del0.5']

def read(dirpath, prefix, seeds):
    f1s, rds = [], []
    for s in seeds:
        p = os.path.join(dirpath, f'seed{s}', f'{prefix}_seed{s}_rounds.csv')
        if not os.path.exists(p):
            continue
        best, hit = 0.0, None
        with open(p) as f:
            for row in csv.DictReader(f):
                v = float(row['f1_macro'])
                if v > best:
                    best = v
                if row['phase'] == 'phase2' and v >= 0.82 and hit is None:
                    hit = int(row['round'])
        f1s.append(best * 100)
        rds.append(hit if hit else 50)
    return np.array(f1s), np.array(rds)

print(f"{'Config':25s}  {'F1 mean':>8s}  {'Rds->82%':>9s}  {'Never':>6s}")
f1, rd = read(BL_BASE, 'aoa8', SEEDS)
if len(f1):
    r = rd[rd < 50]; nev = int((rd == 50).sum())
    print(f"{'lam0.2_del0.3 [BASE]':25s}  {f1.mean():>7.2f}%  {r.mean():>8.1f}   {nev}/5")

for cfg in CONFIGS:
    d = os.path.join(BASE, cfg)
    f1, rd = read(d, 'aoa8', SEEDS)
    if len(f1):
        r = rd[rd < 50]; nev = int((rd == 50).sum())
        rm = r.mean() if len(r) else 50.0
        tag = f' [{len(f1)}/5]' if len(f1) < 5 else ''
        print(f"{cfg:25s}  {f1.mean():>7.2f}%  {rm:>8.1f}   {nev}/5{tag}")
    else:
        print(f"{cfg:25s}  no data yet")
