import os, csv, numpy as np

BASE  = r'C:\Users\Dspike\Documents\FL-AdroidMaLD\results\20c_dir01_optimized'
SEEDS = [42,123,456,789,2024,1,2,3,4,5,6,7,8,9,10]
MAX   = 50

def load(method, seeds):
    f1s, rds = [], []
    for s in seeds:
        p = os.path.join(BASE, method, f'seed{s}', f'{method}_seed{s}_rounds.csv')
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
        rds.append(hit if hit else MAX)
    return np.array(f1s), np.array(rds)

print(f"{'Method':12s}  {'F1 mean':>9s}  {'Std':>6s}  {'Rds->82%':>9s}  {'Never':>7s}  {'n':>3s}")
for method in ['fedavg_all', 'random8', 'aoa8']:
    f1, rd = load(method, SEEDS)
    if len(f1) == 0:
        print(f'{method:12s}  no data yet')
        continue
    nev = int((rd == MAX).sum())
    rm  = rd[rd < MAX].mean() if (rd < MAX).any() else MAX
    print(f'{method:12s}  {f1.mean():>8.2f}%  {f1.std():>5.2f}%  {rm:>8.1f}   {nev:>4d}/{len(f1)}  {len(f1):>3d}')
