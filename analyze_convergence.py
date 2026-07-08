import os, csv
import numpy as np

SEEDS      = [42, 123, 456, 789, 2024]
METHODS    = ['fedavg_all', 'random8', 'aoa8']
BASE       = r'C:\Users\Dspike\Documents\FL-AdroidMaLD\results\20c_dir01_5seed'
THRESHOLDS = [0.75, 0.78, 0.80, 0.82]

print('=== CONVERGENCE SPEED — 20 clients, Dir(0.1), 5 seeds ===')
print('Metric: first phase-2 round where F1-macro >= threshold\n')

for thr in THRESHOLDS:
    print(f'Threshold {thr*100:.0f}%:')
    for method in METHODS:
        rounds_to_thr = []
        for seed in SEEDS:
            path = os.path.join(BASE, method, f'seed{seed}',
                                f'{method}_seed{seed}_rounds.csv')
            if not os.path.exists(path):
                continue
            hit = None
            with open(path) as f:
                reader = csv.reader(f)
                next(reader)
                for row in reader:
                    if row[1] == 'phase2' and float(row[3]) >= thr:
                        hit = int(row[0])
                        break
            rounds_to_thr.append(hit if hit is not None else 999)

        reached = [r for r in rounds_to_thr if r < 999]
        never   = len(rounds_to_thr) - len(reached)
        if reached:
            tag = f'  [never in {never}/5]' if never else ''
            print(f'  {method:12s}: {np.mean(reached):.1f} +/- {np.std(reached):.1f} rounds{tag}')
        else:
            print(f'  {method:12s}: never reached')
    print()

print('Best F1-macro (5 seeds):')
for method in METHODS:
    bests = []
    for seed in SEEDS:
        path = os.path.join(BASE, method, f'seed{seed}',
                            f'{method}_seed{seed}_rounds.csv')
        if os.path.exists(path):
            with open(path) as f:
                reader = csv.reader(f)
                next(reader)
                bests.append(max(float(row[3]) for row in reader) * 100)
    if bests:
        print(f'  {method:12s}: {np.mean(bests):.2f} +/- {np.std(bests):.2f}%')
