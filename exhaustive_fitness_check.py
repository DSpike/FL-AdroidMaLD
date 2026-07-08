"""
Exhaustive fitness sanity check: verify AO finds near-optimal F(S).

Runs ONE AO seed with stat logging, then at each phase-2 round computes:
  - AO's selected subset fitness
  - True optimal fitness (exhaustive search over all C(20,8)=125,970 subsets)
  - Greedy top-k fitness (linear score only, lam=0)

Reports: average optimality gap across all phase-2 rounds.

Usage: python exhaustive_fitness_check.py [--seed 42]
"""

import argparse, os, pickle
import numpy as np
from itertools import combinations

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42)
args = parser.parse_args()

STATS_FILE = rf"C:\Users\Dspike\Documents\FL-AdroidMaLD\results\exhaustive_check\seed{args.seed}_stats.pkl"
os.makedirs(os.path.dirname(STATS_FILE), exist_ok=True)

# ── Phase 1: run AO with stat logging ─────────────────────────────────────────
if not os.path.exists(STATS_FILE):
    print(f"Running AO seed={args.seed} with stat logging...")

    import torch
    import numpy as np
    from federated_data_distribution import load_and_prepare, global_holdout_split, get_all_distributions
    from models.cnn_gru import CNNGRU, get_initial_weights
    from fl_client import FLClient
    from fl_evaluator import GlobalEvaluator
    from aoa_selector import AOAClientSelector
    import fl_main_v2

    # Patch _aoa_select to log stats
    round_stats = []

    _orig_aoa_select = fl_main_v2._aoa_select

    def _logging_select(all_indices, prev_updates, global_weights, global_class_dist,
                        k_select, round_num, random_state, pop_size, max_iter,
                        lam=0.2, delta=0.0, alpha=0.3, beta=0.3, gamma=0.0, **kw):
        n_total = len(all_indices)
        aoa = AOAClientSelector(n_clients=n_total, k_select=k_select,
                                pop_size=pop_size, max_iter=max_iter,
                                alpha=alpha, beta=beta, gamma=gamma, delta=delta,
                                lam=lam, random_state=random_state + round_num)
        partial_scores = aoa.compute_scores(prev_updates, global_weights, global_class_dist)
        mean_score = float(partial_scores.mean()) if len(partial_scores) else 0.0
        n_classes = len(global_class_dist)
        full_scores = np.full(n_total, mean_score)
        full_class_dists = np.tile(global_class_dist, (n_total, 1))
        for j, u in enumerate(prev_updates):
            cid = u.get('client_id', -1) - 1
            if 0 <= cid < n_total:
                full_scores[cid] = partial_scores[j]
                cd = u.get('class_dist')
                if cd is not None:
                    full_class_dists[cid] = cd
        selected_local = aoa.select(full_scores, class_dists=full_class_dists)
        selected = [all_indices[i] for i in selected_local]

        ao_fitness = aoa._fitness(
            np.isin(np.arange(n_total), selected_local).astype(float),
            full_scores, full_class_dists)

        round_stats.append({
            'round':       round_num,
            'scores':      full_scores.copy(),
            'class_dists': full_class_dists.copy(),
            'selected':    selected_local[:],
            'ao_fitness':  ao_fitness,
            'lam':         lam,
            'k':           k_select,
        })
        print(f"   [AO] Selected: clients {sorted([s+1 for s in selected])}")
        return selected

    fl_main_v2._aoa_select = _logging_select

    from ga_pso_selector import GAPSOSelector
    from fl_client import FLClient

    def _no_gapso(self, **kwargs):
        self.feature_mask_ = np.ones(self.X_train.shape[1], dtype=bool)
        sel = GAPSOSelector(n_features=self.X_train.shape[1])
        sel.xgb_imp_   = np.ones(self.X_train.shape[1]) / self.X_train.shape[1]
        sel.best_mask_  = self.feature_mask_
        sel.best_fitness_ = float('nan')
        self.selector_ = sel
        return self.feature_mask_
    FLClient.compute_feature_mask = _no_gapso

    rdir = os.path.join(os.path.dirname(STATS_FILE), f'aoa8_seed{args.seed}')
    os.makedirs(rdir, exist_ok=True)

    fl_main_v2.main(config={
        'distribution': 'Dir_0.1',
        'n_clients':    20,
        'k_select':     8,
        'results_dir':  rdir,
        'experiment':   f'aoa8_seed{args.seed}',
        'device':       'cuda',
        'random_state': args.seed,
        'aoa_pop_size': 20,
        'aoa_max_iter': 30,
        'aoa_lam':      0.2,
        'aoa_delta':    0.0,
        'aoa_alpha':    0.3,
        'aoa_beta':     0.3,
        'aoa_gamma':    0.0,
    })

    with open(STATS_FILE, 'wb') as f:
        pickle.dump(round_stats, f)
    print(f"Saved {len(round_stats)} phase-2 round stats to {STATS_FILE}")
else:
    print(f"Loading cached stats from {STATS_FILE}")
    with open(STATS_FILE, 'rb') as f:
        round_stats = pickle.load(f)

# ── Phase 2: exhaustive fitness analysis ───────────────────────────────────────
print(f"\nRunning exhaustive search over C(20,8)=125,970 subsets for {len(round_stats)} rounds...")

def compute_fitness(subset_idx, scores, class_dists, lam, k):
    s = list(subset_idx)
    linear = scores[s].sum()
    if lam == 0.0 or k < 2:
        return linear
    n_pairs = k * (k - 1) / 2
    total_div = 0.0
    for a in range(k):
        for b in range(a + 1, k):
            diff = class_dists[s[a]] - class_dists[s[b]]
            total_div += float(np.dot(diff, diff) ** 0.5)
    return linear + lam * (total_div / n_pairs)


n, k = 20, 8
all_subsets = list(combinations(range(n), k))  # 125,970 subsets

ao_fits, exhaustive_fits, greedy_fits = [], [], []

for i, stat in enumerate(round_stats):
    scores      = stat['scores']
    class_dists = stat['class_dists']
    lam         = stat['lam']
    ao_sel      = stat['selected']
    ao_fit      = stat['ao_fitness']

    # Exhaustive
    best_fit, best_sub = -np.inf, None
    for subset in all_subsets:
        fit = compute_fitness(subset, scores, class_dists, lam, k)
        if fit > best_fit:
            best_fit = fit
            best_sub = subset

    # Greedy (top-k on linear scores, lam=0)
    greedy_sel = np.argsort(-scores)[:k].tolist()
    greedy_fit = compute_fitness(greedy_sel, scores, class_dists, lam=0.0, k=k)

    ao_fits.append(ao_fit)
    exhaustive_fits.append(best_fit)
    greedy_fits.append(greedy_fit)

    if (i + 1) % 5 == 0 or i == 0:
        gap = (best_fit - ao_fit) / (abs(best_fit) + 1e-10) * 100
        print(f"  Round {stat['round']:>3d}: exhaustive={best_fit:.4f}  AO={ao_fit:.4f}  "
              f"greedy(lam=0)={greedy_fit:.4f}  gap={gap:.2f}%")

ao_fits        = np.array(ao_fits)
exhaustive_fits = np.array(exhaustive_fits)
greedy_fits    = np.array(greedy_fits)

gaps = (exhaustive_fits - ao_fits) / (np.abs(exhaustive_fits) + 1e-10) * 100
greedy_gaps = (exhaustive_fits - greedy_fits) / (np.abs(exhaustive_fits) + 1e-10) * 100

print(f"\n{'='*60}")
print(f" EXHAUSTIVE FITNESS SANITY CHECK — seed={args.seed}")
print(f"{'='*60}")
print(f"  Rounds analysed       : {len(round_stats)}")
print(f"  AO vs exhaustive gap  : {gaps.mean():.3f}% ± {gaps.std():.3f}%  (max={gaps.max():.3f}%)")
print(f"  Greedy vs exhaustive  : {greedy_gaps.mean():.3f}% ± {greedy_gaps.std():.3f}%")
print(f"  AO achieves {100 - gaps.mean():.1f}% of optimal fitness on average")
print(f"  AO better than greedy (lam=0) in {(ao_fits > greedy_fits).sum()}/{len(ao_fits)} rounds")
print(f"{'='*60}")
