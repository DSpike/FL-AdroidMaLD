"""
Single experiment runner — 20 clients, Dir(0.1), select 8.

Methods:
  fedavg_all  — all 20 clients every round (no selection)
  random8     — uniform random selection of 8 clients
  aoa8        — AOA-based selection (our method)
  oort8       — Oort-style: sqrt(n_train) × error_rate utility
  poco8       — Power-of-Choice: candidate pool → highest error_rate
  gwo8        — Grey Wolf Optimizer selection (same fitness as AOA)
  pso8        — Binary PSO selection (same fitness as AOA)

Usage:
  python run_experiment_20c_dir01.py --method aoa8 --seed 42
  python run_experiment_20c_dir01.py --method aoa8 --seed 42 --lam 0.4 --delta 0.1
"""

import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--method', choices=[
    'fedavg_all', 'random8', 'aoa8', 'oort8', 'poco8', 'gwo8', 'pso8', 'divfl8', 'greedy8'
], required=True)
parser.add_argument('--seed',        type=int,   default=42)
parser.add_argument('--lam',         type=float, default=0.2,
                    help='AOA diversity weight λ (ablation)')
parser.add_argument('--delta',       type=float, default=0.3,
                    help='AOA error-rate weight δ (ablation)')
parser.add_argument('--alpha',       type=float, default=0.3,
                    help='AOA divergence weight α (ablation)')
parser.add_argument('--beta',        type=float, default=0.3,
                    help='AOA coverage weight β (ablation)')
parser.add_argument('--gamma',       type=float, default=0.1,
                    help='AOA volume weight γ (ablation)')
parser.add_argument('--results_dir',  type=str,   default=None,
                    help='Override results directory')
parser.add_argument('--distribution', type=str,   default='Dir_0.1',
                    choices=['IID', 'Dir_1.0', 'Dir_0.5', 'Dir_0.1', 'Dir_0.05', 'Pathological'],
                    help='Data distribution (default: Dir_0.1)')
parser.add_argument('--k_select',    type=int,   default=8,
                    help='Number of clients to select per round (default: 8)')
args = parser.parse_args()

import torch
from fl_client                import FLClient
from ga_pso_selector          import GAPSOSelector
from metaheuristic_selectors  import GWOClientSelector, PSOClientSelector
from aoa_selector             import AOAClientSelector
from divfl_selector           import DivFLClientSelector
import fl_main_v2

# ── Always skip GA+PSO ────────────────────────────────────────────────────────
def _no_gapso_mask(self, **kwargs):
    self.feature_mask_ = np.ones(self.X_train.shape[1], dtype=bool)
    sel = GAPSOSelector(n_features=self.X_train.shape[1])
    sel.xgb_imp_      = np.ones(self.X_train.shape[1]) / self.X_train.shape[1]
    sel.best_mask_    = self.feature_mask_
    sel.best_fitness_ = float('nan')
    self.selector_    = sel
    return self.feature_mask_

FLClient.compute_feature_mask = _no_gapso_mask

# ── Client selection overrides ────────────────────────────────────────────────
if args.method == 'random8':
    def _random_select(all_indices, prev_updates, global_weights, global_class_dist,
                       k_select, round_num, random_state, pop_size, max_iter, **kw):
        rng      = np.random.RandomState(random_state + round_num)
        selected = sorted(rng.choice(all_indices, size=k_select, replace=False).tolist())
        print(f"   [Random-8] Selected: clients {[s+1 for s in selected]}")
        return selected
    fl_main_v2._aoa_select = _random_select

elif args.method == 'fedavg_all':
    def _all_select(all_indices, prev_updates, global_weights, global_class_dist,
                    k_select, round_num, random_state, pop_size, max_iter, **kw):
        print(f"   [FedAvg-all] All {len(all_indices)} clients selected")
        return all_indices
    fl_main_v2._aoa_select = _all_select

elif args.method == 'oort8':
    # Oort: utility_k = sqrt(n_train_k) × train_loss_k  (Lai et al. 2021, statistical utility)
    def _oort_select(all_indices, prev_updates, global_weights, global_class_dist,
                     k_select, round_num, random_state, pop_size, max_iter, **kw):
        n_total = len(all_indices)
        utility = np.zeros(n_total)
        if prev_updates:
            for u in prev_updates:
                cid = u.get('client_id', -1) - 1
                if 0 <= cid < n_total:
                    utility[cid] = np.sqrt(u['n_train']) * float(u.get('train_loss', 0.0))
            observed  = utility[utility > 0]
            mean_util = float(observed.mean()) if len(observed) else 0.0
            utility[utility == 0] = mean_util   # impute unobserved clients

        # Deterministic top-k + small noise for tie-breaking
        rng   = np.random.RandomState(random_state + round_num)
        noise = rng.rand(n_total) * 1e-6
        order = np.argsort(-(utility + noise))
        selected = sorted([all_indices[i] for i in order[:k_select]])
        print(f"   [Oort-8] Selected: clients {[s+1 for s in selected]}")
        return selected
    fl_main_v2._aoa_select = _oort_select

elif args.method == 'poco8':
    # Power-of-Choice: sample 2×k candidates, pick k with highest error rate
    def _poco_select(all_indices, prev_updates, global_weights, global_class_dist,
                     k_select, round_num, random_state, pop_size, max_iter, **kw):
        n_total = len(all_indices)
        rng     = np.random.RandomState(random_state + round_num)
        d       = min(2 * k_select, n_total)
        candidate_local = rng.choice(n_total, size=d, replace=False)

        # Local loss per client (Cho et al. 2022: select high-loss clients)
        err = {}
        if prev_updates:
            for u in prev_updates:
                cid = u.get('client_id', -1) - 1
                if 0 <= cid < n_total:
                    err[cid] = float(u.get('train_loss', 0.0))
        mean_err = float(np.mean(list(err.values()))) if err else 0.5

        ranked = sorted(candidate_local,
                        key=lambda i: -err.get(i, mean_err))
        selected = sorted([all_indices[i] for i in ranked[:k_select]])
        print(f"   [PoCo-8] Candidates={sorted([c+1 for c in candidate_local])} "
              f"→ Selected: clients {[s+1 for s in selected]}")
        return selected
    fl_main_v2._aoa_select = _poco_select

elif args.method == 'gwo8':
    def _gwo_select(all_indices, prev_updates, global_weights, global_class_dist,
                    k_select, round_num, random_state, pop_size, max_iter,
                    lam=0.2, delta=0.3, **kw):
        import fl_main_v2 as _fm
        n_total = len(all_indices)
        _dummy  = AOAClientSelector(n_clients=n_total, k_select=k_select,
                                    pop_size=pop_size, max_iter=max_iter,
                                    delta=delta, lam=lam,
                                    random_state=random_state + round_num)
        partial_scores = _dummy.compute_scores(prev_updates, global_weights, global_class_dist)
        mean_score     = float(partial_scores.mean()) if len(partial_scores) else 0.0
        n_classes      = len(global_class_dist)
        full_scores    = np.full(n_total, mean_score)
        full_class_dists = np.tile(global_class_dist, (n_total, 1))
        for j, u in enumerate(prev_updates):
            cid = u.get('client_id', -1) - 1
            if 0 <= cid < n_total:
                full_scores[cid] = partial_scores[j]
                cd = u.get('class_dist')
                if cd is not None:
                    full_class_dists[cid] = cd
        gwo = GWOClientSelector(n_clients=n_total, k_select=k_select,
                                pop_size=pop_size, max_iter=max_iter,
                                delta=delta, lam=lam,
                                random_state=random_state + round_num)
        gwo._last_class_dists = full_class_dists
        selected_local = gwo.select(full_scores, class_dists=full_class_dists)
        selected = [all_indices[i] for i in selected_local]
        print(f"   [GWO-8] Selected: clients {sorted([s+1 for s in selected])}")
        return selected
    fl_main_v2._aoa_select = _gwo_select

elif args.method == 'pso8':
    def _pso_select(all_indices, prev_updates, global_weights, global_class_dist,
                    k_select, round_num, random_state, pop_size, max_iter,
                    lam=0.2, delta=0.3, **kw):
        n_total = len(all_indices)
        _dummy  = AOAClientSelector(n_clients=n_total, k_select=k_select,
                                    pop_size=pop_size, max_iter=max_iter,
                                    delta=delta, lam=lam,
                                    random_state=random_state + round_num)
        partial_scores = _dummy.compute_scores(prev_updates, global_weights, global_class_dist)
        mean_score     = float(partial_scores.mean()) if len(partial_scores) else 0.0
        n_classes      = len(global_class_dist)
        full_scores    = np.full(n_total, mean_score)
        full_class_dists = np.tile(global_class_dist, (n_total, 1))
        for j, u in enumerate(prev_updates):
            cid = u.get('client_id', -1) - 1
            if 0 <= cid < n_total:
                full_scores[cid] = partial_scores[j]
                cd = u.get('class_dist')
                if cd is not None:
                    full_class_dists[cid] = cd
        pso = PSOClientSelector(n_clients=n_total, k_select=k_select,
                                pop_size=pop_size, max_iter=max_iter,
                                delta=delta, lam=lam,
                                random_state=random_state + round_num)
        pso._last_class_dists = full_class_dists
        selected_local = pso.select(full_scores, class_dists=full_class_dists)
        selected = [all_indices[i] for i in selected_local]
        print(f"   [PSO-8] Selected: clients {sorted([s+1 for s in selected])}")
        return selected
    fl_main_v2._aoa_select = _pso_select

elif args.method == 'greedy8':
    def _greedy_select(all_indices, prev_updates, global_weights, global_class_dist,
                       k_select, round_num, random_state, pop_size, max_iter,
                       alpha=0.3, beta=0.3, gamma=0.0, delta=0.0, **kw):
        n_total = len(all_indices)
        _dummy = AOAClientSelector(n_clients=n_total, k_select=k_select,
                                   pop_size=1, max_iter=1,
                                   alpha=alpha, beta=beta, gamma=gamma, delta=delta,
                                   lam=0.0,
                                   random_state=random_state + round_num)
        if prev_updates:
            partial_scores = _dummy.compute_scores(prev_updates, global_weights, global_class_dist)
            mean_score = float(partial_scores.mean())
        else:
            partial_scores = np.array([])
            mean_score = 0.0
        full_scores = np.full(n_total, mean_score)
        for j, u in enumerate(prev_updates):
            cid = u.get('client_id', -1) - 1
            if 0 <= cid < n_total:
                full_scores[cid] = partial_scores[j]
        order = np.argsort(-full_scores)
        selected_local = sorted(order[:k_select].tolist())
        selected = [all_indices[i] for i in selected_local]
        print(f"   [Greedy-8] Selected: clients {sorted([s+1 for s in selected])}")
        return selected
    fl_main_v2._aoa_select = _greedy_select

elif args.method == 'divfl8':
    def _divfl_select(all_indices, prev_updates, global_weights, global_class_dist,
                      k_select, round_num, random_state, pop_size, max_iter, **kw):
        n_total = len(all_indices)
        sel = DivFLClientSelector(n_clients=n_total, k_select=k_select)
        selected_local = sel.select(prev_updates, global_weights)
        selected = [all_indices[i] for i in selected_local]
        print(f"   [DivFL-8] Selected: clients {sorted([s+1 for s in selected])}")
        return selected
    fl_main_v2._aoa_select = _divfl_select

# aoa8: no override — uses AOA with lam/delta from config

# ── Results directory ─────────────────────────────────────────────────────────
if args.results_dir:
    results_dir = args.results_dir
else:
    results_dir = (
        rf"C:\Users\Dspike\Documents\FL-AdroidMaLD\results\20c_dir01_5seed"
        rf"\{args.method}\seed{args.seed}"
    )

# ── Run ───────────────────────────────────────────────────────────────────────
fl_main_v2.main(config={
    'distribution':  args.distribution,
    'n_clients':     20,
    'k_select':      args.k_select,
    'results_dir':   results_dir,
    'experiment':    f"{args.method}_seed{args.seed}",
    'device':        'cuda',
    'random_state':  args.seed,
    'aoa_pop_size':  20,
    'aoa_max_iter':  30,
    'aoa_lam':       args.lam,
    'aoa_delta':     args.delta,
    'aoa_alpha':     args.alpha,
    'aoa_beta':      args.beta,
    'aoa_gamma':     args.gamma,
})
