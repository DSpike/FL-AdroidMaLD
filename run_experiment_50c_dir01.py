"""
Single experiment runner — 50 clients, Dir(0.1), select 10.

C(50,10) = 10,272,278,170 — search space 80,000× larger than 20-client setup.
Participation rate: 10/50 = 20%.
Optimised AOA: λ=0.0, δ=0.3 (from ablation study).

Usage:
  python run_experiment_50c_dir01.py --method [fedavg_all|random10|aoa10] --seed N
"""

import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--method', choices=['fedavg_all', 'random10', 'aoa10'], required=True)
parser.add_argument('--seed',        type=int, default=42)
parser.add_argument('--results_dir', type=str, default=None)
args = parser.parse_args()

from fl_client       import FLClient
from ga_pso_selector import GAPSOSelector
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
if args.method == 'random10':
    def _random_select(all_indices, prev_updates, global_weights, global_class_dist,
                       k_select, round_num, random_state, pop_size, max_iter, **kw):
        rng      = np.random.RandomState(random_state + round_num)
        selected = sorted(rng.choice(all_indices, size=k_select, replace=False).tolist())
        print(f"   [Random-10] Selected: clients {[s+1 for s in selected]}")
        return selected
    fl_main_v2._aoa_select = _random_select

elif args.method == 'fedavg_all':
    def _all_select(all_indices, prev_updates, global_weights, global_class_dist,
                    k_select, round_num, random_state, pop_size, max_iter, **kw):
        print(f"   [FedAvg-all] All {len(all_indices)} clients selected")
        return all_indices
    fl_main_v2._aoa_select = _all_select

# aoa10: uses optimised λ=0.0, δ=0.3 from ablation

# ── Results directory ─────────────────────────────────────────────────────────
if args.results_dir:
    results_dir = args.results_dir
else:
    results_dir = (
        rf"C:\Users\Dspike\Documents\FL-AdroidMaLD\results\50c_dir01"
        rf"\{args.method}\seed{args.seed}"
    )

# ── Run ───────────────────────────────────────────────────────────────────────
fl_main_v2.main(config={
    'distribution':  'Dir_0.1',
    'n_clients':     50,
    'k_select':      10,
    'results_dir':   results_dir,
    'experiment':    f"{args.method}_seed{args.seed}",
    'device':        'cuda',
    'random_state':  args.seed,
    # Larger budget: C(50,10) = 10 billion combinations
    'aoa_pop_size':  30,
    'aoa_max_iter':  40,
    # Optimised hyperparameters from ablation
    'aoa_lam':       0.0,
    'aoa_delta':     0.3,
})
