"""
Single Dir(0.1) experiment runner.
Usage: python run_experiment_dir01.py --method [fedavg_all|random8|aoa8] --seed N

Called by run_multiseed_dir01.py for each seed × method combination.
Each run is a fresh subprocess so monkey-patches don't bleed between runs.
"""

import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--method', choices=['fedavg_all', 'random8', 'aoa8'], required=True)
parser.add_argument('--seed',   type=int, default=42)
args = parser.parse_args()

from fl_client      import FLClient
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

# ── Client selection override ─────────────────────────────────────────────────
if args.method == 'random8':
    def _random_select(all_indices, prev_updates, global_weights, global_class_dist,
                       k_select, round_num, random_state, pop_size, max_iter):
        rng      = np.random.RandomState(random_state + round_num)
        selected = sorted(rng.choice(all_indices, size=k_select, replace=False).tolist())
        print(f"   [Random-8] Selected: clients {[s+1 for s in selected]}")
        return selected
    fl_main_v2._aoa_select = _random_select

elif args.method == 'fedavg_all':
    def _all_select(all_indices, prev_updates, global_weights, global_class_dist,
                    k_select, round_num, random_state, pop_size, max_iter):
        print(f"   [FedAvg-all] All clients selected")
        return all_indices
    fl_main_v2._aoa_select = _all_select

# aoa8: no override — uses fixed AOA with error-rate scoring from aoa_selector.py

# ── Run ───────────────────────────────────────────────────────────────────────
results_dir = (
    rf"C:\Users\Dspike\Documents\FL-AdroidMaLD\results\dir01_5seed"
    rf"\{args.method}\seed{args.seed}"
)

fl_main_v2.main(config={
    'distribution': 'Dir_0.1',
    'results_dir':  results_dir,
    'experiment':   f"{args.method}_seed{args.seed}",
    'device':       'cuda',
    'random_state': args.seed,
})
