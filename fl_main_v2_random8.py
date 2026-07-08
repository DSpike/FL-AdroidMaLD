"""
Baseline: Random-8 client selection (no GA+PSO, no AOA).

Phase 1 (rounds 1-20):  Standard FedAvg, all 10 clients, all 125 features.
Phase 2 (rounds 21-50): FedAvg continues with random selection of 8/10 clients.

Purpose: isolate whether AOA's contribution-aware scoring adds value over
         naive random reduction. Compare against:
             results/v2_no_gapso/  (AOA-8)
             results/baselines/fedavg_only/  (FedAvg-all-10)
"""

import numpy as np
from fl_client import FLClient
import fl_main_v2
from ga_pso_selector import GAPSOSelector

# ── 1. Skip GA+PSO: keep all 125 features ────────────────────────────────────
_orig_compute = FLClient.compute_feature_mask

def _no_gapso_mask(self, **kwargs):
    self.feature_mask_ = np.ones(self.X_train.shape[1], dtype=bool)
    self.selector_ = GAPSOSelector(n_features=self.X_train.shape[1])
    self.selector_.xgb_imp_    = np.ones(self.X_train.shape[1]) / self.X_train.shape[1]
    self.selector_.best_mask_  = self.feature_mask_
    self.selector_.best_fitness_ = float('nan')
    print(f"      [Random-8] Client {self.client_id}: all {self.X_train.shape[1]} features kept")
    return self.feature_mask_

FLClient.compute_feature_mask = _no_gapso_mask

# ── 2. Replace AOA with random selection ─────────────────────────────────────
def _random_select(all_indices, prev_updates, global_weights, global_class_dist,
                   k_select, round_num, random_state, pop_size, max_iter):
    rng      = np.random.RandomState(random_state + round_num)
    selected = sorted(rng.choice(all_indices, size=k_select, replace=False).tolist())
    print(f"   [Random-8] Selected: clients {[s+1 for s in selected]}")
    return selected

fl_main_v2._aoa_select = _random_select

# ── 3. Run ───────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    fl_main_v2.main(config={
        'results_dir': r'C:\Users\Dspike\Documents\FL-AdroidMaLD\results\baselines\random8',
        'experiment':  'random8',
        'device':      'cpu',
    })
