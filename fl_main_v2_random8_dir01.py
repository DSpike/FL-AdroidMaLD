"""
Random-8 client selection, Dir(0.1) non-IID distribution.
No GA+PSO (all 125 features kept).

Purpose: baseline — random 8/10 client selection under Dir(0.1).
Compare against results/dir01/aoa8/ to isolate AOA's contribution.
"""

import numpy as np
from fl_client import FLClient
from ga_pso_selector import GAPSOSelector
import fl_main_v2

def _no_gapso_mask(self, **kwargs):
    self.feature_mask_ = np.ones(self.X_train.shape[1], dtype=bool)
    self.selector_ = GAPSOSelector(n_features=self.X_train.shape[1])
    self.selector_.xgb_imp_    = np.ones(self.X_train.shape[1]) / self.X_train.shape[1]
    self.selector_.best_mask_  = self.feature_mask_
    self.selector_.best_fitness_ = float('nan')
    print(f"      [No-GA+PSO] Client {self.client_id}: all {self.X_train.shape[1]} features kept")
    return self.feature_mask_

FLClient.compute_feature_mask = _no_gapso_mask

def _random_select(all_indices, prev_updates, global_weights, global_class_dist,
                   k_select, round_num, random_state, pop_size, max_iter):
    rng      = np.random.RandomState(random_state + round_num)
    selected = sorted(rng.choice(all_indices, size=k_select, replace=False).tolist())
    print(f"   [Random-8] Selected: clients {[s+1 for s in selected]}")
    return selected

fl_main_v2._aoa_select = _random_select

if __name__ == '__main__':
    fl_main_v2.main(config={
        'distribution': 'Dir_0.1',
        'results_dir':  r'C:\Users\Dspike\Documents\FL-AdroidMaLD\results\dir01\random8',
        'experiment':   'random8_dir01',
        'device':       'cuda',
    })
