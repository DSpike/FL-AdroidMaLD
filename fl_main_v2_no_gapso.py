"""
Ablation: v2 without GA+PSO feature selection.

GA+PSO is skipped — all 125 features used throughout.
AOA client selection is still active in phase 2.

Compare results/v2_no_gapso/ against results/v2/ to isolate GA+PSO contribution.
"""

from fl_client import FLClient
import fl_main_v2

# Monkey-patch: skip GA+PSO by replacing compute_feature_mask with no-op
import numpy as np
from ga_pso_selector import GAPSOSelector

_orig_compute = FLClient.compute_feature_mask

def _no_gapso_mask(self, **kwargs):
    """Skip GA+PSO — keep all features (full mask)."""
    self.feature_mask_ = np.ones(self.X_train.shape[1], dtype=bool)
    self.selector_ = GAPSOSelector(n_features=self.X_train.shape[1])
    self.selector_.xgb_imp_   = np.ones(self.X_train.shape[1]) / self.X_train.shape[1]
    self.selector_.best_mask_  = self.feature_mask_
    self.selector_.best_fitness_ = float('nan')
    print(f"      [No-GA+PSO] Client {self.client_id}: all {self.X_train.shape[1]} features kept")
    return self.feature_mask_

FLClient.compute_feature_mask = _no_gapso_mask

if __name__ == '__main__':
    fl_main_v2.main(config={
        'results_dir': r'C:\Users\Dspike\Documents\FL-AdroidMaLD\results\v2_no_gapso',
        'experiment':  'v2_no_gapso',
        'device':      'cpu',
    })
