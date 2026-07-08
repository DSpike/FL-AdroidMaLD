"""
AOA-8 client selection, Dir(0.1) non-IID distribution.
No GA+PSO (all 125 features kept).

Purpose: test whether AOA's contribution-aware selection outperforms
         random selection under MORE EXTREME non-IID (Dir(0.1) vs Dir(0.5)).
Compare: results/dir01/random8/  vs  results/dir01/aoa8/
"""

import numpy as np
from fl_client import FLClient
from ga_pso_selector import GAPSOSelector
import fl_main_v2

_orig_compute = FLClient.compute_feature_mask

def _no_gapso_mask(self, **kwargs):
    self.feature_mask_ = np.ones(self.X_train.shape[1], dtype=bool)
    self.selector_ = GAPSOSelector(n_features=self.X_train.shape[1])
    self.selector_.xgb_imp_    = np.ones(self.X_train.shape[1]) / self.X_train.shape[1]
    self.selector_.best_mask_  = self.feature_mask_
    self.selector_.best_fitness_ = float('nan')
    print(f"      [No-GA+PSO] Client {self.client_id}: all {self.X_train.shape[1]} features kept")
    return self.feature_mask_

FLClient.compute_feature_mask = _no_gapso_mask

if __name__ == '__main__':
    fl_main_v2.main(config={
        'distribution': 'Dir_0.1',
        'results_dir':  r'C:\Users\Dspike\Documents\FL-AdroidMaLD\results\dir01\aoa8',
        'experiment':   'aoa8_dir01',
        'device':       'cuda',
    })
