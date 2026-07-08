"""
Ablation: FedEL without GA+PSO feature selection.

Same as phase1 but skips compute_feature_mask() at the transition.
The ensemble is built and meta_FC is trained normally — only GA+PSO is absent.
AOA contribution scoring falls back to divergence + class_coverage only
(no feature-mask-based score component).

Used to isolate the contribution of GA+PSO.

Compare results/baselines/no_gapso/ against results/phase1/
"""

import fl_main

fl_main.CONFIG.update({
    'results_dir':   r'C:\Users\Dspike\Documents\FL-AdroidMaLD\results\baselines\no_gapso',
    'experiment':    'no_gapso',
    'device':        'cpu',
    'warmup_rounds': 35,
    'n_rounds':      70,
    'n_epochs':      7,
    'k_select':      8,
})

# Monkey-patch: skip GA+PSO at transition by replacing compute_feature_mask
# with a no-op that returns a full mask (all features selected).
import numpy as np
from fl_client import FLClient

_orig_compute = FLClient.compute_feature_mask

def _no_gapso_mask(self, **kwargs):
    """Skip GA+PSO — select all features (full mask)."""
    import numpy as np
    self.feature_mask_ = np.ones(self.X_train.shape[1], dtype=bool)
    # Still create a dummy selector so AOA doesn't crash on selector_ access
    from ga_pso_selector import GAPSOSelector
    self.selector_ = GAPSOSelector(n_features=self.X_train.shape[1])
    self.selector_.xgb_imp_      = np.ones(self.X_train.shape[1]) / self.X_train.shape[1]
    self.selector_.best_mask_    = self.feature_mask_
    self.selector_.best_fitness_ = float('nan')
    n_sel = int(self.feature_mask_.sum())
    print(f"      [No-GA+PSO] Client {self.client_id}: all {n_sel} features kept")
    return self.feature_mask_

FLClient.compute_feature_mask = _no_gapso_mask


if __name__ == '__main__':
    fl_main.main()
