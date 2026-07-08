"""
Ablation: FedEL without AOA client selection.

Same as phase1 but uses random client selection in the ensemble phase
instead of AOA contribution-based selection.

Used to isolate the contribution of AOA.

Compare results/baselines/no_aoa/ against results/phase1/
"""

import numpy as np
import fl_main
from fl_server import FLServer

fl_main.CONFIG.update({
    'results_dir':   r'C:\Users\Dspike\Documents\FL-AdroidMaLD\results\baselines\no_aoa',
    'experiment':    'no_aoa',
    'device':        'cpu',
    'warmup_rounds': 35,
    'n_rounds':      70,
    'n_epochs':      7,
    'k_select':      8,
})

# Monkey-patch: replace AOA selection with random selection
_orig_select = FLServer.select_clients

def _random_select(self, all_indices, client_updates, global_class_dist, k_select, random_state=42):
    """Replace AOA with random client selection in ensemble phase."""
    if self.phase == 'warmup' or client_updates is None:
        return all_indices

    rng      = np.random.RandomState(random_state + self.round)
    selected = sorted(rng.choice(all_indices, size=k_select, replace=False).tolist())
    print(f"   [Random] Selected: clients {[s+1 for s in selected]}")
    return selected

FLServer.select_clients = _random_select


if __name__ == '__main__':
    fl_main.main()
