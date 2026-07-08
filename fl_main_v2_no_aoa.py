"""
Ablation: v2 without AOA client selection.

AOA is replaced with random client selection in phase 2.
GA+PSO feature masking is still active.

Compare results/v2_no_aoa/ against results/v2/ to isolate AOA contribution.
"""

import numpy as np
import fl_main_v2

# Monkey-patch: replace _aoa_select with random selection
_orig_aoa_select = fl_main_v2._aoa_select

def _random_select(all_indices, prev_updates, global_weights, global_class_dist,
                   k_select, round_num, random_state, pop_size, max_iter):
    """Replace AOA with random client selection."""
    rng      = np.random.RandomState(random_state + round_num)
    selected = sorted(rng.choice(all_indices, size=k_select, replace=False).tolist())
    print(f"   [Random] Selected: clients {[s+1 for s in selected]}")
    return selected

fl_main_v2._aoa_select = _random_select

if __name__ == '__main__':
    fl_main_v2.main(config={
        'results_dir': r'C:\Users\Dspike\Documents\FL-AdroidMaLD\results\v2_no_aoa',
        'experiment':  'v2_no_aoa',
        'device':      'cpu',
    })
