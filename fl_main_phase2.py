"""
FL-AdroidMaLD — Phase 2: Improved hyperparameters on AndMal2020.

Changes from phase1:
    - warmup_rounds: 15 → 20  (better G(·) convergence before freezing)
    - n_rounds:      30 → 50  (more ensemble rounds to stabilise meta_FC)
    - k_select:       5 → 8   (more clients per ensemble round → stable aggregation)
    - n_epochs:       5 → 7   (more local training per round)

Target: improve FileInfector and Trojan_Banker F1, stabilise ensemble phase.
"""

import fl_main

fl_main.CONFIG.update({
    'warmup_rounds': 20,
    'n_rounds':      50,
    'k_select':      8,
    'n_epochs':      7,
    'results_dir':   r'C:\Users\Dspike\Documents\FL-AdroidMaLD\results\phase2',
    'experiment':    'phase2_dir05',
    'device':        'cpu',
})

if __name__ == '__main__':
    fl_main.main()
