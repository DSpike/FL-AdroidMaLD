"""
FL-AdroidMaLD — Phase 3: Extended warmup to fix ensemble underperformance.

Problem found in Phase 2:
    - warmup_rounds=20 freezes G(·) at ~83% F1-macro
    - fedavg_only shows G(·) is still improving until round ~45
    - Result: ensemble phase (84.2%) loses to FedAvg-only (90.31%)

Fix:
    - warmup_rounds=35  (G(·) reaches ~88% before freezing)
    - n_rounds=70       (35 warmup + 35 ensemble rounds)
    - AOA also updated: val_f1 added as 4th scoring component (delta=0.3)
    - class_dist now correctly passed in ensemble phase

Target: ensemble phase should surpass fedavg_only (90.31% F1-macro).
"""

import fl_main

fl_main.CONFIG.update({
    'warmup_rounds': 35,
    'n_rounds':      70,
    'k_select':      8,
    'n_epochs':      7,
    'results_dir':   r'C:\Users\Dspike\Documents\FL-AdroidMaLD\results\phase3',
    'experiment':    'phase3_dir05',
    'device':        'cpu',
})

if __name__ == '__main__':
    fl_main.main()
