"""
FL-AdroidMaLD — Phase 3b: Deeper meta MLP (120->64->12).

Change from phase3:
    - meta_fc: Linear(120->12)  →  Linear(120->64)->ReLU->Dropout->Linear(64->12)
    - Parameters: 1,452  →  8,524  (~6x more capacity)
    - All other settings identical to phase3

Hypothesis: the shallow Linear meta-FC in phase3 (88.43%) hit a capacity
ceiling. A two-layer MLP should learn non-linear client combinations and
push ensemble performance above fedavg_only (90.31%).
"""

import fl_main

fl_main.CONFIG.update({
    'warmup_rounds': 35,
    'n_rounds':      70,
    'k_select':      8,
    'n_epochs':      7,
    'results_dir':   r'C:\Users\Dspike\Documents\FL-AdroidMaLD\results\phase3b',
    'experiment':    'phase3b_dir05',
    'device':        'cpu',
})

if __name__ == '__main__':
    fl_main.main()
