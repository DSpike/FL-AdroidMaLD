"""
FL-AdroidMaLD — CIC-AndMal2017 experiment.

Run AFTER preprocessing:
    python preprocessing/reorganize_cicandmal2017.py
    python preprocessing/cicandmal2017_preprocessor.py
Then:
    python fl_main_cicandmal2017.py
"""

import fl_main

fl_main.CONFIG.update({
    # Data
    'data_path':    r'C:\Users\Dspike\Documents\FL-AdroidMaLD\normalized_dataset_cicandmal2017\combined_dataset.csv',
    'results_dir':  r'C:\Users\Dspike\Documents\FL-AdroidMaLD\results\cicandmal2017',
    'experiment':   'cicandmal2017_dir05',
    'distribution': 'Dir_0.5',

    # Larger batch + fewer epochs to handle 2.6M rows efficiently
    'batch_size':   256,
    'n_epochs':     3,

    'device':       'cpu',   # switch to 'cuda' when GPU is free
})

if __name__ == '__main__':
    fl_main.main()
