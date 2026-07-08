"""
Baseline: FedAvg-only (no ensemble, no GA+PSO, no AOA).

Runs full 30 rounds of standard FedAvg on the CNN+GRU.
Used to isolate the contribution of the FedEL ensemble phase.

Compare results/baselines/fedavg_only/ against results/phase1/
"""

import os
import time
import numpy as np
import torch

from federated_data_distribution import (
    load_and_prepare,
    global_holdout_split,
    get_all_distributions,
)
from models.cnn_gru import CNNGRU, get_initial_weights
from fl_client    import FLClient
from fl_evaluator import GlobalEvaluator
import fl_main

CONFIG = dict(fl_main.CONFIG)
CONFIG.update({
    'n_rounds':      50,
    'warmup_rounds': 50,   # all rounds are warmup → never enters ensemble
    'n_epochs':      7,
    'k_select':      10,   # all clients each round
    'results_dir':   r'C:\Users\Dspike\Documents\FL-AdroidMaLD\results\baselines\fedavg_only',
    'experiment':    'fedavg_only',
    'device':        'cpu',
})


def main():
    start = time.time()
    torch.manual_seed(CONFIG['random_state'])
    np.random.seed(CONFIG['random_state'])

    device = torch.device(CONFIG['device'])
    print(f"\n{'='*60}")
    print(f" Baseline: FedAvg-only (no ensemble)")
    print(f" Device  : {device}  Rounds: {CONFIG['n_rounds']}")
    print(f"{'='*60}\n")

    os.makedirs(CONFIG['results_dir'], exist_ok=True)

    X, y, feature_cols, class_names, le = load_and_prepare(CONFIG['data_path'])
    X_train, X_test, y_train, y_test   = global_holdout_split(X, y, random_state=CONFIG['random_state'])
    print(f"Train={len(y_train)}  Test={len(y_test)}  Features={X_train.shape[1]}")

    distributions = get_all_distributions(X_train, y_train, n_clients=CONFIG['n_clients'],
                                          random_state=CONFIG['random_state'])
    client_data   = distributions[CONFIG['distribution']]

    input_dim   = X_train.shape[1]
    num_classes = len(class_names)

    # Server-side global weights (pure FedAvg, no FLServer needed)
    global_weights = get_initial_weights(input_dim, num_classes=num_classes)

    clients = [
        FLClient(
            client_id    = i + 1,
            X            = client_data[i]['X'],
            y            = client_data[i]['y'],
            num_classes  = num_classes,
            device       = device,
            random_state = CONFIG['random_state'],
        )
        for i in range(CONFIG['n_clients'])
    ]

    evaluator = GlobalEvaluator(
        X_test      = X_test,
        y_test      = y_test,
        class_names = list(class_names),
        results_dir = CONFIG['results_dir'],
        experiment  = CONFIG['experiment'],
    )

    for rnd in range(1, CONFIG['n_rounds'] + 1):
        t_round = time.time()
        print(f"{'─'*60}")
        print(f" Round {rnd:>3d}/{CONFIG['n_rounds']}  [FEDAVG]")

        client_updates = []
        for idx in range(CONFIG['n_clients']):
            upd = clients[idx].train(
                global_weights = global_weights,
                n_epochs       = CONFIG['n_epochs'],
                lr             = CONFIG['lr'],
                batch_size     = CONFIG['batch_size'],
                phase          = 'warmup',
            )
            client_updates.append(upd)
            print(f"   Client {idx+1:>2d}: val_f1={upd['val_f1']*100:.1f}%")

        # FedAvg aggregation
        total = sum(u['n_train'] for u in client_updates)
        aggregated = {}
        for key in global_weights:
            aggregated[key] = torch.zeros_like(global_weights[key], dtype=torch.float32)
            for u in client_updates:
                aggregated[key] += u['weights'][key].float() * (u['n_train'] / total)
        global_weights = aggregated

        # Evaluate
        model = CNNGRU(input_dim, num_classes=num_classes)
        model.load_state_dict(global_weights)
        model = model.to(device)
        evaluator.evaluate(model, device, round_num=rnd, phase='warmup')
        print(f" Round time: {time.time()-t_round:.1f}s")

    evaluator.plot_convergence()
    evaluator.plot_confusion_matrix(round_num=-1)
    evaluator.plot_per_class_metrics(round_num=-1)

    total = time.time() - start
    print(f"\n{'='*60}")
    print(f" FedAvg-only complete in {total/60:.1f} minutes")
    print(f" Best F1-macro: {evaluator.best_f1*100:.2f}%")
    print(f" Results: {CONFIG['results_dir']}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
