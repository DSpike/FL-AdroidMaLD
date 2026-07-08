"""
FL-AdroidMaLD v2: FedAvg + GA+PSO feature regularization + AOA client selection.

No ensemble phase. Pure FedAvg-based system with two enhancements:

    Phase 1 (rounds 1-20):  Standard FedAvg, all 10 clients, all 125 features
    Transition (round 20):  GA+PSO computes per-client feature masks (runs once)
    Phase 2 (rounds 21-50): FedAvg continues with:
                              - Zero-masking applied in local training (GA+PSO masks)
                              - AOA selects top-8 clients each round

Ablation variants:
    fl_main_v2_no_gapso.py  — skip GA+PSO (full features in phase 2, AOA only)
    fl_main_v2_no_aoa.py    — random client selection in phase 2 (GA+PSO only)
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
from aoa_selector import AOAClientSelector

# -----------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------

CONFIG = {
    'data_path':      r'C:\Users\Dspike\Documents\FL-AdroidMaLD\normalized_dataset\combined_dataset.csv',
    'distribution':   'Dir_0.5',

    'n_clients':      10,
    'k_select':       8,       # AOA selects 8 of 10 in phase 2
    'n_rounds':       50,
    'gapso_round':    20,      # GA+PSO runs after this round; masking + AOA start from round 21

    'n_epochs':       7,
    'batch_size':     32,
    'lr':             1e-3,
    'hidden_dim':     64,
    'device':         'cpu',

    'results_dir':    r'C:\Users\Dspike\Documents\FL-AdroidMaLD\results\v2',
    'experiment':     'v2_dir05',
    'random_state':   42,

    # GA+PSO settings
    'n_ga_gen':       10,
    'pop_size':       8,
    'n_pso_iter':     10,

    # AOA settings
    'aoa_pop_size':   10,
    'aoa_max_iter':   15,
    'aoa_lam':        0.2,   # diversity weight (λ)
    'aoa_delta':      0.3,   # error-rate weight (δ)
    'aoa_alpha':      0.3,   # divergence weight (α)
    'aoa_beta':       0.3,   # coverage weight (β)
    'aoa_gamma':      0.1,   # volume weight (γ)
}


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def _fedavg(updates: list, global_weights: dict) -> dict:
    """Weighted FedAvg aggregation."""
    total = sum(u['n_train'] for u in updates)
    aggregated = {}
    for key in global_weights:
        aggregated[key] = torch.zeros_like(global_weights[key], dtype=torch.float32)
        for u in updates:
            aggregated[key] += u['weights'][key].float() * (u['n_train'] / total)
    return aggregated


def _aoa_select(
    all_indices:       list,
    prev_updates:      list,
    global_weights:    dict,
    global_class_dist: np.ndarray,
    k_select:          int,
    round_num:         int,
    random_state:      int,
    pop_size:          int,
    max_iter:          int,
    lam:               float = 0.2,
    delta:             float = 0.3,
    alpha:             float = 0.3,
    beta:              float = 0.3,
    gamma:             float = 0.1,
    **kwargs,
) -> list:
    """AOA client selection — returns selected indices."""
    n_total = len(all_indices)

    aoa = AOAClientSelector(
        n_clients    = n_total,
        k_select     = k_select,
        pop_size     = pop_size,
        max_iter     = max_iter,
        lam          = lam,
        delta        = delta,
        alpha        = alpha,
        beta         = beta,
        gamma        = gamma,
        random_state = random_state + round_num,
    )

    partial_scores = aoa.compute_scores(prev_updates, global_weights, global_class_dist)
    mean_score     = float(partial_scores.mean()) if len(partial_scores) else 0.0

    full_scores = np.full(n_total, mean_score)

    # Build full class_dists array (n_total × n_classes) for diversity term.
    # Clients not in prev_updates fall back to global_class_dist.
    n_classes         = len(global_class_dist)
    full_class_dists  = np.tile(global_class_dist, (n_total, 1))

    for j, u in enumerate(prev_updates):
        cid = u.get('client_id', -1) - 1
        if 0 <= cid < n_total:
            full_scores[cid] = partial_scores[j]
            cd = u.get('class_dist', None)
            if cd is not None:
                full_class_dists[cid] = cd

    selected_local = aoa.select(full_scores, class_dists=full_class_dists)
    selected       = [all_indices[i] for i in selected_local]

    print(f"   [AOA] Scores  : {np.round(full_scores, 3)}")
    print(f"   [AOA] Selected: clients {sorted([s+1 for s in selected])}")
    return selected


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main(config=None):
    cfg = dict(CONFIG)
    if config is not None:
        cfg.update(config)

    start = time.time()
    torch.manual_seed(cfg['random_state'])
    np.random.seed(cfg['random_state'])

    device = torch.device(cfg['device'])
    print(f"\n{'='*60}")
    print(f" FL-AdroidMaLD v2: FedAvg + GA+PSO + AOA (no ensemble)")
    print(f" Device : {device}   Rounds: {cfg['n_rounds']}")
    print(f" GA+PSO transition at round {cfg['gapso_round']}")
    print(f" AOA k_select={cfg['k_select']} (phase 2 only)")
    print(f"{'='*60}\n")

    os.makedirs(cfg['results_dir'], exist_ok=True)

    # Overhead log: records per-round selection time for systems analysis
    _overhead_path = os.path.join(
        cfg['results_dir'], f"{cfg['experiment']}_overhead.csv")
    _overhead_file = open(_overhead_path, 'w', newline='')
    import csv as _csv
    _overhead_writer = _csv.writer(_overhead_file)
    _overhead_writer.writerow(['round', 'phase', 'sel_overhead_s'])

    # --- Data ---
    X, y, feature_cols, class_names, le = load_and_prepare(cfg['data_path'])
    X_train, X_test, y_train, y_test    = global_holdout_split(X, y, random_state=cfg['random_state'])
    print(f"Train={len(y_train)}  Test={len(y_test)}  Features={X_train.shape[1]}")

    distributions = get_all_distributions(
        X_train, y_train,
        n_clients    = cfg['n_clients'],
        random_state = cfg['random_state'],
    )
    client_data = distributions[cfg['distribution']]

    input_dim   = X_train.shape[1]
    num_classes = len(class_names)

    # Global class distribution for AOA coverage scoring
    counts = np.bincount(y_train, minlength=num_classes).astype(float)
    global_class_dist = counts / counts.sum()

    # --- Clients ---
    clients = [
        FLClient(
            client_id    = i + 1,
            X            = client_data[i]['X'],
            y            = client_data[i]['y'],
            num_classes  = num_classes,
            device       = device,
            random_state = cfg['random_state'],
        )
        for i in range(cfg['n_clients'])
    ]

    # --- Server state ---
    global_weights   = get_initial_weights(input_dim, num_classes=num_classes)
    all_indices      = list(range(cfg['n_clients']))
    selected_indices = all_indices.copy()
    prev_updates     = None
    gapso_done       = False
    phase            = 'warmup'

    # --- Evaluator ---
    evaluator = GlobalEvaluator(
        X_test      = X_test,
        y_test      = y_test,
        class_names = list(class_names),
        results_dir = cfg['results_dir'],
        experiment  = cfg['experiment'],
    )

    # -----------------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------------

    for rnd in range(1, cfg['n_rounds'] + 1):
        t_round = time.time()

        # === GA+PSO Transition ===
        if rnd == cfg['gapso_round'] + 1 and not gapso_done:
            print(f"\n{'─'*60}")
            print(f"   [Transition] Running GA+PSO feature selection per client ...")
            for client in clients:
                client.compute_feature_mask(
                    n_ga_gen     = cfg['n_ga_gen'],
                    pop_size     = cfg['pop_size'],
                    n_pso_iter   = cfg['n_pso_iter'],
                    random_state = cfg['random_state'],
                )
            n_sel_avg = np.mean([c.feature_mask_.sum() for c in clients])
            print(f"   Consensus: {n_sel_avg/input_dim*100:.1f}% of features selected on average")
            gapso_done = True
            phase      = 'phase2'

        # === Phase indicator ===
        phase_label = 'warmup' if rnd <= cfg['gapso_round'] else 'phase2'

        print(f"{'─'*60}")
        print(f" Round {rnd:>3d}/{cfg['n_rounds']}  [{phase_label.upper()}]")

        # === AOA Client Selection (phase 2 only) ===
        sel_overhead_s = 0.0
        if phase_label == 'phase2' and prev_updates is not None:
            _t_sel = time.time()
            selected_indices = _aoa_select(
                all_indices       = all_indices,
                prev_updates      = prev_updates,
                global_weights    = global_weights,
                global_class_dist = global_class_dist,
                k_select          = cfg['k_select'],
                round_num         = rnd,
                random_state      = cfg['random_state'],
                pop_size          = cfg['aoa_pop_size'],
                max_iter          = cfg['aoa_max_iter'],
                lam               = cfg.get('aoa_lam',   0.2),
                delta             = cfg.get('aoa_delta', 0.3),
                alpha             = cfg.get('aoa_alpha', 0.3),
                beta              = cfg.get('aoa_beta',  0.3),
                gamma             = cfg.get('aoa_gamma', 0.1),
            )
            sel_overhead_s = time.time() - _t_sel
        elif phase_label == 'warmup':
            selected_indices = all_indices.copy()

        # === Local Training ===
        client_updates = []
        for idx in selected_indices:
            upd = clients[idx].train(
                global_weights = global_weights,
                n_epochs       = cfg['n_epochs'],
                lr             = cfg['lr'],
                batch_size     = cfg['batch_size'],
                phase          = 'warmup',   # always warmup-style (full CNN+GRU updated)
            )
            client_updates.append(upd)
            print(f"   Client {idx+1:>2d}: val_f1={upd['val_f1']*100:.1f}%")

        # === FedAvg Aggregation ===
        global_weights = _fedavg(client_updates, global_weights)
        prev_updates   = client_updates

        # === Evaluate ===
        model = CNNGRU(input_dim, num_classes=num_classes)
        model.load_state_dict(global_weights)
        model = model.to(device)
        evaluator.evaluate(model, device, round_num=rnd, phase=phase_label)
        _overhead_writer.writerow([rnd, phase_label, f'{sel_overhead_s:.4f}'])
        _overhead_file.flush()
        print(f" Round time: {time.time()-t_round:.1f}s  "
              f"[sel_overhead={sel_overhead_s:.2f}s]")

    _overhead_file.close()

    # --- Final plots ---
    evaluator.plot_convergence()
    evaluator.plot_confusion_matrix(round_num=-1)
    evaluator.plot_per_class_metrics(round_num=-1)

    elapsed = time.time() - start
    print(f"\n{'='*60}")
    print(f" v2 complete in {elapsed/60:.1f} minutes")
    print(f" Best F1-macro: {evaluator.best_f1*100:.2f}%")
    print(f" Results: {cfg['results_dir']}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
