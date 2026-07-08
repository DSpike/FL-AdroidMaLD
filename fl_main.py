"""
FL-AdroidMaLD — Phase 1 main training loop.

What this runs:
    - 10 clients, Dirichlet α=0.5 (moderate non-IID)
    - 30 rounds: 15 warmup (FedAvg) + 15 ensemble (FedEL)
    - All 10 clients participate each round (AOA added in Phase 2)
    - Global holdout evaluation after every round
    - Results saved to results/phase1/
"""

import os
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from federated_data_distribution import (
    load_and_prepare,
    global_holdout_split,
    get_all_distributions,
)
from models.cnn_gru import CNNGRU
from fl_client    import FLClient
from fl_server    import FLServer
from fl_evaluator import GlobalEvaluator

# -----------------------------------------------------------------------
# Configuration — single place to change experiment settings
# -----------------------------------------------------------------------

CONFIG = {
    # Data
    'data_path':      r'C:\Users\Dspike\Documents\FL-AdroidMaLD\normalized_dataset\combined_dataset.csv',
    'distribution':   'Dir_0.5',   # IID | Dir_1.0 | Dir_0.5 | Dir_0.1 | Pathological

    # Federation
    'n_clients':      10,
    'k_select':       5,    # clients selected per ensemble round (AOA)
    'n_rounds':       30,
    'warmup_rounds':  15,

    # Local training
    'n_epochs':       5,
    'batch_size':     32,
    'lr':             1e-3,
    'hidden_dim':     64,
    'device':         'cpu',   # 'cuda' | 'cpu'
    'mask_warmup_rounds': 3,    # apply zero-masking in last N warmup rounds so G(·) adapts

    # Output
    'results_dir':    r'C:\Users\Dspike\Documents\FL-AdroidMaLD\results\phase1',
    'experiment':     'phase1_dir05',
    'random_state':   42,
}


# -----------------------------------------------------------------------
# Feature analysis plots (called once at warmup→ensemble transition)
# -----------------------------------------------------------------------

def plot_feature_analysis(clients, feature_cols, results_dir, experiment):
    """
    Generate two plots after GA+PSO runs on all clients:

    1. XGBoost feature importance — mean ± std across clients (top-30 bar chart)
    2. Feature selection heatmap — clients × features binary grid
    """
    n_clients  = len(clients)
    n_features = len(feature_cols)

    # Collect per-client data
    imp_matrix  = np.zeros((n_clients, n_features))   # XGBoost importances
    mask_matrix = np.zeros((n_clients, n_features))   # GA+PSO binary masks

    for i, client in enumerate(clients):
        if client.selector_ is not None and client.selector_.xgb_imp_ is not None:
            imp_matrix[i]  = client.selector_.xgb_imp_
        if client.feature_mask_ is not None:
            mask_matrix[i] = client.feature_mask_.astype(float)

    # ------------------------------------------------------------------
    # Plot 1: XGBoost feature importance (top 30, mean ± std across clients)
    # ------------------------------------------------------------------
    mean_imp = imp_matrix.mean(axis=0)
    std_imp  = imp_matrix.std(axis=0)

    top30_idx   = np.argsort(mean_imp)[::-1][:30]
    top30_names = [feature_cols[i] for i in top30_idx]
    top30_mean  = mean_imp[top30_idx]
    top30_std   = std_imp[top30_idx]

    fig, ax = plt.subplots(figsize=(14, 5))
    x = np.arange(30)
    ax.bar(x, top30_mean, yerr=top30_std, capsize=3, color='steelblue', alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(top30_names, rotation=60, ha='right', fontsize=7)
    ax.set_ylabel('Mean XGBoost Importance')
    ax.set_title('Top-30 Feature Importances (XGBoost surrogate, mean ± std across clients)')
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()

    path1 = os.path.join(results_dir, f'{experiment}_xgb_importance.png')
    plt.savefig(path1, dpi=150)
    plt.close()
    print(f"XGBoost importance plot saved: {path1}")

    # ------------------------------------------------------------------
    # Plot 2: Feature selection heatmap (clients × features)
    # ------------------------------------------------------------------
    # Sort features by selection frequency (most selected on left)
    freq        = mask_matrix.mean(axis=0)
    sorted_idx  = np.argsort(freq)[::-1]
    sorted_mask = mask_matrix[:, sorted_idx]
    sorted_names = [feature_cols[i] for i in sorted_idx]

    fig, ax = plt.subplots(figsize=(16, 4))
    im = ax.imshow(sorted_mask, aspect='auto', cmap='Blues', vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.02, label='Selected (1) / Not selected (0)')

    ax.set_yticks(range(n_clients))
    ax.set_yticklabels([f'Client {i+1}' for i in range(n_clients)], fontsize=8)
    ax.set_xlabel('Features (sorted by selection frequency)')
    ax.set_title('GA+PSO Feature Selection per Client')

    # Only label every Nth feature to avoid crowding
    step = max(1, n_features // 20)
    ax.set_xticks(range(0, n_features, step))
    ax.set_xticklabels(
        [sorted_names[i] for i in range(0, n_features, step)],
        rotation=60, ha='right', fontsize=6,
    )

    # Annotate selection frequency on a second x-axis
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(range(0, n_features, step))
    ax2.set_xticklabels(
        [f'{freq[sorted_idx[i]]*100:.0f}%' for i in range(0, n_features, step)],
        fontsize=6,
    )
    ax2.set_xlabel('Selection frequency across clients')

    plt.tight_layout()
    path2 = os.path.join(results_dir, f'{experiment}_gapso_heatmap.png')
    plt.savefig(path2, dpi=150)
    plt.close()
    print(f"GA+PSO heatmap saved: {path2}")

    # ------------------------------------------------------------------
    # Summary: how many features each client selected
    # ------------------------------------------------------------------
    print("\n   [Feature Analysis] Per-client feature counts:")
    for i, client in enumerate(clients):
        n_sel = int(mask_matrix[i].sum())
        fit   = client.selector_.best_fitness_ if client.selector_ else float('nan')
        print(f"      Client {i+1:>2d}: {n_sel:>3d}/{n_features} features selected  "
              f"(fitness={fit:.4f})")
    print(f"   Consensus: {freq.mean()*100:.1f}% of features selected on average per client")


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    start = time.time()
    torch.manual_seed(CONFIG['random_state'])
    np.random.seed(CONFIG['random_state'])

    device = torch.device(CONFIG.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    print(f"\n{'='*60}")
    print(f" FL-AdroidMaLD  Phase 1")
    print(f" Device   : {device}")
    print(f" Rounds   : {CONFIG['n_rounds']}  (warmup={CONFIG['warmup_rounds']})")
    print(f" Clients  : {CONFIG['n_clients']}")
    print(f" Dist     : {CONFIG['distribution']}")
    print(f"{'='*60}\n")

    os.makedirs(CONFIG['results_dir'], exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load data and global holdout split
    # ------------------------------------------------------------------
    print("[1/4] Loading dataset ...")
    X, y, feature_cols, class_names, le = load_and_prepare(CONFIG['data_path'])
    X_train, X_test, y_train, y_test = global_holdout_split(
        X, y, random_state=CONFIG['random_state']
    )
    # Global class distribution — used by AOA contribution scoring
    global_class_dist = (np.bincount(y_train, minlength=len(class_names)).astype(float)
                         / len(y_train))
    print(f"      Train={len(y_train)}  Test={len(y_test)}  "
          f"Features={X_train.shape[1]}  Classes={len(class_names)}")

    # ------------------------------------------------------------------
    # 2. Distribute training data to clients
    # ------------------------------------------------------------------
    print(f"\n[2/4] Distributing data ({CONFIG['distribution']}) ...")
    distributions = get_all_distributions(
        X_train, y_train,
        n_clients=CONFIG['n_clients'],
        random_state=CONFIG['random_state'],
    )
    client_data = distributions[CONFIG['distribution']]
    for i, cd in enumerate(client_data):
        print(f"      Client {i+1:>2d}: {cd['n']:>5d} samples")

    # ------------------------------------------------------------------
    # 3. Initialise server, clients, evaluator
    # ------------------------------------------------------------------
    print("\n[3/4] Initialising FL components ...")
    input_dim   = X_train.shape[1]
    num_classes = len(class_names)

    server = FLServer(
        input_dim     = input_dim,
        num_classes   = num_classes,
        warmup_rounds = CONFIG['warmup_rounds'],
        device        = device,
    )

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

    # ------------------------------------------------------------------
    # 4. FL training loop
    # ------------------------------------------------------------------
    print("\n[4/4] Starting FL training ...\n")
    ensemble_built   = False
    masks_computed   = False
    last_updates     = None   # stores previous round's updates for AOA scoring
    mask_trigger     = CONFIG['warmup_rounds'] - CONFIG['mask_warmup_rounds']

    for rnd in range(1, CONFIG['n_rounds'] + 1):
        phase     = server.phase
        t_round   = time.time()
        all_idx   = list(range(CONFIG['n_clients']))

        # AOA client selection (ensemble phase only)
        selected_idx = server.select_clients(
            all_indices       = all_idx,
            client_updates    = last_updates,
            global_class_dist = global_class_dist,
            k_select          = CONFIG['k_select'],
            random_state      = CONFIG['random_state'],
        )

        print(f"{'─'*60}")
        print(f" Round {rnd:>3d}/{CONFIG['n_rounds']}  [{phase.upper()}]")

        # ---- Client training ----
        client_updates = []

        if phase == 'warmup':
            global_weights = server.get_global_weights()
            for idx in selected_idx:
                upd = clients[idx].train(
                    global_weights = global_weights,
                    n_epochs       = CONFIG['n_epochs'],
                    lr             = CONFIG['lr'],
                    batch_size     = CONFIG['batch_size'],
                    phase          = 'warmup',
                )
                client_updates.append(upd)
                print(f"   Client {idx+1:>2d}: val_f1={upd['val_f1']*100:.1f}%  "
                      f"val_acc={upd['val_acc']*100:.1f}%")

            # ---- Server aggregation ----
            server.aggregate_warmup(client_updates)

            # ---- Build ensemble at warmup→ensemble transition ----
            if server.phase == 'ensemble' and not ensemble_built:
                print("\n   [Transition] Running GA+PSO feature selection per client ...")
                for client in clients:
                    client.compute_feature_mask(
                        n_ga_gen     = 10,
                        pop_size     = 8,
                        n_pso_iter   = 10,
                        random_state = CONFIG['random_state'],
                    )
                masks_computed = True
                plot_feature_analysis(
                    clients     = clients,
                    feature_cols = list(feature_cols),
                    results_dir  = CONFIG['results_dir'],
                    experiment   = CONFIG['experiment'],
                )
                print("\n   [Transition] Building FedEL ensemble ...")
                server.build_ensemble(client_updates)
                ensemble_built = True

        else:  # ensemble phase
            ensemble_info = server.get_ensemble_info()
            for idx in selected_idx:
                upd = clients[idx].train_ensemble(
                    ensemble_info = ensemble_info,
                    n_epochs      = CONFIG['n_epochs'],
                    lr            = CONFIG['lr'],
                    batch_size    = CONFIG['batch_size'],
                )
                client_updates.append(upd)
                print(f"   Client {idx+1:>2d}: val_f1={upd['val_f1']*100:.1f}%  "
                      f"val_acc={upd['val_acc']*100:.1f}%")

            # ---- Server aggregation (meta_FC only) ----
            server.aggregate_ensemble(client_updates)

        last_updates = client_updates   # used by AOA next round

        # ---- Global evaluation ----
        eval_model = server.get_eval_model()
        metrics    = evaluator.evaluate(
            model     = eval_model,
            device    = device,
            round_num = rnd,
            phase     = phase,
        )

        elapsed = time.time() - t_round
        print(f" Round time: {elapsed:.1f}s")

    # ------------------------------------------------------------------
    # Final outputs
    # ------------------------------------------------------------------
    evaluator.plot_convergence()
    evaluator.plot_confusion_matrix(round_num=-1)
    evaluator.plot_per_class_metrics(round_num=-1)

    total = time.time() - start
    print(f"\n{'='*60}")
    print(f" Training complete in {total/60:.1f} minutes")
    print(f" Best F1-macro: {evaluator.best_f1*100:.2f}%")
    print(f" Results saved: {CONFIG['results_dir']}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
