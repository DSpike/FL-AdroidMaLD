"""
Federated Learning Data Distribution Module
Android Malware Classification - FL-AdroidMaLD

Handles:
  - Global holdout split (done FIRST, never touched during training)
  - 5 distribution strategies: IID, Dir(1.0), Dir(0.5), Dir(0.1), Pathological
  - N=10 clients
  - Adaptive per-class minimum sample floor
  - Visualization and summary statistics
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from collections import defaultdict
import os

# ─────────────────────────────────────────────
# 1. DATA LOADING
# ─────────────────────────────────────────────

def load_and_prepare(data_path, label_col='Category', random_state=42):
    """
    Load dataset, impute nulls, encode labels.
    Returns X (ndarray), y (ndarray), feature_names, class_names, label_encoder.
    """
    df = pd.read_csv(data_path)

    drop_cols = [c for c in ['Family'] if c in df.columns]
    feature_cols = [c for c in df.columns if c not in [label_col] + drop_cols]

    # Impute nulls with column mean
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(df[feature_cols])

    le = LabelEncoder()
    y = le.fit_transform(df[label_col])

    print(f"Loaded: {X.shape[0]} samples, {X.shape[1]} features, {len(le.classes_)} classes")
    print("Class distribution:")
    for cls, count in zip(le.classes_, np.bincount(y)):
        print(f"  {cls:<20}: {count:>5}")

    return X, y, feature_cols, le.classes_, le


# ─────────────────────────────────────────────
# 2. GLOBAL HOLDOUT SPLIT — DONE FIRST
# ─────────────────────────────────────────────

def global_holdout_split(X, y, test_size=0.2, random_state=42):
    """
    Carve out a stratified global test set before ANY client distribution.
    This set is NEVER seen during federated training.
    Returns: X_train, X_test, y_train, y_test
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    print(f"\nGlobal holdout split:")
    print(f"  Train : {X_train.shape[0]} samples ({100*(1-test_size):.0f}%)")
    print(f"  Test  : {X_test.shape[0]} samples ({100*test_size:.0f}%) — LOCKED")
    return X_train, X_test, y_train, y_test


# ─────────────────────────────────────────────
# 3. ADAPTIVE MINIMUM FLOOR
# ─────────────────────────────────────────────

def _compute_adaptive_floor(y_train, n_clients, desired_min=20):
    """
    Per-class adaptive floor: min(desired_min, total_class_count // n_clients).
    Prevents CV collapse on minority classes (FileInfector=119, Trojan_Banker=123).
    """
    floors = {}
    for cls in np.unique(y_train):
        total = int(np.sum(y_train == cls))
        floors[cls] = max(5, min(desired_min, total // n_clients))
    return floors


def _enforce_floor(client_indices, y_train, floors, random_state=42):
    """
    Ensure each client has at least floors[cls] samples per class.
    Moves surplus samples from over-represented clients to under-represented ones.
    """
    rng = np.random.default_rng(random_state)
    n_clients = len(client_indices)
    client_indices = [list(idx) for idx in client_indices]

    for cls, floor in floors.items():
        cls_all = set(np.where(y_train == cls)[0].tolist())

        for i in range(n_clients):
            client_cls = [idx for idx in client_indices[i] if y_train[idx] == cls]
            deficit = floor - len(client_cls)
            if deficit <= 0:
                continue

            # Collect available surplus from other clients
            pool = []
            for j in range(n_clients):
                if j == i:
                    continue
                j_cls = [idx for idx in client_indices[j] if y_train[idx] == cls]
                surplus = len(j_cls) - floor
                if surplus > 0:
                    pool.extend(j_cls[:surplus])

            if len(pool) == 0:
                continue

            n_take = min(deficit, len(pool))
            chosen = rng.choice(pool, size=n_take, replace=False).tolist()
            chosen_set = set(chosen)

            client_indices[i].extend(chosen)
            for j in range(n_clients):
                if j != i:
                    client_indices[j] = [
                        idx for idx in client_indices[j] if idx not in chosen_set
                    ]

    return [np.array(idx) for idx in client_indices]


def _build_clients(X, y, client_indices):
    return [
        {'X': X[idx], 'y': y[idx], 'n': len(idx)}
        for idx in client_indices
    ]


# ─────────────────────────────────────────────
# 4. DISTRIBUTION STRATEGIES
# ─────────────────────────────────────────────

def distribute_iid(X_train, y_train, n_clients=10, min_samples=20, random_state=42):
    """IID: random shuffle, equal split across all clients."""
    rng = np.random.default_rng(random_state)
    indices = rng.permutation(len(y_train))
    splits = np.array_split(indices, n_clients)
    floors = _compute_adaptive_floor(y_train, n_clients, min_samples)
    splits = _enforce_floor(splits, y_train, floors, random_state)
    return _build_clients(X_train, y_train, splits)


def distribute_dirichlet(X_train, y_train, alpha=0.5, n_clients=10,
                         min_samples=20, random_state=42):
    """
    Non-IID via Dirichlet(alpha).
      alpha=1.0 → mild heterogeneity
      alpha=0.5 → moderate (standard in FL literature)
      alpha=0.1 → severe heterogeneity
    Lower alpha = more skewed class distribution per client.
    """
    rng = np.random.default_rng(random_state)
    client_indices = [[] for _ in range(n_clients)]

    for cls in np.unique(y_train):
        cls_idx = np.where(y_train == cls)[0]
        rng.shuffle(cls_idx)

        proportions = rng.dirichlet(np.ones(n_clients) * alpha)
        proportions = (np.cumsum(proportions) * len(cls_idx)).astype(int)
        proportions = np.clip(proportions, 0, len(cls_idx))

        splits = np.split(cls_idx, proportions[:-1])
        for i, split in enumerate(splits):
            client_indices[i].extend(split.tolist())

    floors = _compute_adaptive_floor(y_train, n_clients, min_samples)
    client_indices = _enforce_floor(client_indices, y_train, floors, random_state)
    return _build_clients(X_train, y_train, client_indices)


def distribute_pathological(X_train, y_train, dominant_k=3, dominant_fraction=0.8,
                             n_clients=10, min_samples=20, random_state=42):
    """
    Pathological non-IID: each client is dominated by dominant_k classes
    (dominant_fraction of its data), with token samples from the rest.
    Classes are assigned in round-robin with shift per client.
    """
    rng = np.random.default_rng(random_state)
    classes = np.unique(y_train)
    n_classes = len(classes)

    shuffled = rng.permutation(classes)
    dominant_map = {}
    for i in range(n_clients):
        dominant_map[i] = set(
            shuffled[(i * dominant_k + j) % n_classes]
            for j in range(dominant_k)
        )

    # Split each class into dominant pool and minor pool
    class_idx = {}
    for cls in classes:
        idx = np.where(y_train == cls)[0].copy()
        rng.shuffle(idx)
        n_dom = int(len(idx) * dominant_fraction)
        class_idx[cls] = {'dominant': idx[:n_dom], 'minor': idx[n_dom:]}

    client_indices = [[] for _ in range(n_clients)]

    for cls in classes:
        # Dominant fraction → owning clients only
        owners = [i for i in range(n_clients) if cls in dominant_map[i]]
        if owners:
            dom_splits = np.array_split(class_idx[cls]['dominant'], len(owners))
            for i, owner in enumerate(owners):
                client_indices[owner].extend(dom_splits[i].tolist())

        # Minor fraction → all clients equally
        min_splits = np.array_split(class_idx[cls]['minor'], n_clients)
        for i in range(n_clients):
            client_indices[i].extend(min_splits[i].tolist())

    floors = _compute_adaptive_floor(y_train, n_clients, min_samples)
    client_indices = _enforce_floor(client_indices, y_train, floors, random_state)
    return _build_clients(X_train, y_train, client_indices)


# ─────────────────────────────────────────────
# 5. GENERATE ALL 5 DISTRIBUTIONS
# ─────────────────────────────────────────────

DISTRIBUTION_CONFIGS = {
    'IID':          {'strategy': 'iid',          'alpha': None,  'label': 'IID'},
    'Dir_1.0':      {'strategy': 'dirichlet',     'alpha': 1.0,   'label': 'Dirichlet α=1.0 (Mild)'},
    'Dir_0.5':      {'strategy': 'dirichlet',     'alpha': 0.5,   'label': 'Dirichlet α=0.5 (Moderate)'},
    'Dir_0.1':      {'strategy': 'dirichlet',     'alpha': 0.1,   'label': 'Dirichlet α=0.1 (Severe)'},
    'Dir_0.05':     {'strategy': 'dirichlet',     'alpha': 0.05,  'label': 'Dirichlet α=0.05 (Extreme)'},
    'Pathological': {'strategy': 'pathological',  'alpha': None,  'label': 'Pathological (K=3)'},
}


def get_all_distributions(X_train, y_train, n_clients=10, min_samples=20, random_state=42):
    """
    Generate all 5 distribution settings.
    Returns dict: {name: list of client dicts}
    """
    print("\nGenerating all distributions...")
    result = {}

    result['IID'] = distribute_iid(
        X_train, y_train, n_clients, min_samples, random_state)

    for alpha in [1.0, 0.5, 0.1, 0.05]:
        key = f'Dir_{alpha}'
        result[key] = distribute_dirichlet(
            X_train, y_train, alpha, n_clients, min_samples, random_state)

    result['Pathological'] = distribute_pathological(
        X_train, y_train, dominant_k=3, dominant_fraction=0.8,
        n_clients=n_clients, min_samples=min_samples, random_state=random_state)

    print("Done. Distributions generated:", list(result.keys()))
    return result


# ─────────────────────────────────────────────
# 6. VERIFICATION — check floor holds
# ─────────────────────────────────────────────

def verify_distributions(distributions, y_train, n_clients=10, class_names=None):
    """
    Print per-distribution statistics and flag any floor violations.
    """
    floors = _compute_adaptive_floor(y_train, n_clients, desired_min=20)
    classes = np.unique(y_train)

    print(f"\n{'='*70}")
    print("DISTRIBUTION VERIFICATION")
    print(f"{'='*70}")

    for dist_name, clients in distributions.items():
        violations = []
        total_samples = [c['n'] for c in clients]
        min_cls_per_client = []

        for i, client in enumerate(clients):
            for cls in classes:
                count = int(np.sum(client['y'] == cls))
                min_cls_per_client.append(count)
                if count < floors[cls]:
                    violations.append((i, cls, count, floors[cls]))

        label = DISTRIBUTION_CONFIGS.get(dist_name, {}).get('label', dist_name)
        print(f"\n{label}")
        print(f"  Client sizes : min={min(total_samples)}, "
              f"max={max(total_samples)}, mean={np.mean(total_samples):.0f}")
        print(f"  Min class/client : {min(min_cls_per_client)}")
        if violations:
            print(f"  FLOOR VIOLATIONS: {len(violations)} "
                  f"(class names where count < floor)")
            for client_id, cls, count, floor in violations[:5]:
                name = class_names[cls] if class_names is not None else cls
                print(f"    Client {client_id+1}, {name}: {count} < floor {floor}")
        else:
            print(f"  Floor check: PASSED")


# ─────────────────────────────────────────────
# 7. VISUALIZATION
# ─────────────────────────────────────────────

def plot_distribution(clients, class_names, title='', save_path=None):
    """Heatmap of class distribution across clients (% of each class held by each client)."""
    n_clients = len(clients)
    n_classes = len(class_names)

    count_matrix = np.zeros((n_clients, n_classes))
    for i, client in enumerate(clients):
        for j in range(n_classes):
            count_matrix[i, j] = np.sum(client['y'] == j)

    # Normalise per class so colours show where each class lives
    pct_matrix = count_matrix / (count_matrix.sum(axis=0, keepdims=True) + 1e-8) * 100

    fig, axes = plt.subplots(1, 2, figsize=(18, 5))

    # Left: heatmap (% of class)
    sns.heatmap(pct_matrix,
                xticklabels=[n[:9] for n in class_names],
                yticklabels=[f'C{i+1}' for i in range(n_clients)],
                annot=True, fmt='.0f', cmap='YlOrRd',
                linewidths=0.3, ax=axes[0])
    axes[0].set_title(f'{title}\n% of each class held by client')
    axes[0].set_xlabel('Class')
    axes[0].set_ylabel('Client')
    axes[0].tick_params(axis='x', rotation=45)

    # Right: stacked bar (sample counts)
    bottom = np.zeros(n_clients)
    colors = plt.cm.tab20(np.linspace(0, 1, n_classes))
    for j in range(n_classes):
        axes[1].bar(range(n_clients), count_matrix[:, j], bottom=bottom,
                    color=colors[j], label=class_names[j][:10])
        bottom += count_matrix[:, j]
    axes[1].set_xticks(range(n_clients))
    axes[1].set_xticklabels([f'C{i+1}' for i in range(n_clients)])
    axes[1].set_title(f'{title}\nSample counts per client')
    axes[1].set_xlabel('Client')
    axes[1].set_ylabel('Samples')
    axes[1].legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=7)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    return fig


def plot_all_distributions(distributions, class_names, save_dir=None):
    """Plot all 5 distributions side by side for comparison."""
    figs = {}
    for dist_name, clients in distributions.items():
        label = DISTRIBUTION_CONFIGS.get(dist_name, {}).get('label', dist_name)
        save_path = None
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f'dist_{dist_name}.png')
        figs[dist_name] = plot_distribution(clients, class_names, label, save_path)
    return figs


def print_distribution_table(distributions, class_names):
    """Print concise per-distribution summary table."""
    print(f"\n{'Distribution':<20} {'Min/client':>12} {'Max/client':>12} "
          f"{'Min cls count':>15} {'Heterogeneity':>15}")
    print('-' * 76)

    for dist_name, clients in distributions.items():
        sizes = [c['n'] for c in clients]
        all_cls_counts = []
        for client in clients:
            for cls in np.unique(client['y']):
                all_cls_counts.append(int(np.sum(client['y'] == cls)))

        # Heterogeneity: std of class proportions across clients
        n_classes = len(class_names)
        prop_matrix = np.zeros((len(clients), n_classes))
        for i, client in enumerate(clients):
            total = len(client['y'])
            for j in range(n_classes):
                prop_matrix[i, j] = np.sum(client['y'] == j) / total
        heterogeneity = np.mean(np.std(prop_matrix, axis=0))

        label = DISTRIBUTION_CONFIGS.get(dist_name, {}).get('label', dist_name)
        print(f"{label:<20} {min(sizes):>12} {max(sizes):>12} "
              f"{min(all_cls_counts):>15} {heterogeneity:>15.4f}")


# ─────────────────────────────────────────────
# 8. SAVE / LOAD DISTRIBUTIONS
# ─────────────────────────────────────────────

def save_distributions(distributions, X_train, y_train, save_dir):
    """Save each client's data to CSV for reproducibility."""
    os.makedirs(save_dir, exist_ok=True)
    for dist_name, clients in distributions.items():
        dist_dir = os.path.join(save_dir, dist_name)
        os.makedirs(dist_dir, exist_ok=True)
        for i, client in enumerate(clients):
            df = pd.DataFrame(client['X'])
            df['label'] = client['y']
            path = os.path.join(dist_dir, f'client_{i+1}.csv')
            df.to_csv(path, index=False)
    print(f"Distributions saved to: {save_dir}")


# ─────────────────────────────────────────────
# MAIN — standalone test
# ─────────────────────────────────────────────

if __name__ == '__main__':
    DATA_PATH = r'C:\Users\Dspike\Documents\FL-AdroidMaLD\normalized_dataset\combined_dataset.csv'
    RESULTS_DIR = r'C:\Users\Dspike\Documents\FL-AdroidMaLD\results\distributions'
    N_CLIENTS = 10

    # Step 1: load
    X, y, feature_names, class_names, le = load_and_prepare(DATA_PATH)

    # Step 2: global holdout — FIRST
    X_train, X_test, y_train, y_test = global_holdout_split(X, y, test_size=0.2)

    print(f"\nGlobal test set locked: {X_test.shape[0]} samples")
    print("Training set for FL: ", X_train.shape[0], "samples")

    # Step 3: generate all 5 distributions
    distributions = get_all_distributions(X_train, y_train, n_clients=N_CLIENTS)

    # Step 4: verify
    verify_distributions(distributions, y_train, N_CLIENTS, class_names)

    # Step 5: summary table
    print_distribution_table(distributions, class_names)

    # Step 6: visualize
    plot_all_distributions(distributions, class_names, save_dir=RESULTS_DIR)
    plt.show()

    print("\nDone. Global test set shape:", X_test.shape)
    print("Use X_test, y_test ONLY for final evaluation.")
