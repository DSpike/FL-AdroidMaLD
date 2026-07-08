"""
DivFL client selector for FL.

Reference: Balakrishnan et al. (2022). "Diverse client selection for
federated learning via submodular maximization." ICLR 2022.

Algorithm: greedy facility location maximization over gradient proxy vectors.

Gradient proxy = flatten(w_local_i - w_global) across all shared parameters,
then optionally random-projected to PROJ_DIM dimensions for efficiency.

Selection objective:
    F(S) = sum_{i=1}^{n} max_{j in S} sim(grad_i, grad_j)
    sim(i, j) = (1 + cosine_similarity(grad_i, grad_j)) / 2   in [0, 1]

Greedy: iteratively add the client j* that maximises marginal coverage gain.
Unobserved clients (not in prev_updates) are imputed with the mean gradient
vector from the observed clients of the same round.
"""

import numpy as np

PROJ_DIM    = 512     # target dimension after random projection
_PROJ_SEED  = 0       # fixed so the projection matrix is round-reproducible

# Module-level cache: total_params → projection matrix.
# Rebuilt only when model architecture changes between experiments.
_proj_cache: dict = {}


def _get_proj(total_params: int) -> np.ndarray:
    """Return (PROJ_DIM × total_params) random projection matrix (cached)."""
    if total_params not in _proj_cache:
        rng = np.random.RandomState(_PROJ_SEED)
        P = rng.randn(PROJ_DIM, total_params).astype(np.float32)
        P /= np.sqrt(PROJ_DIM)   # normalise so E[||Px||^2] = ||x||^2
        _proj_cache[total_params] = P
    return _proj_cache[total_params]


def _build_grad_proxy(update: dict, global_weights: dict,
                      proj: np.ndarray = None) -> np.ndarray:
    """
    Build gradient proxy for one client: flatten(w_local - w_global),
    then project if proj is not None.
    """
    local = update.get('weights') or update.get('meta_state', {})
    diffs = []
    for key in global_weights:
        g = global_weights[key].float().cpu().numpy().ravel()
        l = local[key].float().cpu().numpy().ravel() if key in local else np.zeros_like(g)
        diffs.append(l - g)
    vec = np.concatenate(diffs)
    return (proj @ vec) if proj is not None else vec


class DivFLClientSelector:
    """
    Greedy facility location client selector.

    Parameters
    ----------
    n_clients : int
        Total number of available clients.
    k_select : int
        Number of clients to select per round.
    proj_dim : int
        Projection dimension for gradient vectors (default: PROJ_DIM=512).
        Set to None to disable projection (only safe for very small models).
    """

    def __init__(self, n_clients: int, k_select: int, proj_dim: int = PROJ_DIM):
        self.n_clients = n_clients
        self.k_select  = k_select
        self.proj_dim  = proj_dim

    def select(self, prev_updates: list, global_weights: dict) -> list:
        """
        Select k clients that maximise facility location coverage.

        Parameters
        ----------
        prev_updates : list of update dicts
            Each dict must have 'client_id' (1-indexed) and 'weights' keys.
        global_weights : dict
            Current global model state dict (PyTorch tensors).

        Returns
        -------
        list of selected 0-indexed client indices, sorted ascending.
        """
        n, k = self.n_clients, self.k_select
        if n <= k:
            return list(range(n))

        # Decide whether to project
        total_params = sum(v.numel() for v in global_weights.values())
        proj = _get_proj(total_params) if (
            self.proj_dim is not None and total_params > self.proj_dim
        ) else None
        dim = self.proj_dim if proj is not None else total_params

        # Build gradient proxies for observed clients
        obs: dict = {}
        for u in prev_updates:
            cid = u.get('client_id', -1) - 1   # 0-indexed
            if 0 <= cid < n:
                obs[cid] = _build_grad_proxy(u, global_weights, proj)

        # Impute unobserved clients with mean gradient (neutral direction)
        mean_vec = (np.mean(list(obs.values()), axis=0)
                    if obs else np.zeros(dim, dtype=np.float32))
        vecs = np.array(
            [obs.get(i, mean_vec) for i in range(n)], dtype=np.float32
        )   # (n, dim)

        # Cosine similarity matrix mapped to [0, 1]
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms = np.where(norms < 1e-10, 1e-10, norms)
        vn  = vecs / norms                          # unit vectors (n, dim)
        sim = (1.0 + vn @ vn.T) / 2.0              # (n, n), values in [0, 1]

        # Greedy facility location
        selected: list = []
        coverage = np.zeros(n, dtype=np.float64)

        for _ in range(k):
            best_j, best_gain = -1, -1.0
            for j in range(n):
                if j in selected:
                    continue
                # Marginal gain from adding client j
                gain = float(np.sum(np.maximum(coverage, sim[:, j]) - coverage))
                if gain > best_gain:
                    best_gain, best_j = gain, j
            assert best_j >= 0, "DivFL: no client selected (check k <= n)"
            selected.append(best_j)
            coverage = np.maximum(coverage, sim[:, best_j])

        return sorted(selected)


# ---------------------------------------------------------------------------
# Quick validation
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import torch

    np.random.seed(42)
    torch.manual_seed(42)

    n, k, n_classes = 20, 8, 12

    # Toy global weights (matches CNN-GRU scale roughly)
    global_weights = {
        'conv1.weight': torch.randn(32, 1, 3),
        'fc.weight':    torch.randn(n_classes, 64),
    }
    total_params = sum(v.numel() for v in global_weights.values())

    # Fake updates: 15 of 20 clients have been observed (phase 2 scenario)
    observed = list(range(15))
    prev_updates = []
    for i in observed:
        local_w = {k: v + torch.randn_like(v) * 0.1 for k, v in global_weights.items()}
        prev_updates.append({'client_id': i + 1, 'weights': local_w})

    sel = DivFLClientSelector(n_clients=n, k_select=k)
    result = sel.select(prev_updates, global_weights)

    assert len(result) == k,             f"Expected {k} clients, got {len(result)}"
    assert len(set(result)) == k,        "Duplicate clients in selection"
    assert all(0 <= c < n for c in result), "Out-of-range client index"

    print(f"DivFL validation passed — selected {result}")
    print(f"  n={n}, k={k}, observed={len(observed)}, "
          f"imputed={n - len(observed)}, proj_dim={PROJ_DIM}")
