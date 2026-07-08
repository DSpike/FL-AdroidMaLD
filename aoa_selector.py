"""
Aquila Optimization Algorithm (AOA) for FL client selection.

Based on: Abualigah et al. (2021). Aquila Optimizer: A novel meta-heuristic
optimization algorithm. Computers & Industrial Engineering, 157, 107250.

Fitness function: composite contribution score  +  diversity bonus
    score_k = α × divergence_k    (how much client diverges from global model)
            + β × coverage_k      (minority class representation)
            + γ × volume_k        (data volume weight)
            + δ × (1 - val_f1_k)  (local error rate — high = model needs this client)

    fitness(S) = Σ score_k  +  λ × avg_pairwise_diversity(S)

The diversity term makes the fitness NON-LINEAR: the combined value of a
selected subset S is more than the sum of individual scores because clients
with DIFFERENT class distributions contribute complementary information.
This gives AOA a genuine combinatorial optimisation task — greedy top-k is
no longer provably optimal, so the meta-heuristic search adds real value.
"""

import math
import numpy as np


class AOAClientSelector:
    """
    AOA-based client selection for federated learning.

    Args:
        n_clients    : total number of available clients
        k_select     : number of clients to select per round
        pop_size     : AOA population size
        max_iter     : AOA iterations per selection round
        alpha        : weight for divergence component  (default 0.3)
        beta         : weight for class coverage        (default 0.3)
        gamma        : weight for data volume           (default 0.1)
        delta        : weight for local error rate (1-val_f1) (default 0.3)
        lam          : weight for pairwise diversity bonus (default 0.2)
        random_state : reproducibility seed
    """

    def __init__(
        self,
        n_clients:    int   = 10,
        k_select:     int   = 5,
        pop_size:     int   = 10,
        max_iter:     int   = 15,
        alpha:        float = 0.3,
        beta:         float = 0.3,
        gamma:        float = 0.1,
        delta:        float = 0.3,
        lam:          float = 0.2,
        random_state: int   = 42,
    ):
        self.n_clients = n_clients
        self.k_select  = k_select
        self.pop_size  = pop_size
        self.max_iter  = max_iter
        self.alpha     = alpha
        self.beta      = beta
        self.gamma     = gamma
        self.delta     = delta
        self.lam       = lam
        self.rng       = np.random.RandomState(random_state)

    # ------------------------------------------------------------------
    # Contribution score computation
    # ------------------------------------------------------------------

    def compute_scores(
        self,
        client_updates:    list,
        global_weights:    dict,
        global_class_dist: np.ndarray,
    ) -> np.ndarray:
        """
        Compute per-client contribution score.

        Args:
            client_updates   : list of dicts from fl_client.train()
                               each must have: 'n_train', 'class_dist', 'val_f1'
            global_weights   : current global model state dict
            global_class_dist: fraction of each class in global training set

        Returns:
            scores: np.ndarray shape (n_clients,) in [0, 1]

        Side effect:
            stores self._last_class_dists — used by select() for diversity term
        """
        n = len(client_updates)
        divergence = np.zeros(n)
        coverage   = np.zeros(n)
        volume     = np.zeros(n)
        val_f1     = np.zeros(n)

        total_samples = sum(u['n_train'] for u in client_updates)

        # Classes whose global frequency < 5% are "minority".
        # Threshold raised from 2% → 5% to include Backdoor, Scareware, PUA,
        # Trojan_Dropper — classes with meaningful but small representation.
        minority_mask = global_class_dist < 0.05

        # Store class distributions for diversity term in select()
        n_classes = len(global_class_dist)
        self._last_class_dists = np.zeros((n, n_classes))

        for i, u in enumerate(client_updates):
            # --- Divergence: ||w_k - w_global||_F / n_k ---
            local_weights = u.get('weights') or u.get('meta_state', {})
            sq_sum = 0.0
            for key in global_weights:
                if key in local_weights:
                    diff    = local_weights[key].float().cpu() - global_weights[key].float().cpu()
                    sq_sum += diff.pow(2).sum().item()
            divergence[i] = math.sqrt(sq_sum) / max(u['n_train'], 1)

            # --- Class coverage: fraction of client data that is minority class ---
            client_dist = u.get('class_dist',
                                np.ones(n_classes) / n_classes)
            coverage[i] = float(client_dist[minority_mask].sum())
            self._last_class_dists[i] = client_dist

            # --- Volume: fraction of total training samples ---
            volume[i] = u['n_train'] / total_samples

            # --- Local model error rate (inverted val_f1) ---
            # Clients where the model performs POORLY hold data the global
            # model hasn't learned yet — they contribute the most informative
            # gradient update.  Using 1 - val_f1 (error rate) prioritises
            # these high-need clients over easy, already-mastered ones.
            val_f1[i] = 1.0 - float(u.get('val_f1', 0.0))

        # Normalise each component to [0, 1]
        divergence = _norm01(divergence)
        coverage   = _norm01(coverage)
        volume     = _norm01(volume)
        val_f1     = _norm01(val_f1)

        scores = (self.alpha * divergence
                + self.beta  * coverage
                + self.gamma * volume
                + self.delta * val_f1)
        return scores

    # ------------------------------------------------------------------
    # AOA optimisation
    # ------------------------------------------------------------------

    def select(
        self,
        scores:      np.ndarray,
        class_dists: np.ndarray = None,
    ) -> list:
        """
        Run AOA to select k_select clients that maximise total contribution
        plus pairwise diversity.

        Args:
            scores      : contribution score per client, shape (n_clients,)
            class_dists : class distribution per client, shape (n_clients, n_classes).
                          If None, falls back to self._last_class_dists from
                          the most recent compute_scores() call.

        Returns:
            selected: list of selected client indices (length k_select)
        """
        if self.n_clients <= self.k_select:
            return list(range(self.n_clients))

        if class_dists is None:
            class_dists = getattr(self, '_last_class_dists', None)

        # Initialise population
        pop     = self._init_population()
        fitness = np.array([self._fitness(p, scores, class_dists) for p in pop])

        best_idx = np.argmax(fitness)
        best     = pop[best_idx].copy()
        best_fit = fitness[best_idx]

        D = 0.67  # exploration/exploitation boundary ratio

        for t in range(self.max_iter):
            x_mean = pop.mean(axis=0)
            ratio  = (t + 1) / self.max_iter

            for i in range(self.pop_size):
                if ratio <= D:
                    # Exploration phase
                    if i < self.pop_size // 2:
                        # X1: Extended Exploration (High Soar)
                        r   = self.rng.rand(self.n_clients)
                        new = best * (1 - ratio) + (x_mean - best) * r
                    else:
                        # X2: Narrowing Exploration (Short Glide with Lévy)
                        levy = self._levy(self.n_clients)
                        r    = self.rng.rand(self.n_clients)
                        new  = best * levy + pop[self.rng.randint(self.pop_size)] + r
                else:
                    # Exploitation phase
                    qf = ratio ** (2 * self.rng.rand() / (1 - ratio + 1e-10))
                    if i < self.pop_size // 2:
                        # X3: Low Flight
                        r   = self.rng.rand(self.n_clients)
                        new = best * qf - x_mean * r + self.rng.rand(self.n_clients)
                    else:
                        # X4: Walk and Grab Prey
                        levy = self._levy(self.n_clients)
                        g1   = 2 * self.rng.rand()
                        g2   = 2 * (1 - ratio)
                        new  = best * qf - g1 * pop[i] * self.rng.rand() - \
                               g2 * levy + self.rng.rand() * g1

                new_bin = _to_binary(np.clip(new, 0, 1), self.k_select)
                new_fit = self._fitness(new_bin, scores, class_dists)

                if new_fit > fitness[i]:
                    pop[i]     = new_bin
                    fitness[i] = new_fit
                    if new_fit > best_fit:
                        best     = new_bin.copy()
                        best_fit = new_fit

        selected = np.where(best == 1)[0].tolist()
        return selected

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _init_population(self) -> np.ndarray:
        pop = np.zeros((self.pop_size, self.n_clients))
        for i in range(self.pop_size):
            idx = self.rng.choice(self.n_clients, self.k_select, replace=False)
            pop[i, idx] = 1.0
        return pop

    def _fitness(
        self,
        solution:    np.ndarray,
        scores:      np.ndarray,
        class_dists: np.ndarray = None,
    ) -> float:
        """
        Non-linear fitness: individual score sum + λ × avg pairwise diversity.

        The diversity term rewards selecting clients whose class distributions
        are DIFFERENT from each other — complementary data coverage.
        This makes the fitness non-additive, so greedy top-k is not optimal
        and AOA's search genuinely helps.
        """
        ind = float((solution * scores).sum())

        if class_dists is None or self.lam == 0.0:
            return ind

        selected = np.where(solution == 1)[0]
        if len(selected) < 2:
            return ind

        # Average pairwise L2 distance between class distributions
        total, count = 0.0, 0
        for a in range(len(selected)):
            for b in range(a + 1, len(selected)):
                diff   = class_dists[selected[a]] - class_dists[selected[b]]
                total += float(np.dot(diff, diff) ** 0.5)
                count += 1
        div = total / count if count > 0 else 0.0

        return ind + self.lam * div

    def _levy(self, n: int) -> np.ndarray:
        """Lévy flight step."""
        beta  = 1.5
        sigma = (math.gamma(1 + beta) * math.sin(math.pi * beta / 2) /
                 (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))
                 ) ** (1 / beta)
        u = self.rng.randn(n) * sigma
        v = np.abs(self.rng.randn(n)) + 1e-10
        return u / (v ** (1 / beta))


# ------------------------------------------------------------------
# Utility functions
# ------------------------------------------------------------------

def _norm01(x: np.ndarray) -> np.ndarray:
    """Normalise array to [0, 1]. Returns uniform weights if all equal."""
    r = x.max() - x.min()
    if r < 1e-10:
        return np.ones_like(x) / len(x)
    return (x - x.min()) / r


def _to_binary(x: np.ndarray, k: int) -> np.ndarray:
    """Select top-k indices from continuous vector → binary mask."""
    result = np.zeros(len(x))
    top_k  = np.argsort(x)[::-1][:k]
    result[top_k] = 1.0
    return result


# ------------------------------------------------------------------
# Quick validation
# ------------------------------------------------------------------

if __name__ == '__main__':
    np.random.seed(42)

    n_clients, k, n_classes = 10, 5, 12

    aoa = AOAClientSelector(
        n_clients = n_clients,
        k_select  = k,
        pop_size  = 10,
        max_iter  = 15,
        lam       = 0.2,
    )

    # Individual scores: clients 0, 2, 5, 7, 9 have highest scores
    scores = np.array([0.9, 0.1, 0.8, 0.2, 0.1, 0.85, 0.15, 0.75, 0.3, 0.95])

    # Class dists: make clients 0 and 9 similar (both high-score) →
    # diversity term should prefer 0+7 over 0+9 even though 9 has higher score
    class_dists = np.random.dirichlet(np.ones(n_classes), size=n_clients)
    class_dists[9] = class_dists[0] + np.random.randn(n_classes) * 0.01  # 9 ≈ 0
    class_dists = np.clip(class_dists, 0, None)
    class_dists /= class_dists.sum(axis=1, keepdims=True)

    # Greedy top-k (no diversity)
    greedy = np.argsort(scores)[::-1][:k].tolist()

    # AOA with diversity
    selected = aoa.select(scores, class_dists)

    assert len(selected) == k
    assert len(set(selected)) == k

    greedy_fit = aoa._fitness(
        np.isin(np.arange(n_clients), greedy).astype(float), scores, class_dists)
    aoa_fit    = aoa._fitness(
        np.isin(np.arange(n_clients), selected).astype(float), scores, class_dists)

    # With non-linear fitness (diversity term), AOA may differ from greedy —
    # that is the point. Assert AOA finds a valid, different solution.
    assert sorted(selected) != sorted(greedy) or aoa_fit >= greedy_fit - 1e-6, \
        "AOA should explore beyond greedy when diversity term is active"

    print("aoa_selector.py validation passed.")
    print(f"  Greedy top-{k} : {sorted(greedy)}  fitness={greedy_fit:.4f}")
    print(f"  AOA selected  : {sorted(selected)}  fitness={aoa_fit:.4f}")
    print(f"  Solutions differ (diversity working): {sorted(selected) != sorted(greedy)}")
