"""
GWO and PSO client selectors for FL — same fitness function as AOA.

Both inherit AOAClientSelector for score computation and fitness evaluation.
Only the search strategy (select() method) differs.

GWO: Mirjalili et al. (2014). Grey Wolf Optimizer. Advances in Engineering
     Software, 69, 46-61.

PSO: Kennedy & Eberhart (1995). Particle swarm optimization. ICNN 1995.
     Binary transfer: sigmoid velocity → top-k selection.
"""

import numpy as np
from aoa_selector import AOAClientSelector, _to_binary


class GWOClientSelector(AOAClientSelector):
    """
    Grey Wolf Optimizer for FL client selection.

    Three social hierarchy levels guide the search:
        α (alpha) — best solution found
        β (beta)  — second best
        δ (delta) — third best

    Each wolf updates its position by averaging the pulls from all three
    leaders. Continuous positions are binarised with top-k selection.

    All fitness computation is inherited from AOAClientSelector.
    """

    def select(self, scores: np.ndarray, class_dists: np.ndarray = None) -> list:
        if self.n_clients <= self.k_select:
            return list(range(self.n_clients))
        if class_dists is None:
            class_dists = getattr(self, '_last_class_dists', None)

        pop     = self._init_population()                                       # (pop, n)
        fitness = np.array([self._fitness(p, scores, class_dists) for p in pop])

        # Initialise leaders from first population
        order = np.argsort(-fitness)
        alpha, alpha_fit = pop[order[0]].copy(), fitness[order[0]]
        beta             = pop[order[1]].copy() if self.pop_size > 1 else alpha.copy()
        delta            = pop[order[2]].copy() if self.pop_size > 2 else beta.copy()

        for t in range(self.max_iter):
            a = 2.0 - 2.0 * t / max(self.max_iter - 1, 1)   # linearly 2 → 0

            new_pop = np.zeros_like(pop)
            for i in range(self.pop_size):
                step_sum = np.zeros(self.n_clients)
                for leader in (alpha, beta, delta):
                    r1  = self.rng.rand(self.n_clients)
                    r2  = self.rng.rand(self.n_clients)
                    A   = 2.0 * a * r1 - a
                    C   = 2.0 * r2
                    D   = np.abs(C * leader - pop[i])
                    step_sum += leader - A * D
                new_cont   = np.clip(step_sum / 3.0, 0.0, 1.0)
                new_pop[i] = _to_binary(new_cont, self.k_select)

            fitness = np.array([self._fitness(p, scores, class_dists) for p in new_pop])
            pop     = new_pop

            # Update leaders
            order = np.argsort(-fitness)
            if fitness[order[0]] > alpha_fit:
                alpha_fit = fitness[order[0]]
            alpha = pop[order[0]].copy()
            beta  = pop[order[1]].copy() if self.pop_size > 1 else alpha.copy()
            delta = pop[order[2]].copy() if self.pop_size > 2 else beta.copy()

        return np.where(alpha == 1)[0].tolist()


class PSOClientSelector(AOAClientSelector):
    """
    Binary PSO for FL client selection.

    Standard PSO velocity update with inertia weight decay (0.9 → 0.4).
    Sigmoid transfer maps velocity to selection probabilities; top-k
    binarisation ensures exactly k clients are selected each round.

    All fitness computation is inherited from AOAClientSelector.
    """

    def select(self, scores: np.ndarray, class_dists: np.ndarray = None) -> list:
        if self.n_clients <= self.k_select:
            return list(range(self.n_clients))
        if class_dists is None:
            class_dists = getattr(self, '_last_class_dists', None)

        pop       = self._init_population().astype(float)                       # (pop, n)
        pbest     = pop.copy()
        fit       = np.array([self._fitness(p, scores, class_dists) for p in pop])
        pbest_fit = fit.copy()

        gbest_i   = int(np.argmax(fit))
        gbest     = pbest[gbest_i].copy()
        gbest_fit = pbest_fit[gbest_i]

        vel   = self.rng.uniform(-1.0, 1.0, size=(self.pop_size, self.n_clients))
        c1 = c2 = 2.0
        w_max, w_min = 0.9, 0.4

        for t in range(self.max_iter):
            w  = w_max - (w_max - w_min) * t / max(self.max_iter - 1, 1)
            r1 = self.rng.rand(self.pop_size, self.n_clients)
            r2 = self.rng.rand(self.pop_size, self.n_clients)

            vel = (w * vel
                   + c1 * r1 * (pbest - pop)
                   + c2 * r2 * (gbest - pop))
            vel = np.clip(vel, -4.0, 4.0)

            # Sigmoid → top-k binary
            prob    = 1.0 / (1.0 + np.exp(-vel))
            new_pop = np.array([_to_binary(prob[i], self.k_select)
                                for i in range(self.pop_size)])
            new_fit = np.array([self._fitness(new_pop[i], scores, class_dists)
                                for i in range(self.pop_size)])

            # Update personal bests
            improved             = new_fit > pbest_fit
            pbest[improved]      = new_pop[improved]
            pbest_fit[improved]  = new_fit[improved]

            # Update global best
            best_i = int(np.argmax(pbest_fit))
            if pbest_fit[best_i] > gbest_fit:
                gbest     = pbest[best_i].copy()
                gbest_fit = pbest_fit[best_i]

            pop = new_pop

        return np.where(gbest == 1)[0].tolist()


# ----------------------------------------------------------------------
# Quick validation
# ----------------------------------------------------------------------

if __name__ == '__main__':
    np.random.seed(42)
    n, k, nc = 20, 8, 12
    scores      = np.random.rand(n)
    class_dists = np.random.dirichlet(np.ones(nc), size=n)

    for Cls, name in [(GWOClientSelector, 'GWO'), (PSOClientSelector, 'PSO')]:
        sel = Cls(n_clients=n, k_select=k, pop_size=20, max_iter=30,
                  lam=0.2, random_state=42)
        result = sel.select(scores, class_dists)
        assert len(result) == k and len(set(result)) == k, f"{name}: wrong selection size"
        print(f"{name} validation passed — selected {sorted(result)}")
