"""
GA+PSO Hybrid Feature Selector with XGBoost-guided initialisation.

Design decisions:
    - Sequential hybrid: GA (exploration) → PSO (exploitation)
    - XGBoost feature importances bias the initial GA population
    - Surrogate fitness: XGBoost single train/val split (fast, ~0.1s per eval)
    - PSO uses V-shaped transfer function (|tanh(v)|) — better than sigmoid
      for binary feature selection (Mirjalili et al., 2016)
    - Fitness = F1_macro - λ × (n_selected / n_features)
    - Output: boolean mask of length n_features, applied as zero-masking
    - Runs ONCE per client at the Phase 1→2 transition (not every round)
"""

import time
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score


class GAPSOSelector:
    """
    GA+PSO hybrid feature selector.

    Args:
        n_features      : total number of input features (125)
        n_ga_gen        : number of GA generations
        pop_size        : GA population size (also PSO particle count)
        n_pso_iter      : number of PSO iterations
        lambda_penalty  : weight for feature-count penalty in fitness
        min_features    : minimum features to select (fallback to XGBoost top-K)
        random_state    : reproducibility seed
    """

    def __init__(
        self,
        n_features:     int   = 125,
        n_ga_gen:       int   = 10,
        pop_size:       int   = 8,
        n_pso_iter:     int   = 10,
        lambda_penalty: float = 0.01,
        min_features:   int   = 10,
        random_state:   int   = 42,
    ):
        self.n_features     = n_features
        self.n_ga_gen       = n_ga_gen
        self.pop_size       = pop_size
        self.n_pso_iter     = n_pso_iter
        self.lambda_penalty = lambda_penalty
        self.min_features   = min_features
        self.rng            = np.random.RandomState(random_state)

        self.best_mask_    = None
        self.best_fitness_ = None
        self.xgb_imp_      = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Run XGBoost → GA → PSO and return boolean feature mask.

        Args:
            X: (n_samples, n_features) local client training data
            y: (n_samples,) labels

        Returns:
            mask: boolean array of shape (n_features,)
        """
        t0 = time.time()
        n = self.n_features

        # Step 1: XGBoost importance → initialisation probabilities
        init_prob = self._xgb_importance(X, y)

        # Step 2: GA exploration
        ga_best, ga_fit = self._ga_phase(X, y, init_prob)
        n_ga = int(ga_best.sum())

        # Step 3: PSO exploitation starting from GA best
        pso_best, pso_fit = self._pso_phase(X, y, ga_best)
        n_pso = int(pso_best.sum())

        # Fallback: if too few features selected, use XGBoost top-K
        if n_pso < self.min_features:
            pso_best = self._xgb_topk(self.min_features)
            pso_fit  = self._fitness(pso_best, X, y)
            n_pso    = int(pso_best.sum())

        self.best_mask_    = pso_best.astype(bool)
        self.best_fitness_ = pso_fit
        elapsed = time.time() - t0

        print(f"      [GA+PSO] GA: {n_ga} feats (fit={ga_fit:.4f}) → "
              f"PSO: {n_pso} feats (fit={pso_fit:.4f})  [{elapsed:.1f}s]")

        return self.best_mask_

    def apply(self, X: np.ndarray) -> np.ndarray:
        """
        Apply zero-masking: unselected features set to 0.
        Input/output shape is identical (n_samples, n_features).
        Model architecture is unchanged.
        """
        assert self.best_mask_ is not None, "Call fit() first"
        X_masked = X.copy()
        X_masked[:, ~self.best_mask_] = 0.0
        return X_masked

    # ------------------------------------------------------------------
    # XGBoost initialisation
    # ------------------------------------------------------------------

    def _xgb_importance(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Train XGBoost and return per-feature selection probability
        biased toward high-importance features: prob ∈ [0.2, 0.8].
        """
        clf = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            use_label_encoder=False,
            eval_metric='mlogloss',
            verbosity=0,
            n_jobs=-1,
        )
        clf.fit(X, y)
        imp = clf.feature_importances_        # shape (n_features,)
        self.xgb_imp_ = imp
        imp_norm = (imp - imp.min()) / (imp.max() - imp.min() + 1e-10)
        return 0.2 + 0.6 * imp_norm           # maps to [0.2, 0.8]

    def _xgb_topk(self, k: int) -> np.ndarray:
        """Return binary mask keeping top-k features by XGBoost importance."""
        assert self.xgb_imp_ is not None
        mask = np.zeros(self.n_features, dtype=float)
        top_idx = np.argsort(self.xgb_imp_)[::-1][:k]
        mask[top_idx] = 1.0
        return mask

    # ------------------------------------------------------------------
    # Fitness function (surrogate: XGBoost single split)
    # ------------------------------------------------------------------

    def _fitness(self, mask: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
        """
        fitness = F1_macro(XGBoost, X_selected) - λ × (n_selected / n_features)

        XGBoost is the SURROGATE — cheap proxy for CNN+GRU fitness.
        Single 80/20 stratified split for speed (~0.1s per call).
        """
        n_sel = int(mask.sum())
        if n_sel < self.min_features:
            return 0.0

        X_sel = X[:, mask.astype(bool)]

        try:
            X_tr, X_val, y_tr, y_val = train_test_split(
                X_sel, y, test_size=0.2, stratify=y, random_state=42
            )
        except ValueError:
            # Stratify fails if some class has 1 sample
            X_tr, X_val, y_tr, y_val = train_test_split(
                X_sel, y, test_size=0.2, random_state=42
            )

        clf = xgb.XGBClassifier(
            n_estimators=50,
            max_depth=4,
            use_label_encoder=False,
            eval_metric='mlogloss',
            verbosity=0,
            n_jobs=-1,
        )
        clf.fit(X_tr, y_tr)
        preds  = clf.predict(X_val)
        f1     = f1_score(y_val, preds, average='macro', zero_division=0)
        density = n_sel / self.n_features

        return f1 - self.lambda_penalty * density

    # ------------------------------------------------------------------
    # GA phase
    # ------------------------------------------------------------------

    def _ga_phase(
        self,
        X: np.ndarray,
        y: np.ndarray,
        init_prob: np.ndarray,
    ) -> tuple:
        """
        Genetic Algorithm: tournament selection, uniform crossover,
        bit-flip mutation.  Population seeded from XGBoost importance.
        """
        n   = self.n_features
        pop = self.rng.binomial(1, init_prob, (self.pop_size, n)).astype(float)

        fitness = np.array([self._fitness(p, X, y) for p in pop])
        best_idx = np.argmax(fitness)
        best, best_fit = pop[best_idx].copy(), fitness[best_idx]

        for _ in range(self.n_ga_gen):
            new_pop = []

            # Tournament selection
            for _ in range(self.pop_size):
                i1, i2 = self.rng.randint(0, self.pop_size, 2)
                winner = pop[i1] if fitness[i1] >= fitness[i2] else pop[i2]
                new_pop.append(winner.copy())

            # Uniform crossover (p_cross=0.8)
            for i in range(0, self.pop_size - 1, 2):
                if self.rng.rand() < 0.8:
                    cx_mask = self.rng.randint(0, 2, n)
                    c1 = np.where(cx_mask, new_pop[i],   new_pop[i+1])
                    c2 = np.where(cx_mask, new_pop[i+1], new_pop[i])
                    new_pop[i], new_pop[i+1] = c1, c2

            # Bit-flip mutation (p_mut=0.02 per bit)
            for i in range(self.pop_size):
                mut = self.rng.rand(n) < 0.02
                new_pop[i] = np.where(mut, 1.0 - new_pop[i], new_pop[i])

            pop     = np.array(new_pop)
            fitness = np.array([self._fitness(p, X, y) for p in pop])

            idx = np.argmax(fitness)
            if fitness[idx] > best_fit:
                best, best_fit = pop[idx].copy(), fitness[idx]

        return best, best_fit

    # ------------------------------------------------------------------
    # PSO phase
    # ------------------------------------------------------------------

    def _pso_phase(
        self,
        X: np.ndarray,
        y: np.ndarray,
        ga_best: np.ndarray,
    ) -> tuple:
        """
        Binary PSO with V-shaped transfer function |tanh(v)|.
        Particles initialised near GA best solution.
        Inertia weight decays linearly 0.9 → 0.4.
        """
        n   = self.n_features
        c1  = c2 = 2.0
        w_max, w_min = 0.9, 0.4

        # Init particles: small random perturbations around GA best
        particles  = np.zeros((self.pop_size, n))
        velocities = self.rng.uniform(-1.0, 1.0, (self.pop_size, n))

        for i in range(self.pop_size):
            flip = self.rng.rand(n) < 0.1   # flip ~10% of bits from GA best
            particles[i] = np.where(flip, 1.0 - ga_best, ga_best)

        pbest     = particles.copy()
        pbest_fit = np.array([self._fitness(p, X, y) for p in pbest])

        gbest_idx = np.argmax(pbest_fit)
        gbest     = pbest[gbest_idx].copy()
        gbest_fit = pbest_fit[gbest_idx]

        for t in range(self.n_pso_iter):
            w = w_max - (w_max - w_min) * t / self.n_pso_iter

            for i in range(self.pop_size):
                r1 = self.rng.rand(n)
                r2 = self.rng.rand(n)

                velocities[i] = (
                    w  * velocities[i]
                    + c1 * r1 * (pbest[i] - particles[i])
                    + c2 * r2 * (gbest   - particles[i])
                )

                # V-shaped transfer: P(flip bit) = |tanh(v)|
                transfer = np.abs(np.tanh(velocities[i]))
                flip     = self.rng.rand(n) < transfer
                particles[i] = np.where(flip, 1.0 - particles[i], particles[i])

                fit = self._fitness(particles[i], X, y)
                if fit > pbest_fit[i]:
                    pbest[i]     = particles[i].copy()
                    pbest_fit[i] = fit
                    if fit > gbest_fit:
                        gbest     = particles[i].copy()
                        gbest_fit = fit

        return gbest, gbest_fit


# ----------------------------------------------------------------------
# Quick validation
# ----------------------------------------------------------------------

if __name__ == '__main__':
    np.random.seed(42)

    # Simulate a client: 800 samples, 125 features, 12 classes (imbalanced)
    n, d, c = 800, 125, 12
    X = np.random.randn(n, d).astype(np.float32)
    # Make first 30 features informative, rest noise
    y_base = (X[:, :30].mean(axis=1) > 0).astype(int)
    y = (y_base * 3 + np.random.randint(0, 4, n)) % c

    selector = GAPSOSelector(
        n_features     = d,
        n_ga_gen       = 5,
        pop_size       = 5,
        n_pso_iter     = 5,
        lambda_penalty = 0.01,
        min_features   = 10,
        random_state   = 42,
    )

    mask = selector.fit(X, y)

    assert mask.dtype == bool
    assert len(mask) == d
    assert mask.sum() >= 10, f"Too few features: {mask.sum()}"

    # Zero-masking preserves shape
    X_masked = selector.apply(X)
    assert X_masked.shape == X.shape
    assert (X_masked[:, ~mask] == 0).all(), "Unselected features should be zero"
    assert not (X_masked[:, mask] == 0).all(), "Selected features should not all be zero"

    print(f"\nga_pso_selector.py validation passed.")
    print(f"  Features selected : {mask.sum()} / {d} ({mask.sum()/d*100:.1f}%)")
    print(f"  Best fitness      : {selector.best_fitness_:.4f}")
