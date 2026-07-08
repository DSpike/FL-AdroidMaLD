import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score

from models.cnn_gru import CNNGRU, FedELEnsemble, build_criterion
from ga_pso_selector import GAPSOSelector


class FLClient:
    """
    Federated Learning client.

    Responsibilities:
        - Holds local data (fixed 80/20 internal split for validation reporting)
        - Receives global weights from server each round
        - Fine-tunes CNN+GRU locally and returns updated weights
        - In Phase 1 (warmup):  full model fine-tuned, full weights returned
        - In Phase 2 (ensemble): G(·) frozen, only C(·) fine-tuned,
                                  classifier_state returned for FedEL
    """

    def __init__(
        self,
        client_id: int,
        X: np.ndarray,
        y: np.ndarray,
        num_classes: int,
        device: torch.device,
        random_state: int = 42,
    ):
        self.client_id   = client_id
        self.num_classes = num_classes
        self.device      = device

        # Fixed 80/20 internal split — never changes across rounds.
        # Stratified by default; falls back to random split when a class has
        # only 1 sample (can happen with many clients + severe non-IID).
        try:
            self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
                X, y, test_size=0.2, stratify=y, random_state=random_state,
            )
        except ValueError:
            self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
                X, y, test_size=0.2, stratify=None, random_state=random_state,
            )

        self.n_train   = len(self.y_train)
        self.input_dim = self.X_train.shape[1]

        # Class-weighted criterion computed from local training labels
        self.criterion = build_criterion(self.y_train, num_classes, device)

        # Feature mask — computed once at Phase 1→2 transition via GA+PSO
        self.feature_mask_: np.ndarray | None = None
        self.selector_:     GAPSOSelector | None = None

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        global_weights: dict,
        n_epochs: int    = 5,
        lr: float        = 1e-3,
        batch_size: int  = 32,
        phase: str       = 'warmup',   # 'warmup' | 'ensemble'
    ) -> dict:
        """
        Run one local training round.

        Args:
            global_weights: CNN+GRU state dict received from server.
            n_epochs:       local epochs to run.
            lr:             Adam learning rate.
            batch_size:     mini-batch size.
            phase:          'warmup'  -> unfreeze all, return full state dict
                            'ensemble'-> freeze G(·), fine-tune C(·) only

        Returns dict with keys:
            client_id, n_train, weights, classifier_state, val_f1, val_acc
        """
        model = CNNGRU(
            input_dim  = self.input_dim,
            num_classes = self.num_classes,
        ).to(self.device)

        model.load_state_dict(global_weights)

        if phase == 'ensemble':
            model.freeze_extractor()
        else:
            model.unfreeze_all()

        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr,
        )

        # Apply GA+PSO zero-masking if feature mask has been computed (Phase 2 onwards)
        X_train_data = (self.selector_.apply(self.X_train)
                        if self.feature_mask_ is not None and self.selector_ is not None
                        else self.X_train)
        X_t = torch.FloatTensor(X_train_data).to(self.device)
        y_t = torch.LongTensor(self.y_train).to(self.device)
        loader = DataLoader(
            TensorDataset(X_t, y_t),
            batch_size=batch_size,
            shuffle=True,
        )

        model.train()
        train_loss = 0.0
        for epoch in range(n_epochs):
            epoch_loss, n_batches = 0.0, 0
            for xb, yb in loader:
                optimizer.zero_grad()
                loss = self.criterion(model(xb), yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches  += 1
            if epoch == n_epochs - 1:
                train_loss = epoch_loss / max(n_batches, 1)

        # Validation on internal held-out split
        val_f1, val_acc = self._validate(model)

        return {
            'client_id':        self.client_id,
            'n_train':          self.n_train,
            'weights':          {k: v.clone().cpu() for k, v in model.state_dict().items()},
            'classifier_state': model.get_classifier_state(),
            'class_dist':       self._class_distribution(),
            'val_f1':           val_f1,
            'val_acc':          val_acc,
            'train_loss':       train_loss,
            'phase':            phase,
        }

    # ------------------------------------------------------------------
    # Feature selection (runs ONCE at Phase 1→2 transition)
    # ------------------------------------------------------------------

    def compute_feature_mask(
        self,
        n_ga_gen:   int   = 10,
        pop_size:   int   = 8,
        n_pso_iter: int   = 10,
        random_state: int = 42,
    ) -> np.ndarray:
        """
        Run XGBoost → GA → PSO on local training data.
        Stores mask in self.feature_mask_ for all subsequent ensemble rounds.
        Returns boolean mask of shape (n_features,).
        """
        print(f"   Client {self.client_id}: running GA+PSO feature selection ...")
        self.selector_ = GAPSOSelector(
            n_features   = self.input_dim,
            n_ga_gen     = n_ga_gen,
            pop_size     = pop_size,
            n_pso_iter   = n_pso_iter,
            random_state = random_state,
        )
        self.feature_mask_ = self.selector_.fit(self.X_train, self.y_train)
        n_sel = int(self.feature_mask_.sum())
        print(f"   Client {self.client_id}: {n_sel}/{self.input_dim} features selected "
              f"({n_sel/self.input_dim*100:.1f}%)")
        return self.feature_mask_

    # ------------------------------------------------------------------
    # Phase 2: ensemble fine-tuning
    # ------------------------------------------------------------------

    def train_ensemble(
        self,
        ensemble_info: dict,
        n_epochs:   int   = 5,
        lr:         float = 1e-3,
        batch_size: int   = 32,
    ) -> dict:
        """
        Fine-tune meta_FC of the FedELEnsemble on local data.
        G(·) and all C_k(·) heads remain frozen.

        Args:
            ensemble_info: dict from FLServer.get_ensemble_info().

        Returns dict with keys:
            client_id, n_train, meta_state, val_f1, val_acc
        """
        extractor = CNNGRU(
            input_dim   = ensemble_info['input_dim'],
            num_classes = ensemble_info['num_classes'],
        )
        extractor.load_state_dict(ensemble_info['extractor_state'])

        ensemble = FedELEnsemble(
            extractor,
            ensemble_info['classifier_states'],
            ensemble_info['num_classes'],
        ).to(self.device)

        ensemble.meta_fc.load_state_dict(ensemble_info['meta_state'])

        optimizer = torch.optim.Adam(ensemble.meta_fc.parameters(), lr=lr)

        X_t = torch.FloatTensor(self.X_train).to(self.device)
        y_t = torch.LongTensor(self.y_train).to(self.device)
        loader = DataLoader(
            TensorDataset(X_t, y_t),
            batch_size=batch_size,
            shuffle=True,
        )

        ensemble.train()
        for _ in range(n_epochs):
            for xb, yb in loader:
                optimizer.zero_grad()
                loss = self.criterion(ensemble(xb), yb)
                loss.backward()
                optimizer.step()

        val_f1, val_acc = self._validate(ensemble)

        return {
            'client_id':  self.client_id,
            'n_train':    self.n_train,
            'meta_state': ensemble.get_meta_state(),
            'class_dist': self._class_distribution(),
            'val_f1':     val_f1,
            'val_acc':    val_acc,
        }

    # ------------------------------------------------------------------
    # Internal validation
    # ------------------------------------------------------------------

    def _class_distribution(self) -> np.ndarray:
        """Fraction of each class in local training set — sent to server for AOA scoring."""
        counts = np.bincount(self.y_train, minlength=self.num_classes).astype(float)
        total  = counts.sum()
        return counts / total if total > 0 else counts

    def _validate(self, model: nn.Module, X_val_override: np.ndarray = None) -> tuple:
        """Evaluate model on the internal validation split. Returns (f1_macro, acc)."""
        model.eval()
        X_v_np = X_val_override if X_val_override is not None else self.X_val
        X_v    = torch.FloatTensor(X_v_np).to(self.device)
        y_v    = self.y_val

        with torch.no_grad():
            logits = model(X_v)
            preds  = logits.argmax(dim=1).cpu().numpy()

        f1  = f1_score(y_v, preds, average='macro', zero_division=0)
        acc = accuracy_score(y_v, preds)
        return f1, acc

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"FLClient(id={self.client_id}, "
            f"train={self.n_train}, val={len(self.y_val)}, "
            f"input_dim={self.input_dim})"
        )


# ----------------------------------------------------------------------
# Quick validation
# ----------------------------------------------------------------------

if __name__ == '__main__':
    from models.cnn_gru import get_initial_weights

    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device('cpu')

    # Simulate a client with 500 samples, 80 features, 12 classes
    n, d, c = 500, 80, 12
    X_dummy = np.random.randn(n, d).astype(np.float32)
    # Imbalanced labels: class 0 dominates, class 11 rare
    y_dummy = np.array([0]*200 + [1]*100 + [2]*50 + list(range(3, 12))*
                       (250 // 9 + 1))[:n]
    y_dummy = np.clip(y_dummy, 0, c - 1)

    client = FLClient(client_id=1, X=X_dummy, y=y_dummy,
                      num_classes=c, device=device)
    print(client)

    weights = get_initial_weights(input_dim=d, num_classes=c)

    # Warmup round
    result = client.train(weights, n_epochs=2, phase='warmup')
    assert result['weights'] is not None
    assert 0.0 <= result['val_f1'] <= 1.0
    print(f"  Warmup  | val_f1={result['val_f1']:.4f} | val_acc={result['val_acc']:.4f}")

    # Ensemble round (freeze G, fine-tune C only)
    result2 = client.train(weights, n_epochs=2, phase='ensemble')
    assert result2['classifier_state'] is not None
    print(f"  Ensemble| val_f1={result2['val_f1']:.4f} | val_acc={result2['val_acc']:.4f}")

    print("fl_client.py validation passed.")
