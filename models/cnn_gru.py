import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.utils.class_weight import compute_class_weight


class CNNGRU(nn.Module):
    """
    CNN+GRU classifier for tabular malware features.

    Proposed architecture for dynamic Android malware features:
        - Conv1D layers capture local correlations between API-call statistics
        - GRU captures sequential dependencies across the feature sequence
        - Split into G(·) / C(·) following the FedEL paradigm (Wu et al., ESWA 2024)

        G(·)  feature extractor: Conv layers + GRU + fc_extract  -> 64-dim vector
        C(·)  classifier head : Linear(64 -> num_classes)

    In Phase 1 (warm-up):  full model weights are FedAvg'd each round.
    In Phase 2 (FL):       AOA selects clients; G(·) and C(·) both updated via FedAvg.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64, num_classes: int = 12):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        # --- G(·): feature extractor ---
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm1d(64)
        self.pool  = nn.MaxPool1d(2)
        # GRU input_size=64 matches conv2 output channels after permute
        self.gru   = nn.GRU(
            input_size=64,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
        )
        self.fc_extract = nn.Linear(hidden_dim, 64)
        self.dropout    = nn.Dropout(0.3)

        # --- C(·): classifier head ---
        self.classifier = nn.Linear(64, num_classes)

    # ------------------------------------------------------------------
    # Forward passes
    # ------------------------------------------------------------------

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """G(·): maps (B, n_features) -> (B, 64) feature vector."""
        x = x.unsqueeze(1)                               # (B, 1, n_feat)
        x = F.relu(self.bn1(self.conv1(x)))              # (B, 32, n_feat)
        x = self.pool(F.relu(self.bn2(self.conv2(x))))   # (B, 64, n_feat//2)
        x = x.permute(0, 2, 1)                           # (B, seq, 64)
        _, hn = self.gru(x)                              # hn: (2, B, hidden)
        h = self.dropout(F.relu(self.fc_extract(hn[-1])))  # (B, 64)
        return h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full forward pass: (B, n_features) -> (B, num_classes) logits."""
        return self.classifier(self.forward_features(x))

    # ------------------------------------------------------------------
    # FedEL helpers
    # ------------------------------------------------------------------

    def get_classifier_state(self) -> dict:
        """Return C(·) state dict for FedEL ensemble construction on server."""
        return {k: v.clone() for k, v in self.classifier.state_dict().items()}

    def freeze_extractor(self):
        """Freeze G(·) parameters — used in FedEL Phase 2 fine-tuning."""
        for name, param in self.named_parameters():
            if 'classifier' not in name:
                param.requires_grad = False

    def unfreeze_all(self):
        """Unfreeze all parameters — used at start of each warm-up round."""
        for param in self.parameters():
            param.requires_grad = True


# ----------------------------------------------------------------------
# Ensemble meta-model (FedEL Step 3)
# ----------------------------------------------------------------------

class FedELEnsemble(nn.Module):
    """
    Global ensemble model built by the server after warm-up phase.

    Architecture:
        Shared G(·) [frozen]
        -> N parallel C_k(·) heads (one per participating client)
        -> concat: (B, N * num_classes)
        -> meta MLP: Linear(N*C -> meta_hidden) -> ReLU -> Dropout -> Linear(meta_hidden -> C)

    Only the meta MLP is updated in Phase 2 FL rounds.
    meta_hidden_dim=64 gives the meta-learner enough capacity to learn
    non-linear combinations of client head outputs.
    """

    def __init__(
        self,
        extractor: CNNGRU,
        classifier_states: list,   # list of C(·) state dicts from N clients
        num_classes: int = 12,
        meta_hidden_dim: int = 64,
    ):
        super().__init__()
        n_heads = len(classifier_states)
        feat_dim = extractor.classifier.in_features  # 64

        # Freeze and store shared extractor
        self.extractor = extractor
        for param in self.extractor.parameters():
            param.requires_grad = False

        # Build N frozen classifier heads
        self.heads = nn.ModuleList()
        for state in classifier_states:
            head = nn.Linear(feat_dim, num_classes)
            head.load_state_dict(state)
            for param in head.parameters():
                param.requires_grad = False
            self.heads.append(head)

        # Trainable meta MLP — the only layer FedAvg'd in Phase 2
        # Two-layer design: concat(N heads) -> hidden -> classes
        self.meta_fc = nn.Sequential(
            nn.Linear(n_heads * num_classes, meta_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(meta_hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.extractor.forward_features(x)           # (B, 64)
        head_outputs = [head(h) for head in self.heads]  # N × (B, C)
        combined = torch.cat(head_outputs, dim=1)         # (B, N*C)
        return self.meta_fc(combined)                     # (B, C)

    def get_meta_state(self) -> dict:
        """Return meta FC state dict for FedAvg on server."""
        return {k: v.clone() for k, v in self.meta_fc.state_dict().items()}


# ----------------------------------------------------------------------
# Utility functions
# ----------------------------------------------------------------------

def get_initial_weights(
    input_dim: int,
    hidden_dim: int = 64,
    num_classes: int = 12,
) -> dict:
    """Create a fresh CNNGRU and return its state dict for server initialisation."""
    model = CNNGRU(input_dim, hidden_dim, num_classes)
    return {k: v.clone() for k, v in model.state_dict().items()}


def compute_class_weights(y: np.ndarray, num_classes: int) -> torch.Tensor:
    """
    Balanced class weights for CrossEntropyLoss.
    weight_c = n_samples / (num_classes * n_samples_c)
    Handles severe imbalance: FileInfector (119) vs Riskware (6792).
    """
    # Only compute weights for classes present in y; assign weight=1 to absent
    # classes (can occur with many clients + severe non-IID where a client's
    # train split loses a rare class entirely after the 80/20 split).
    classes_present = np.unique(y)
    all_classes     = np.arange(num_classes)
    weights         = np.ones(num_classes, dtype=float)
    if len(classes_present) > 1:
        w = compute_class_weight('balanced', classes=classes_present, y=y)
        for cls, wt in zip(classes_present, w):
            weights[cls] = wt
    return torch.FloatTensor(weights)


def build_criterion(
    y: np.ndarray,
    num_classes: int,
    device: torch.device,
) -> nn.CrossEntropyLoss:
    """Weighted CrossEntropy loss built from global training label distribution."""
    weights = compute_class_weights(y, num_classes).to(device)
    return nn.CrossEntropyLoss(weight=weights)


# ----------------------------------------------------------------------
# Quick validation
# ----------------------------------------------------------------------

if __name__ == '__main__':
    torch.manual_seed(42)
    B, D, C = 16, 80, 12            # batch, features, classes

    model = CNNGRU(input_dim=D, num_classes=C)
    x = torch.randn(B, D)
    logits = model(x)
    assert logits.shape == (B, C), f"Expected ({B},{C}), got {logits.shape}"

    feats = model.forward_features(x)
    assert feats.shape == (B, 64), f"Feature shape wrong: {feats.shape}"

    # FedEL ensemble with 3 client heads
    states = [model.get_classifier_state() for _ in range(3)]
    ensemble = FedELEnsemble(model, states, num_classes=C)
    out = ensemble(x)
    assert out.shape == (B, C), f"Ensemble output shape wrong: {out.shape}"

    # Class weights
    y_dummy = np.array([0]*6792 + [11]*119 + list(range(1, 11))*200)
    w = compute_class_weights(y_dummy, C)
    assert len(w) == C

    print("cnn_gru.py validation passed.")
    print(f"  CNNGRU params : {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Ensemble heads: 3  meta_fc params: {sum(p.numel() for p in ensemble.meta_fc.parameters())}")
