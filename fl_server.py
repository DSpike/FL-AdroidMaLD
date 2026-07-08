import numpy as np
import torch
import torch.nn as nn

from models.cnn_gru import CNNGRU, FedELEnsemble, get_initial_weights
from aoa_selector import AOAClientSelector


class FLServer:
    """
    Federated Learning server.

    Two-phase operation (FedEL-style):

    Phase 1 — Warmup (rounds 1 .. warmup_rounds):
        Standard FedAvg on full CNN+GRU weights.
        Goal: build a stable shared feature extractor G(·).

    Phase 2 — Ensemble (rounds warmup_rounds+1 .. total_rounds):
        At the transition point, server collects the final round's
        C(·) heads from all clients and builds a FedELEnsemble.
        Only the meta_FC layer is FedAvg'd each round.

    AOA client selection is added in Phase 2 (Phase 1 uses all clients).
    """

    def __init__(
        self,
        input_dim:     int,
        num_classes:   int = 12,
        warmup_rounds: int = 15,
        device:        torch.device = torch.device('cpu'),
    ):
        self.input_dim     = input_dim
        self.num_classes   = num_classes
        self.warmup_rounds = warmup_rounds
        self.device        = device
        self.round         = 0

        # Global CNN+GRU weights (used in warmup phase)
        self.global_weights = get_initial_weights(input_dim, num_classes=num_classes)

        # FedEL ensemble (built at warmup->ensemble transition)
        self.ensemble: FedELEnsemble | None = None

    # ------------------------------------------------------------------
    # Phase property
    # ------------------------------------------------------------------

    @property
    def phase(self) -> str:
        return 'warmup' if self.round < self.warmup_rounds else 'ensemble'

    # ------------------------------------------------------------------
    # Phase 1: FedAvg on full CNN+GRU
    # ------------------------------------------------------------------

    def aggregate_warmup(self, client_updates: list):
        """
        Weighted FedAvg on full CNN+GRU model weights.

        client_updates: list of dicts with keys 'weights' and 'n_train'.
        """
        total = sum(u['n_train'] for u in client_updates)

        aggregated = {}
        for key in self.global_weights:
            aggregated[key] = torch.zeros_like(self.global_weights[key],
                                               dtype=torch.float32)
            for u in client_updates:
                aggregated[key] += u['weights'][key].float() * (u['n_train'] / total)

        self.global_weights = aggregated
        self.round += 1

    # ------------------------------------------------------------------
    # Phase transition: build FedEL ensemble
    # ------------------------------------------------------------------

    def build_ensemble(self, client_updates: list) -> FedELEnsemble:
        """
        FedEL Step 2+3: collect C(·) classifier heads from all clients,
        attach them to the shared frozen G(·), add meta_FC.

        Called once at the warmup->ensemble transition.

        client_updates: list of dicts with key 'classifier_state'.
        Returns the built FedELEnsemble.
        """
        # Build shared extractor from current global weights
        extractor = CNNGRU(self.input_dim, num_classes=self.num_classes)
        extractor.load_state_dict(self.global_weights)

        classifier_states = [u['classifier_state'] for u in client_updates]
        self.ensemble = FedELEnsemble(
            extractor, classifier_states, self.num_classes
        ).to(self.device)

        print(f"\n[Server] FedEL ensemble built with {len(classifier_states)} heads.")
        print(f"         meta_FC input dim: {len(classifier_states) * self.num_classes}")
        return self.ensemble

    # ------------------------------------------------------------------
    # Phase 2: FedAvg on meta_FC only
    # ------------------------------------------------------------------

    def aggregate_ensemble(self, client_updates: list):
        """
        FedAvg on meta_FC weights only.

        client_updates: list of dicts with keys 'meta_state' and 'n_train'.
        """
        assert self.ensemble is not None, "Call build_ensemble() before aggregate_ensemble()"

        total = sum(u['n_train'] for u in client_updates)
        new_meta = {}
        ref_state = client_updates[0]['meta_state']

        for key in ref_state:
            new_meta[key] = torch.zeros_like(ref_state[key], dtype=torch.float32)
            for u in client_updates:
                new_meta[key] += u['meta_state'][key].float() * (u['n_train'] / total)

        self.ensemble.meta_fc.load_state_dict(new_meta)
        self.round += 1

    # ------------------------------------------------------------------
    # Distribution to clients
    # ------------------------------------------------------------------

    def get_global_weights(self) -> dict:
        """Return current CNN+GRU state dict for warmup-phase clients."""
        return {k: v.clone() for k, v in self.global_weights.items()}

    def get_ensemble_info(self) -> dict:
        """
        Package everything a client needs to reconstruct and fine-tune
        the FedELEnsemble locally.
        """
        assert self.ensemble is not None

        extractor_state     = {k: v.clone().cpu()
                               for k, v in self.ensemble.extractor.state_dict().items()}
        classifier_states   = [{k: v.clone().cpu()
                                for k, v in head.state_dict().items()}
                               for head in self.ensemble.heads]
        meta_state          = {k: v.clone().cpu()
                               for k, v in self.ensemble.meta_fc.state_dict().items()}

        return {
            'extractor_state':   extractor_state,
            'classifier_states': classifier_states,
            'meta_state':        meta_state,
            'input_dim':         self.input_dim,
            'num_classes':       self.num_classes,
        }

    # ------------------------------------------------------------------
    # Evaluation helpers
    # ------------------------------------------------------------------

    def get_eval_model(self) -> nn.Module:
        """
        Return the current best model for global holdout evaluation.
        Phase 1: CNNGRU with global weights.
        Phase 2: FedELEnsemble.
        """
        if self.ensemble is not None:
            return self.ensemble

        model = CNNGRU(self.input_dim, num_classes=self.num_classes)
        model.load_state_dict(self.global_weights)
        return model.to(self.device)

    # ------------------------------------------------------------------
    # AOA client selection
    # ------------------------------------------------------------------

    def select_clients(
        self,
        all_indices:       list,
        client_updates:    list,
        global_class_dist: np.ndarray,
        k_select:          int,
        random_state:      int = 42,
    ) -> list:
        """
        Phase 1 (warmup): return all client indices.
        Phase 2 (ensemble): AOA selects k_select clients by contribution score.

        Args:
            all_indices      : list of all available client indices
            client_updates   : last round's updates (divergence + class_dist)
            global_class_dist: fraction per class in global training set
            k_select         : number of clients to select
            random_state     : AOA seed

        Returns:
            selected: list of client indices for the next round
        """
        if self.phase == 'warmup' or client_updates is None:
            return all_indices

        n_total = len(all_indices)

        aoa = AOAClientSelector(
            n_clients    = n_total,
            k_select     = k_select,
            pop_size     = 10,
            max_iter     = 15,
            random_state = random_state,
        )

        # Reference weights: meta_fc state in ensemble phase, global CNN+GRU in warmup
        if self.ensemble is not None:
            ref_weights = {k: v.cpu() for k, v in self.ensemble.meta_fc.state_dict().items()}
        else:
            ref_weights = self.global_weights

        # compute_scores returns one score per entry in client_updates (may be a subset)
        partial_scores = aoa.compute_scores(client_updates, ref_weights, global_class_dist)
        mean_score     = float(partial_scores.mean()) if len(partial_scores) else 0.0

        # Map partial scores onto the full n_clients array; non-participants get mean
        full_scores = np.full(n_total, mean_score)
        for j, u in enumerate(client_updates):
            cid = u.get('client_id', -1) - 1   # 0-indexed
            if 0 <= cid < n_total:
                full_scores[cid] = partial_scores[j]

        selected_local = aoa.select(full_scores)
        selected       = [all_indices[i] for i in selected_local]

        print(f"   [AOA] Scores  : {np.round(full_scores, 3)}")
        print(f"   [AOA] Selected: clients {sorted([s+1 for s in selected])}")
        return selected


# ----------------------------------------------------------------------
# Quick validation
# ----------------------------------------------------------------------

if __name__ == '__main__':
    from models.cnn_gru import get_initial_weights
    torch.manual_seed(42)

    device = torch.device('cpu')
    d, c   = 80, 12

    server = FLServer(input_dim=d, num_classes=c, warmup_rounds=2, device=device)
    assert server.phase == 'warmup'

    # Simulate 3 client warmup updates
    def fake_warmup_update(n):
        weights = get_initial_weights(d, num_classes=c)
        # Perturb weights slightly
        weights = {k: v.float() + torch.randn_like(v.float()) * 0.01 for k, v in weights.items()}
        model   = CNNGRU(d, num_classes=c)
        model.load_state_dict(weights)
        return {
            'n_train':          n,
            'weights':          weights,
            'classifier_state': model.get_classifier_state(),
        }

    updates = [fake_warmup_update(n) for n in [400, 300, 200]]

    # Round 1 warmup
    server.aggregate_warmup(updates)
    assert server.round == 1
    assert server.phase == 'warmup'

    # Round 2 warmup (last warmup round since warmup_rounds=2)
    server.aggregate_warmup(updates)
    assert server.round == 2
    assert server.phase == 'ensemble'

    # Build ensemble
    ensemble = server.build_ensemble(updates)
    assert isinstance(ensemble, FedELEnsemble)

    # Eval model should now be ensemble
    eval_model = server.get_eval_model()
    assert isinstance(eval_model, FedELEnsemble)

    # Test forward pass through ensemble
    x = torch.randn(8, d)
    out = eval_model(x)
    assert out.shape == (8, c), f"Wrong output shape: {out.shape}"

    # Simulate ensemble phase update
    info = server.get_ensemble_info()
    assert 'extractor_state' in info
    assert len(info['classifier_states']) == 3

    # FedAvg meta_FC
    meta_updates = [
        {'n_train': n, 'meta_state': {k: v.clone() for k,v in ensemble.meta_fc.state_dict().items()}}
        for n in [400, 300, 200]
    ]
    server.aggregate_ensemble(meta_updates)
    assert server.round == 3

    print("fl_server.py validation passed.")
    print(f"  Warmup rounds : 2")
    print(f"  Ensemble heads: {len(ensemble.heads)}")
    print(f"  meta_FC params: {sum(p.numel() for p in ensemble.meta_fc.parameters())}")
