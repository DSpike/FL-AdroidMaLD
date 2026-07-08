import os
import csv
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix
)
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from models.cnn_gru import CNNGRU, FedELEnsemble


class GlobalEvaluator:
    """
    Evaluates the global model on the locked holdout set.

    The holdout set is split once before any FL training and never
    touched during training — it is the single source of truth for
    all reported results in the paper.
    """

    def __init__(
        self,
        X_test:      np.ndarray,
        y_test:      np.ndarray,
        class_names: list,
        results_dir: str = 'results',
        experiment:  str = 'fl_run',
    ):
        self.X_test      = X_test
        self.y_test      = y_test
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.results_dir = results_dir
        self.experiment  = experiment

        os.makedirs(results_dir, exist_ok=True)

        self._log_path = os.path.join(results_dir, f'{experiment}_rounds.csv')
        self._history  = []   # list of round metric dicts

        # Write CSV header
        with open(self._log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(
                ['round', 'phase', 'accuracy', 'f1_macro'] +
                [f'f1_{c}'        for c in class_names] +
                [f'precision_{c}' for c in class_names] +
                [f'recall_{c}'    for c in class_names]
            )

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        model:     nn.Module,
        device:    torch.device,
        round_num: int  = 0,
        phase:     str  = 'warmup',
    ) -> dict:
        """
        Evaluate model on the global holdout set.

        Args:
            model:     trained CNNGRU or FedELEnsemble instance.
            device:    torch device.
            round_num: current FL round (for logging).
            phase:     'warmup' | 'ensemble' (for logging).

        Returns:
            dict with accuracy, f1_macro, per_class_f1, per_class_precision,
            per_class_recall, confusion_matrix.
        """
        model.eval()
        X_t = torch.FloatTensor(self.X_test).to(device)

        with torch.no_grad():
            logits = model(X_t)
            preds  = logits.argmax(dim=1).cpu().numpy()

        labels   = np.arange(self.num_classes)
        acc      = accuracy_score(self.y_test, preds)
        f1_macro = f1_score(self.y_test, preds, average='macro', zero_division=0)
        f1_per   = f1_score(self.y_test, preds, average=None,    labels=labels, zero_division=0)
        prec_per = precision_score(self.y_test, preds, average=None, labels=labels, zero_division=0)
        rec_per  = recall_score(self.y_test, preds, average=None,    labels=labels, zero_division=0)
        cm       = confusion_matrix(self.y_test, preds, labels=labels)

        metrics = {
            'round':              round_num,
            'phase':              phase,
            'accuracy':           acc,
            'f1_macro':           f1_macro,
            'per_class_f1':       dict(zip(self.class_names, f1_per)),
            'per_class_precision': dict(zip(self.class_names, prec_per)),
            'per_class_recall':    dict(zip(self.class_names, rec_per)),
            'confusion_matrix':   cm,
        }

        self._log(metrics)
        self._print(metrics)
        return metrics

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------

    def _log(self, metrics: dict):
        """Append metrics to in-memory history and CSV."""
        self._history.append(metrics)
        with open(self._log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            row = (
                [metrics['round'], metrics['phase'],
                 f"{metrics['accuracy']:.6f}", f"{metrics['f1_macro']:.6f}"] +
                [f"{metrics['per_class_f1'][c]:.6f}"        for c in self.class_names] +
                [f"{metrics['per_class_precision'][c]:.6f}" for c in self.class_names] +
                [f"{metrics['per_class_recall'][c]:.6f}"    for c in self.class_names]
            )
            writer.writerow(row)

    def _print(self, metrics: dict):
        print(f"\n[Round {metrics['round']:>3d} | {metrics['phase']:>8s}] "
              f"Acc={metrics['accuracy']*100:.2f}%  "
              f"F1-macro={metrics['f1_macro']*100:.2f}%")
        for cls in self.class_names:
            f1   = metrics['per_class_f1'][cls]
            prec = metrics['per_class_precision'][cls]
            rec  = metrics['per_class_recall'][cls]
            flag = ' <-- minority' if f1 < 0.3 else ''
            print(f"    {cls:<20s}: F1={f1*100:.1f}%  P={prec*100:.1f}%  R={rec*100:.1f}%{flag}")

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plot_convergence(self):
        """Save F1-macro convergence curve across all logged rounds."""
        if len(self._history) < 2:
            return

        rounds  = [m['round']    for m in self._history]
        f1s     = [m['f1_macro'] for m in self._history]
        accs    = [m['accuracy'] for m in self._history]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(rounds, f1s,  marker='o', label='F1-macro (global holdout)')
        ax.plot(rounds, accs, marker='s', linestyle='--', label='Accuracy')
        ax.set_xlabel('Communication Round')
        ax.set_ylabel('Score')
        ax.set_title('FL Convergence — Global Holdout')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        path = os.path.join(self.results_dir, f'{self.experiment}_convergence.png')
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"Convergence plot saved: {path}")

    def plot_confusion_matrix(self, round_num: int = -1):
        """Save confusion matrix for a specific round (-1 = last logged round)."""
        if not self._history:
            return

        metrics = self._history[round_num]
        cm      = metrics['confusion_matrix']

        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(cm, cmap='Blues')
        plt.colorbar(im, ax=ax)

        ax.set_xticks(range(self.num_classes))
        ax.set_yticks(range(self.num_classes))
        ax.set_xticklabels(self.class_names, rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels(self.class_names, fontsize=8)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title(f'Confusion Matrix — Round {metrics["round"]}')

        thresh = cm.max() / 2
        for i in range(self.num_classes):
            for j in range(self.num_classes):
                ax.text(j, i, str(cm[i, j]),
                        ha='center', va='center', fontsize=6,
                        color='white' if cm[i, j] > thresh else 'black')

        plt.tight_layout()
        path = os.path.join(
            self.results_dir,
            f'{self.experiment}_cm_round{metrics["round"]}.png'
        )
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"Confusion matrix saved: {path}")

    def plot_per_class_metrics(self, round_num: int = -1):
        """Bar chart of per-class Precision, Recall, F1 for a given round."""
        if not self._history:
            return

        metrics = self._history[round_num]
        classes = self.class_names
        x       = np.arange(len(classes))
        width   = 0.25

        prec = [metrics['per_class_precision'][c] for c in classes]
        rec  = [metrics['per_class_recall'][c]    for c in classes]
        f1s  = [metrics['per_class_f1'][c]        for c in classes]

        fig, ax = plt.subplots(figsize=(max(10, len(classes) * 0.9), 5))
        ax.bar(x - width, prec, width, label='Precision')
        ax.bar(x,         rec,  width, label='Recall')
        ax.bar(x + width, f1s,  width, label='F1')
        ax.set_xticks(x)
        ax.set_xticklabels(classes, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('Score')
        ax.set_ylim(0, 1.05)
        ax.set_title(f'Per-Class Metrics — Round {metrics["round"]}')
        ax.legend()
        ax.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()

        path = os.path.join(
            self.results_dir,
            f'{self.experiment}_perclass_round{metrics["round"]}.png'
        )
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"Per-class metrics plot saved: {path}")

    @property
    def best_f1(self) -> float:
        if not self._history:
            return 0.0
        return max(m['f1_macro'] for m in self._history)

    @property
    def history(self) -> list:
        return self._history


# ----------------------------------------------------------------------
# Quick validation
# ----------------------------------------------------------------------

if __name__ == '__main__':
    from models.cnn_gru import CNNGRU, get_initial_weights

    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device('cpu')

    n, d, c = 400, 80, 12
    X_test  = np.random.randn(n, d).astype(np.float32)
    y_test  = np.random.randint(0, c, size=n)
    names   = [f'Class_{i}' for i in range(c)]

    evaluator = GlobalEvaluator(
        X_test, y_test, names,
        results_dir='results/test_eval',
        experiment='test'
    )

    # Random model should produce ~1/12 ≈ 8.3% accuracy
    model = CNNGRU(input_dim=d, num_classes=c)
    metrics = evaluator.evaluate(model, device, round_num=0, phase='warmup')

    assert 'accuracy'     in metrics
    assert 'f1_macro'     in metrics
    assert 'per_class_f1' in metrics
    assert metrics['accuracy'] < 0.25, "Random model should not exceed 25% accuracy"

    evaluator.plot_convergence()
    print("\nfl_evaluator.py validation passed.")
