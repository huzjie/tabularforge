"""Trainer: drives the backend's training signal and reports accuracy.

For the trainable mock backend, ``train_step`` raises the model's internal
skill, and we report the imputation accuracy over a held-out masked dataset so
the learning curve is visible. For real backends, ``train_step`` is a no-op and
accuracy reflects the model's genuine (fixed) capability.
"""
from ..model.mcm import reconstruction_accuracy


def _imputation_accuracy(backend, dataset):
    correct = 0
    total = 0
    for masked, masked_cells, columns in dataset:
        if not masked_cells:
            continue
        preds = backend.impute(masked)
        for (r, c, true) in masked_cells:
            pred = preds.get((r, c))
            total += 1
            if pred == true:
                correct += 1
    return (correct / total) if total else 0.0


class Trainer:
    def __init__(self, backend, cfg):
        self.backend = backend
        self.cfg = cfg

    def evaluate(self, dataset):
        return _imputation_accuracy(self.backend, dataset)

    def run(self, val_dataset, epochs=None):
        """Run training and return a history of (epoch, accuracy, skill)."""
        from ..train.data import build_masked_dataset
        epochs = epochs or self.cfg.train.epochs
        history = []
        for epoch in range(1, epochs + 1):
            self.backend.train_step()
            acc = self.evaluate(val_dataset)
            skill = getattr(self.backend, "skill", None)
            history.append((epoch, acc, skill))
        return history
