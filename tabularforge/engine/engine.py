"""TabularEngine: the high-level, backend-agnostic API.

Wraps a backend and exposes the four core capabilities of a tabular foundation
model: impute (fill missing cells), classify, regress, and generate.
"""
from ..backends import get_backend
from ..tabular.table import Table


class TabularEngine:
    def __init__(self, cfg, backend=None):
        self.cfg = cfg
        self.backend = backend if backend is not None else get_backend(cfg.backend, cfg)

    # -- primary capability -------------------------------------------------
    def impute(self, table):
        """Fill missing cells, returning a filled Table and the predictions."""
        preds = self.backend.impute(table)
        return table.fill(preds), preds

    # -- convenience --------------------------------------------------------
    def impute_records(self, records, columns=None):
        """Impute a list of dict records (for the JSON API)."""
        table = Table.from_records(records, columns)
        filled, _ = self.impute(table)
        return filled.to_records()

    def classify(self, table, target_col):
        return self.backend.classify(table, target_col)

    def regress(self, table, target_col):
        return self.backend.regress(table, target_col)

    def generate(self, columns, n_rows):
        return self.backend.generate(columns, n_rows)

    def train_step(self):
        return self.backend.train_step()

    @property
    def skill(self):
        return self.backend.skill

    def health(self):
        return {
            "backend": self.backend.name,
            "skill": getattr(self.backend, "skill", None),
            "ok": True,
        }
