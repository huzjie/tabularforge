"""Deterministic trainable mock backend.

The mock is a stand-in for a real pretrained foundation model, with one key
property: it is DETERMINISTIC and TRAINABLE. Its single parameter ``skill``
controls how often it consults the ground-truth "oracle" (the structural causal
model that generated the table) versus guessing. ``train_step`` raises skill
monotonically, so imputation accuracy climbs from 0.5 to 1.0 -- a clean,
reproducible learning curve that requires no real data and no GPU.

Reusable trick (from prior projects): the training signal must be monotonic --
``skill`` only ever goes UP, never oscillates -- and the noise/guess path must
span the full [0, 1] range so the accuracy curve does not saturate early.
"""
from ..scm.synth import ScmSynthesizer
from ..utils.stable import stable_float
from .base import Backend, register_backend


@register_backend("mock")
class MockBackend(Backend):
    name = "mock"

    def __init__(self, cfg):
        super().__init__(cfg)
        self._skill = 0.5
        self._synth = ScmSynthesizer(seed=cfg.scm.seed, edge_prob=cfg.scm.edge_prob)

    def _oracle_rows(self, table):
        """Regenerate the SCM ground-truth table matching ``table``'s shape.

        Because the SCM is deterministic (same seed + shape), this reproduces
        the exact values the benchmark generated -- the model's "causal memory".
        """
        rows, _specs, _parents = self._synth.generate(
            n_columns=table.n_cols, n_rows=table.n_rows)
        return rows

    def _wrong(self, table, c, true):
        col = table.columns[c]
        if col.dtype == "categorical" and col.domain:
            # pick a different category
            for v in col.domain:
                if v != true:
                    return v
            return true
        try:
            return float(true) + 1.0
        except (TypeError, ValueError):
            return "?" + str(true)

    def _oracle_value(self, table, r, c):
        oracle = self._oracle_rows(table)
        return oracle[r][c]

    def classify(self, table, target_col):
        """Predict a categorical target column (oracle vs guess by skill)."""
        oracle = self._oracle_rows(table)
        out = []
        for r in range(table.n_rows):
            true = oracle[r][target_col]
            u = stable_float(f"mock:cls:{table.name}:{r}:{target_col}")
            out.append(true if u < self._skill else self._wrong(table, target_col, true))
        return out

    def regress(self, table, target_col):
        """Predict a numeric target column (oracle vs guess by skill)."""
        oracle = self._oracle_rows(table)
        out = []
        for r in range(table.n_rows):
            true = oracle[r][target_col]
            u = stable_float(f"mock:reg:{table.name}:{r}:{target_col}")
            try:
                out.append(float(true) if u < self._skill else float(true) + 1.0)
            except (TypeError, ValueError):
                out.append(0.0)
        return out

    def generate(self, columns, n_rows):
        """Generate a valid table matching the requested schema."""
        from ..utils.stable import stable_choice
        rows = []
        for r in range(n_rows):
            row = []
            for c, col in enumerate(columns):
                if col.dtype == "categorical" and col.domain:
                    row.append(stable_choice(f"mock:gen:{r}:{c}", col.domain))
                elif col.dtype == "numeric":
                    row.append(round(stable_float(f"mock:gen:{r}:{c}", -5.0, 5.0), 2))
                elif col.dtype == "datetime":
                    row.append(f"2026-01-{(r % 28) + 1:02d}")
                else:
                    row.append(f"gen-{r}-{c}")
            rows.append(row)
        return rows

    def impute(self, table):
        oracle = self._oracle_rows(table)
        preds = {}
        for r, c in table.missing_cells():
            true = oracle[r][c]
            u = stable_float(f"mock:{table.name}:{r}:{c}")
            if u < self._skill:
                preds[(r, c)] = true
            else:
                preds[(r, c)] = self._wrong(table, c, true)
        return preds

    def train_step(self):
        """Monotonically increase skill (never decrease)."""
        before = self._skill
        self._skill = min(1.0, self._skill + 0.05)
        return self._skill

    @property
    def skill(self):
        return self._skill
