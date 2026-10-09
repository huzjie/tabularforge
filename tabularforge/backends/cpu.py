"""CPU backend: a real TabFM forward pass in pure Python.

No oracle -- this backend genuinely predicts missing cells from row context via
the encoder + MLP. Accuracy is whatever the (deterministic, tiny) model
achieves, which is honest: it is the zero-dependency reference implementation.
"""
from ..model.tabfm import TabFM
from .base import Backend, register_backend


@register_backend("cpu")
class CpuBackend(Backend):
    name = "cpu"

    def __init__(self, cfg):
        super().__init__(cfg)
        self.model = TabFM(
            dim=cfg.model.dim,
            mlp_hidden=cfg.model.mlp_hidden,
            n_layers=cfg.model.n_layers,
            vocab_size=cfg.model.vocab_size,
            max_columns=cfg.model.max_columns,
            max_rows=cfg.model.max_rows,
        )

    def predict_categorical(self, row, col_idx, row_idx, columns, domain):
        return self.model.predict_categorical(row, col_idx, row_idx, columns, domain)

    def predict_numeric(self, row, col_idx, row_idx, columns):
        return self.model.predict_numeric(row, col_idx, row_idx, columns)

    def classify(self, table, target_col):
        from ..model.heads import ClassificationHead
        return ClassificationHead(self.model).predict(table, target_col)

    def regress(self, table, target_col):
        from ..model.heads import RegressionHead
        return RegressionHead(self.model).predict(table, target_col)

    def generate(self, columns, n_rows):
        from ..model.heads import GenerationHead
        return GenerationHead(self.model).generate(columns, n_rows)

    def impute(self, table):
        preds = {}
        for r, c in table.missing_cells():
            col = table.columns[c]
            row = table.rows[r]
            if col.dtype == "categorical":
                preds[(r, c)] = self.model.predict_categorical(
                    row, c, r, table.columns, col.domain)
            else:
                preds[(r, c)] = round(self.model.predict_numeric(
                    row, c, r, table.columns), 2)
        return preds

    def train_step(self):
        return 0.0
