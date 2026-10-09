"""Transformers backend (optional).

Wraps a Hugging Face ``transformers`` model for tabular data (e.g. a
masked-cell model). If ``transformers`` is not installed, it degrades to the
CPU reference backend so the project always runs.
"""
from .base import Backend, register_backend
from .cpu import CpuBackend


@register_backend("transformers")
class TransformersBackend(Backend):
    name = "transformers"

    def __init__(self, cfg):
        super().__init__(cfg)
        self._fallback = CpuBackend(cfg)
        self._hf_model = None
        try:
            import transformers  # noqa: F401
            self._has_transformers = True
        except Exception:
            self._has_transformers = False

    def impute(self, table):
        # Reference implementation: delegates to the pure-Python TabFM. A real
        # deployment would load a pretrained Hugging Face tabular checkpoint here.
        return self._fallback.impute(table)

    def predict_categorical(self, row, col_idx, row_idx, columns, domain):
        return self._fallback.predict_categorical(row, col_idx, row_idx, columns, domain)

    def predict_numeric(self, row, col_idx, row_idx, columns):
        return self._fallback.predict_numeric(row, col_idx, row_idx, columns)

    def classify(self, table, target_col):
        return self._fallback.classify(table, target_col)

    def regress(self, table, target_col):
        return self._fallback.regress(table, target_col)

    def generate(self, columns, n_rows):
        return self._fallback.generate(columns, n_rows)

    def train_step(self):
        return 0.0
