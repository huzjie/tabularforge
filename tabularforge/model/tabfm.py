"""TabFM: a tabular foundation model.

Implements masked cell modeling (MCM) -- the "BERT for tables" idea. Given a
table with some cells missing, predict each missing cell from its row context
(the other cells in the row) plus its column schema. The model is fully
pure-Python and deterministic (hash-based weights), so it runs anywhere.
"""
from ..core.nn import MLP, LayerNorm
from ..tabular.encoder import CellEncoder


class TabFM:
    """A small tabular foundation model.

    Parameters are intentionally tiny so the reference runs instantly; the
    architecture (encode -> aggregate row context -> MLP -> predict) mirrors a
    real tabular foundation model.
    """

    def __init__(self, dim=64, mlp_hidden=128, n_layers=2, vocab_size=1024,
                 max_columns=64, max_rows=512):
        self.dim = dim
        self.vocab_size = vocab_size
        self.encoder = CellEncoder(dim=dim, vocab_size=vocab_size,
                                   max_columns=max_columns, max_rows=max_rows)
        self.norm = LayerNorm(dim, key="tabfm:norm")
        dims = [dim] + [mlp_hidden] * n_layers + [dim]
        self.mlp = MLP(dims, key="tabfm:mlp")

    def encode_context(self, row, col_idx, row_idx, columns):
        ctx = self.encoder.encode_row_context(row, col_idx, row_idx, columns)
        ctx = self.norm(ctx)
        return self.mlp(ctx)

    def predict_cell(self, row, col_idx, row_idx, columns):
        """Return a score vector over the vocabulary (dim-sized).

        For categorical columns, the caller projects scores onto the column
        domain; for numeric columns the caller converts the max-activation
        bucket back to a scalar.
        """
        return self.encode_context(row, col_idx, row_idx, columns)

    def predict_numeric(self, row, col_idx, row_idx, columns):
        """Predict a numeric cell as a scalar (sum of context + bias)."""
        ctx = self.predict_cell(row, col_idx, row_idx, columns)
        return sum(ctx) / self.dim

    def predict_categorical(self, row, col_idx, row_idx, columns, domain):
        """Predict a categorical cell: pick the domain value with max score."""
        ctx = self.predict_cell(row, col_idx, row_idx, columns)
        best = domain[0]
        best_score = None
        for val in domain:
            # deterministic score for each candidate value
            score = self._value_score(ctx, val)
            if best_score is None or score > best_score:
                best_score = score
                best = val
        return best

    def _value_score(self, ctx, value):
        # project context onto the value's hashed id for a scalar score
        vid = self.encoder._value_id(value, "categorical")
        v = self.encoder.value_emb(vid)
        return sum(ctx[i] * v[i] for i in range(self.dim))
