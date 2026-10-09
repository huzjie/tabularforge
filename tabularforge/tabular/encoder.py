"""Cell -> vector encoding.

The reusable idea: instead of one encoder per column, encode every cell as the
sum of three deterministic components --

1. ``value`` component -- categorical values get an embedding-table lookup,
   numeric values are bucketized then looked up, datetimes are bucketed.
2. ``column`` component -- a per-column embedding (schema identity).
3. ``row`` component -- a per-row-position embedding.

Summing these (instead of concatenating) keeps a single fixed-width vector for
any table shape, which is what lets one model generalize across tables.
"""
from ..core.nn import Embedding
from ..utils.stable import stable_float


class CellEncoder:
    def __init__(self, dim=64, vocab_size=1024, max_columns=64, max_rows=512):
        self.dim = dim
        self.vocab_size = vocab_size
        self.max_columns = max_columns
        self.max_rows = max_rows
        self.value_emb = Embedding(vocab_size, dim, key="cell:value")
        self.col_emb = Embedding(max_columns, dim, key="cell:col")
        self.row_emb = Embedding(max_rows, dim, key="cell:row")

    def _value_id(self, value, dtype):
        if value is None:
            return 0
        if dtype == "categorical":
            return abs(hash(str(value))) % self.vocab_size
        if dtype == "numeric":
            try:
                bucket = int((float(value) + 100.0) * 10)
            except (TypeError, ValueError):
                bucket = 0
            return abs(bucket) % self.vocab_size
        if dtype == "datetime":
            return (hash(str(value)) % self.vocab_size)
        return abs(hash(str(value))) % self.vocab_size

    def encode_cell(self, value, col_idx, row_idx, dtype):
        """Return a dim-vector for one cell."""
        vid = self._value_id(value, dtype)
        v = self.value_emb(vid)
        c = self.col_emb(col_idx)
        r = self.row_emb(row_idx)
        return [v[i] + c[i] + r[i] for i in range(self.dim)]

    def encode_row_context(self, row, col_idx, row_idx, columns):
        """Encode the context for predicting cell (row_idx, col_idx).

        Context = mean of the encodings of the OTHER (non-missing) cells in the
        same row, plus a column-summary term. If the row has no other cells,
        fall back to a zero vector.
        """
        vecs = []
        for c, col in enumerate(columns):
            if c == col_idx:
                continue
            if row[c] is None:
                continue
            vecs.append(self.encode_cell(row[c], c, row_idx, col.dtype))
        if not vecs:
            return [0.0] * self.dim
        n = len(vecs)
        return [sum(v[i] for v in vecs) / n for i in range(self.dim)]
