"""Core tabular data structures.

A ``Table`` is a list of ``Column`` schemas plus a list of rows (each row is a
list of cell values). Missing cells are represented by the ``MISSING`` sentinel
(or ``None``). The framework is deliberately schema-first: knowing each
column's dtype and domain is what lets a foundation model generalize across
tables of different shapes.
"""

MISSING = object()  # sentinel for a masked / missing cell


class Column:
    def __init__(self, name, dtype="numeric", domain=None):
        self.name = name
        self.dtype = dtype  # numeric | categorical | datetime | text
        self.domain = list(domain) if domain else []

    def to_dict(self):
        return {"name": self.name, "dtype": self.dtype, "domain": list(self.domain)}


class Table:
    def __init__(self, columns, rows=None, name="table"):
        self.columns = [c if isinstance(c, Column) else Column(**c) for c in columns]
        self.rows = [list(r) for r in (rows or [])]
        self.name = name

    @property
    def n_rows(self):
        return len(self.rows)

    @property
    def n_cols(self):
        return len(self.columns)

    def column(self, idx):
        """Return the list of values in column ``idx`` (MISSING preserved)."""
        return [r[idx] for r in self.rows]

    def is_missing(self, row_idx, col_idx):
        v = self.rows[row_idx][col_idx]
        return v is None or v is MISSING

    def missing_cells(self):
        """Return the list of (row_idx, col_idx) of missing cells."""
        out = []
        for r in range(self.n_rows):
            for c in range(self.n_cols):
                if self.is_missing(r, c):
                    out.append((r, c))
        return out

    def mask(self, mask_prob=0.15, seed=0):
        """Return a copy of this table with a random fraction of cells masked.

        Also returns the list of (row, col, true_value) for the masked cells so
        callers can score reconstruction.
        """
        from ..utils.stable import stable_bool
        rows = [list(r) for r in self.rows]
        masked = []
        for r in range(self.n_rows):
            for c in range(self.n_cols):
                if stable_bool(f"mask:{seed}:{r}:{c}", mask_prob):
                    masked.append((r, c, rows[r][c]))
                    rows[r][c] = MISSING
        return Table(self.columns, rows, self.name + "-masked"), masked

    def fill(self, predictions):
        """Return a copy with the given {(row, col): value} filled in."""
        rows = [list(r) for r in self.rows]
        for (r, c), v in predictions.items():
            rows[r][c] = v
        return Table(self.columns, rows, self.name + "-filled")

    def to_records(self):
        """Return a list of dicts (for JSON serialization / HTTP API)."""
        return [{col.name: (None if self.is_missing(r, c) else self.rows[r][c])
                 for c, col in enumerate(self.columns)}
                for r in range(self.n_rows)]

    @classmethod
    def from_records(cls, records, columns=None, name="table"):
        """Build a table from a list of dicts (JSON API input)."""
        if columns is None:
            keys = []
            for rec in records:
                for k in rec:
                    if k not in keys:
                        keys.append(k)
            columns = [Column(k, "text") for k in keys]
        rows = []
        for rec in records:
            row = []
            for col in columns:
                v = rec.get(col.name)
                row.append(MISSING if v is None else v)
            rows.append(row)
        return cls(columns, rows, name)
