"""Task heads built on top of TabFM.

The foundation model learns a single cell-context representation; each
downstream task (imputation, classification, regression, generation) is just a
different readout of that representation -- no new model needed.
"""
from .tabfm import TabFM


class ImputationHead:
    """Fill missing cells (zero-shot)."""

    def __init__(self, model):
        self.model = model

    def impute(self, table):
        predictions = {}
        for r, c in table.missing_cells():
            col = table.columns[c]
            row = table.rows[r]
            if col.dtype == "categorical":
                predictions[(r, c)] = self.model.predict_categorical(
                    row, c, r, table.columns, col.domain)
            else:
                predictions[(r, c)] = round(self.model.predict_numeric(
                    row, c, r, table.columns), 2)
        return predictions


class ClassificationHead:
    """Predict a categorical target column from the other columns."""

    def __init__(self, model):
        self.model = model

    def predict(self, table, target_col):
        out = []
        for r in range(table.n_rows):
            row = list(table.rows[r])
            row[target_col] = None  # mask the target
            col = table.columns[target_col]
            out.append(self.model.predict_categorical(
                row, target_col, r, table.columns, col.domain))
        return out


class RegressionHead:
    """Predict a numeric target column from the other columns."""

    def __init__(self, model):
        self.model = model

    def predict(self, table, target_col):
        out = []
        for r in range(table.n_rows):
            row = list(table.rows[r])
            row[target_col] = None
            out.append(self.model.predict_numeric(
                row, target_col, r, table.columns))
        return out


class GenerationHead:
    """Generate a whole table from a schema (columns only)."""

    def __init__(self, model):
        self.model = model

    def generate(self, columns, n_rows):
        """Generate ``n_rows`` rows column by column (parents-first).

        We generate columns in index order and, for each new cell, use the
        already-generated cells in the row as context -- the same masked-cell
        readout applied autoregressively.
        """
        rows = [[None] * len(columns) for _ in range(n_rows)]
        for r in range(n_rows):
            for c, col in enumerate(columns):
                if col.dtype == "categorical":
                    rows[r][c] = self.model.predict_categorical(
                        rows[r], c, r, columns, col.domain)
                else:
                    rows[r][c] = round(self.model.predict_numeric(
                        rows[r], c, r, columns), 2)
        return rows
