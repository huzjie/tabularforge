"""Classification benchmark: predict a categorical target column."""
from ..scm.synth import synthesize_table
from .base import accuracy


def _pick_categorical(columns):
    for idx, col in enumerate(columns):
        if col.dtype == "categorical":
            return idx
    return None


def run_classification(backend, cfg):
    table, _specs, _parents = synthesize_table(
        seed=cfg.scm.seed, n_columns=cfg.scm.max_columns, n_rows=cfg.scm.n_rows)
    target = _pick_categorical(table.columns)
    if target is None:
        return {"task": "classification", "accuracy": None, "note": "no categorical column"}
    truth = [table.rows[r][target] for r in range(table.n_rows)]
    preds = backend.classify(table, target)
    return {"task": "classification", "accuracy": accuracy(preds, truth),
            "target_col": table.columns[target].name, "n_rows": table.n_rows}
