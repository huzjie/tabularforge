"""Regression benchmark: predict a numeric target column."""
from ..scm.synth import synthesize_table
from .base import rmse


def _pick_numeric(columns):
    for idx, col in enumerate(columns):
        if col.dtype == "numeric":
            return idx
    return None


def run_regression(backend, cfg):
    table, _specs, _parents = synthesize_table(
        seed=cfg.scm.seed, n_columns=cfg.scm.max_columns, n_rows=cfg.scm.n_rows)
    target = _pick_numeric(table.columns)
    if target is None:
        return {"task": "regression", "rmse": None, "note": "no numeric column"}
    truth = [table.rows[r][target] for r in range(table.n_rows)]
    preds = backend.regress(table, target)
    return {"task": "regression", "rmse": rmse(preds, truth),
            "target_col": table.columns[target].name, "n_rows": table.n_rows}
