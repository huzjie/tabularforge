"""Generation benchmark: synthesize a table from a schema and check validity."""
from ..scm.synth import synthesize_table


def _valid(rows, columns):
    ok = 0
    total = 0
    for r in rows:
        for c, col in enumerate(columns):
            v = r[c]
            total += 1
            if col.dtype == "categorical" and col.domain:
                if v in col.domain:
                    ok += 1
            elif col.dtype == "numeric":
                try:
                    float(v)
                    ok += 1
                except (TypeError, ValueError):
                    pass
            else:
                ok += 1
    return ok / total if total else 0.0


def run_generation(backend, cfg, n_rows=None):
    n_rows = n_rows or min(cfg.scm.n_rows, 64)
    table, _specs, _parents = synthesize_table(
        seed=cfg.scm.seed, n_columns=cfg.scm.max_columns, n_rows=cfg.scm.n_rows)
    columns = table.columns
    rows = backend.generate(columns, n_rows)
    validity = _valid(rows, columns)
    return {"task": "generation", "validity": validity,
            "n_rows": n_rows, "n_cols": len(columns),
            "shape_ok": len(rows) == n_rows and all(len(r) == len(columns) for r in rows)}
