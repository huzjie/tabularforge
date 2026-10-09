"""Scaling benchmark: imputation throughput as the table grows."""
from ..scm.synth import synthesize_table


def run_scaling(backend, cfg, sizes=(8, 16, 32, 64, 128)):
    curve = []
    for n_rows in sizes:
        table, _specs, _parents = synthesize_table(
            seed=cfg.scm.seed, n_columns=cfg.scm.max_columns, n_rows=n_rows)
        masked, cells = table.mask(mask_prob=cfg.train.mask_prob, seed=0)
        preds = backend.impute(masked)
        correct = sum(1 for (r, c, true) in cells if preds.get((r, c)) == true)
        acc = correct / len(cells) if cells else 1.0
        curve.append({"n_rows": n_rows, "accuracy": acc, "n_cells": len(cells)})
    return {"task": "scaling", "curve": curve}
