"""Robustness benchmark: imputation accuracy under varying missing rates."""
from ..scm.synth import synthesize_table


def run_robustness(backend, cfg, rates=(0.05, 0.1, 0.2, 0.3, 0.5)):
    table, _specs, _parents = synthesize_table(
        seed=cfg.scm.seed, n_columns=cfg.scm.max_columns, n_rows=cfg.scm.n_rows)
    curve = []
    for rate in rates:
        masked, cells = table.mask(mask_prob=rate, seed=int(rate * 100))
        preds = backend.impute(masked)
        correct = sum(1 for (r, c, true) in cells if preds.get((r, c)) == true)
        acc = correct / len(cells) if cells else 1.0
        curve.append({"missing_rate": rate, "accuracy": acc, "n_cells": len(cells)})
    return {"task": "robustness", "curve": curve}
