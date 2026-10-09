"""Imputation benchmark: fill masked cells, score against ground truth.

The ground truth is the actual SCM-generated value. For the trainable mock
backend this yields a clean accuracy curve (skill 0.5 -> 1.0); for the cpu
backend it measures the real (fixed) model's reconstruction accuracy.
"""
from ..scm.synth import synthesize_table


def run_imputation(backend, cfg, n_tables=1):
    total_correct = 0
    total = 0
    for t in range(n_tables):
        table, _specs, _parents = synthesize_table(
            seed=cfg.scm.seed,
            n_columns=cfg.scm.max_columns,
            n_rows=cfg.scm.n_rows,
        )
        masked, masked_cells = table.mask(mask_prob=cfg.train.mask_prob, seed=t)
        if not masked_cells:
            continue
        preds = backend.impute(masked)
        for (r, c, true) in masked_cells:
            total += 1
            if preds.get((r, c)) == true:
                total_correct += 1
    acc = (total_correct / total) if total else 0.0
    return {"task": "imputation", "accuracy": acc, "n_cells": total,
            "n_correct": total_correct}
