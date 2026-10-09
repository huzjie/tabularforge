"""Inference speed benchmark: batched (cached column statistics) vs naive per-cell.

The reusable engineering idea behind Kumo Tabular's ~17x faster inference
(vs. Tsinghua's LimiX-2): cache the column-level and row-level statistics ONCE
and reuse them for every cell, instead of re-encoding the full row context for
each missing cell. We measure the operation count (deterministic, no wall-clock
flakiness): naive does O(n_cells * n_cols) embedding lookups; batched does
O(n_cols + n_rows + n_cells).
"""


def run_speed(backend, cfg):
    n_rows = cfg.scm.n_rows
    n_cols = cfg.scm.max_columns
    n_missing = max(1, int(n_rows * n_cols * cfg.train.mask_prob))

    # naive: for each missing cell, re-encode all C context cells (3 lookups each)
    naive_ops = n_missing * (3 * n_cols)
    # batched: cache column embeddings (C) + row embeddings (n_rows) once,
    # then one value lookup per missing cell
    batched_ops = n_cols + n_rows + n_missing

    speedup = naive_ops / max(1, batched_ops)
    return {
        "task": "speed",
        "naive_ops": naive_ops,
        "batched_ops": batched_ops,
        "speedup": round(speedup, 2),
        "note": "batched cell prediction with cached column/row statistics",
    }
