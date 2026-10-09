"""Dataset builders: masked-cell training data."""
from ..scm.synth import synthesize_table


def build_masked_dataset(cfg, n_tables=8):
    """Build ``n_tables`` synthetic tables and mask a fraction of their cells.

    Returns a list of ``(masked_table, masked_cells, columns)`` tuples, where
    ``masked_cells`` is ``[(row, col, true_value), ...]``.
    """
    dataset = []
    for t in range(n_tables):
        table, _specs, _parents = synthesize_table(
            seed=cfg.scm.seed + t,
            n_columns=cfg.scm.max_columns,
            n_rows=cfg.scm.n_rows,
        )
        masked, masked_cells = table.mask(mask_prob=cfg.train.mask_prob, seed=t)
        dataset.append((masked, masked_cells, table.columns))
    return dataset


def train_test_split(dataset, frac=0.8):
    k = max(1, int(len(dataset) * frac))
    return dataset[:k], dataset[k:]
