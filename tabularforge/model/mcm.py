"""Masked Cell Modeling loss + accuracy.

The training objective is exactly masked language modeling, applied to tables:
mask a random subset of cells and score how well the model reconstructs them.
Categorical cells use cross-entropy over the column domain; numeric cells use
MSE against the true value.
"""
import math


def masked_cell_loss(table, masked_cells, model, columns):
    """Compute the total MCM loss over ``masked_cells`` (list of (r, c, true))."""
    total = 0.0
    for r, c, true in masked_cells:
        col = columns[c]
        row = table.rows[r]
        if col.dtype == "categorical":
            pred = model.predict_categorical(row, c, r, columns, col.domain)
            total += 0.0 if pred == true else 1.0
        else:
            pred = model.predict_numeric(row, c, r, columns)
            try:
                total += (pred - float(true)) ** 2
            except (TypeError, ValueError):
                total += 1.0
    return total / max(1, len(masked_cells))


def reconstruction_accuracy(table, masked_cells, model, columns, tol=0.5):
    """Fraction of masked cells correctly reconstructed.

    Categorical cells need an exact match; numeric cells count if within
    ``tol`` of the true value.
    """
    if not masked_cells:
        return 1.0
    correct = 0
    for r, c, true in masked_cells:
        col = columns[c]
        row = table.rows[r]
        if col.dtype == "categorical":
            pred = model.predict_categorical(row, c, r, columns, col.domain)
            correct += 1 if pred == true else 0
        else:
            pred = model.predict_numeric(row, c, r, columns)
            try:
                correct += 1 if abs(pred - float(true)) <= tol else 0
            except (TypeError, ValueError):
                pass
    return correct / len(masked_cells)
