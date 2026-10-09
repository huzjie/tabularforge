"""Column statistics: cardinality, missing ratio, mode, basic summaries."""
from collections import Counter


def column_stats(table, col_idx):
    """Return a summary dict for one column."""
    col = table.columns[col_idx]
    values = [r[col_idx] for r in table.rows if r[col_idx] is not None]
    missing = table.n_rows - len(values)
    stats = {
        "name": col.name,
        "dtype": col.dtype,
        "n": table.n_rows,
        "missing": missing,
        "missing_ratio": missing / table.n_rows if table.n_rows else 0.0,
        "cardinality": len(set(map(str, values))),
    }
    if col.dtype == "numeric":
        nums = [float(v) for v in values]
        stats["min"] = min(nums) if nums else None
        stats["max"] = max(nums) if nums else None
        stats["mean"] = (sum(nums) / len(nums)) if nums else None
    if col.dtype == "categorical":
        counts = Counter(map(str, values))
        stats["mode"] = counts.most_common(1)[0][0] if counts else None
    return stats


def table_profile(table):
    """Profile every column of a table."""
    return [column_stats(table, i) for i in range(table.n_cols)]
