"""Column dtype inference.

Given raw values, infer whether a column is numeric / categorical / datetime /
text. Inference is used by the JSON API (where incoming tables have no schema)
and by the synthesizer to assign dtypes.
"""
import re

_DT_RE = re.compile(r"^\d{4}-\d{2}-\d{2}([ T]\d{2}:\d{2}(:\d{2})?)?$")


def _is_float(v):
    try:
        float(v)
        return True
    except (TypeError, ValueError):
        return False


def _is_datetime(v):
    return isinstance(v, str) and bool(_DT_RE.match(v))


def infer_dtype(values):
    """Infer a single dtype for a list of non-missing values."""
    vals = [v for v in values if v is not None]
    if not vals:
        return "text"
    if all(_is_float(v) for v in vals):
        return "numeric"
    if all(_is_datetime(v) for v in vals):
        return "datetime"
    distinct = len(set(map(str, vals)))
    if distinct <= max(8, len(vals) * 0.2):
        return "categorical"
    return "text"


def infer_schema(records):
    """Infer a list of ``Column`` from a list of dict records."""
    from .table import Column, MISSING
    keys = []
    for rec in records:
        for k in rec:
            if k not in keys:
                keys.append(k)
    columns = []
    for k in keys:
        vals = [rec.get(k) for rec in records if rec.get(k) is not None]
        dtype = infer_dtype(vals)
        domain = None
        if dtype == "categorical":
            domain = sorted({str(v) for v in vals})
        columns.append(Column(k, dtype, domain))
    return columns
