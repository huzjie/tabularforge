"""CSV / JSON table loaders (stdlib only)."""
import csv
import json
from pathlib import Path

from ..tabular.schema import infer_schema
from ..tabular.table import Table, MISSING


def _coerce(v):
    if v is None or v == "":
        return MISSING
    return v


def load_csv(path, columns=None, name=None):
    """Load a CSV into a Table, inferring the schema when not provided."""
    path = Path(path)
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        records = list(reader)
    cols = columns or infer_schema(records)
    rows = [[_coerce(rec.get(c.name)) for c in cols] for rec in records]
    return Table(cols, rows, name or path.stem)


def load_json(path, columns=None, name=None):
    """Load a JSON array of objects into a Table."""
    path = Path(path)
    records = json.loads(path.read_text(encoding="utf-8"))
    cols = columns or infer_schema(records)
    rows = [[_coerce(rec.get(c.name)) for c in cols] for rec in records]
    return Table(cols, rows, name or path.stem)


def save_csv(table, path):
    path = Path(path)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([c.name for c in table.columns])
        for r in table.rows:
            writer.writerow([None if v is MISSING else v for v in r])
    return path
