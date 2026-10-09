"""Impute a CSV file's missing values.

Usage:
    python impute_csv.py path/to/data.csv [--target-col age]

This is a real, end-to-end use case: read a CSV, infer the schema, impute
missing cells, and write the result back out.
"""
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tabularforge.config import load_config
from tabularforge.engine import TabularEngine
from tabularforge.tabular.schema import infer_schema
from tabularforge.tabular.table import Table, MISSING


def _coerce(value):
    if value is None or value == "":
        return MISSING
    return value


def main(argv):
    if not argv:
        print("usage: python impute_csv.py <input.csv> [output.csv]")
        return 1
    src = Path(argv[0])
    dst = Path(argv[1]) if len(argv) > 1 else src.with_name(src.stem + "-imputed.csv")

    with src.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        records = list(reader)

    columns = infer_schema(records)
    rows = [[_coerce(rec.get(c.name)) for c in columns] for rec in records]
    table = Table(columns, rows, name=src.stem)

    cfg = load_config()
    engine = TabularEngine(cfg)
    filled, preds = engine.impute(table)

    with dst.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([c.name for c in columns])
        for r in filled.rows:
            writer.writerow([None if v is MISSING else v for v in r])

    print(f"[impute_csv] imputed {len(preds)} cells -> {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
