"""Generate a synthetic table from a schema and save it as CSV.

Run: python -m tabularforge.examples.generate_table
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tabularforge.config import load_config
from tabularforge.engine import TabularEngine
from tabularforge.tabular.table import Column, Table
from tabularforge.data.loader import save_csv


def main():
    cfg = load_config()
    engine = TabularEngine(cfg)
    columns = [
        Column("age", "numeric"),
        Column("income", "numeric"),
        Column("segment", "categorical", ["low", "mid", "high"]),
        Column("city", "categorical", ["beijing", "shanghai", "shenzhen"]),
        Column("signup_date", "datetime"),
    ]
    rows = engine.generate(columns, n_rows=64)
    table = Table(columns, rows, name="generated")
    out = save_csv(table, Path("generated_table.csv"))
    print(f"generated -> {out}")


if __name__ == "__main__":
    main()
