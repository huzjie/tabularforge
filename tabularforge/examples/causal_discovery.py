"""Discover likely causal parents of a column (bonus capability).

Run:
    python -m tabularforge.examples.causal_discovery
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tabularforge.config import load_config
from tabularforge.scm.synth import synthesize_table
from tabularforge.scm.causal import discover_parents


def main():
    cfg = load_config()
    table, specs, true_parents = synthesize_table(
        seed=cfg.scm.seed, n_columns=cfg.scm.max_columns, n_rows=cfg.scm.n_rows)

    target = min(cfg.scm.max_columns - 1, 5)
    candidates = [i for i in range(cfg.scm.max_columns) if i != target]
    discovered, scores = discover_parents(table.rows, target, candidates, top_k=3)

    print(f"target column: {table.columns[target].name}")
    print(f"true parents : {true_parents[target]}")
    print(f"discovered   : {discovered}")
    print("MI scores:")
    for col, s in sorted(scores.items(), key=lambda kv: -kv[1]):
        print(f"  {table.columns[col].name}: {s:.3f}")


if __name__ == "__main__":
    main()
