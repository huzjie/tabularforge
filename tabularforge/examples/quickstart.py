"""Quickstart: synthesize a table, mask it, impute, and benchmark.

Run from anywhere:
    python -m tabularforge.examples.quickstart
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tabularforge.config import Config, load_config
from tabularforge.engine import TabularEngine
from tabularforge.scm.synth import synthesize_table


def main():
    cfg = load_config()
    engine = TabularEngine(cfg)

    # 1. synthesize a table from a structural causal model (no real data)
    table, specs, parents = synthesize_table(
        seed=cfg.scm.seed, n_columns=cfg.scm.max_columns, n_rows=16)
    print(f"[quickstart] synthesized {table.n_rows}x{table.n_cols} table "
          f"(causal parents: {parents[:3]}...)")

    # 2. mask some cells and impute them (zero-shot)
    masked, masked_cells = table.mask(mask_prob=0.2, seed=0)
    filled, preds = engine.impute(masked)
    correct = sum(1 for (r, c, true) in masked_cells if preds.get((r, c)) == true)
    print(f"[quickstart] imputed {len(masked_cells)} cells, "
          f"correct={correct} ({correct / max(1, len(masked_cells)):.2f})")

    # 3. run the TabArena benchmark
    from tabularforge.bench import run_all, summarize
    print("[quickstart] benchmark\n" + summarize(run_all(engine.backend, cfg)))

    # 4. show the first few filled rows
    print("[quickstart] sample filled rows:")
    for rec in filled.to_records()[:3]:
        print("  ", rec)


if __name__ == "__main__":
    main()
