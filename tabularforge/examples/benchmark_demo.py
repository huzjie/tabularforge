"""Run the full benchmark + robustness + scaling.

Run: python -m tabularforge.examples.benchmark_demo
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tabularforge.config import load_config
from tabularforge.backends import get_backend
from tabularforge.bench import run_all, summarize
from tabularforge.bench.robustness import run_robustness
from tabularforge.bench.scaling import run_scaling


def main():
    cfg = load_config()
    backend = get_backend(cfg.backend, cfg)
    print(summarize(run_all(backend, cfg)))
    rob = run_robustness(backend, cfg)
    print("robustness (accuracy vs missing rate):")
    for pt in rob["curve"]:
        print(f"  rate={pt['missing_rate']:.2f}  acc={pt['accuracy']:.3f}")
    sc = run_scaling(backend, cfg)
    print("scaling (accuracy vs rows):")
    for pt in sc["curve"]:
        print(f"  rows={pt['n_rows']:4d}  acc={pt['accuracy']:.3f}")


if __name__ == "__main__":
    main()
