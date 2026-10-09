"""Training demo: watch the mock backend's accuracy climb.

Run: python -m tabularforge.examples.train_demo
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tabularforge.config import load_config
from tabularforge.backends import get_backend
from tabularforge.bench.imputation import run_imputation


def main():
    cfg = load_config()
    backend = get_backend("mock", cfg)
    print(f"before training: accuracy = {run_imputation(backend, cfg)['accuracy']:.3f}")
    for epoch in range(1, 11):
        backend.train_step()
        acc = run_imputation(backend, cfg)["accuracy"]
        print(f"  epoch {epoch:2d}: skill={backend.skill:.2f}  accuracy={acc:.3f}")


if __name__ == "__main__":
    main()
