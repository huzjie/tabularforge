"""Tests for the benchmark harness."""
import unittest

from tabularforge.config import load_config
from tabularforge.backends import get_backend
from tabularforge.bench import run_all


class TestBench(unittest.TestCase):
    def test_run_all_mock(self):
        cfg = load_config()
        backend = get_backend("mock", cfg)
        results = run_all(backend, cfg)
        self.assertEqual(len(results), 5)
        tasks = {r["task"] for r in results}
        self.assertEqual(tasks, {"imputation", "classification", "regression",
                                 "generation", "speed"})
        imp = next(r for r in results if r["task"] == "imputation")
        self.assertIsInstance(imp["accuracy"], float)
        self.assertGreaterEqual(imp["accuracy"], 0.0)
        self.assertLessEqual(imp["accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
