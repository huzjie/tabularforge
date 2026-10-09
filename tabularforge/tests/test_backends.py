"""Tests for the backends."""
import unittest

from tabularforge.config import Config, load_config
from tabularforge.backends import get_backend, list_backends
from tabularforge.scm.synth import synthesize_table


class TestBackends(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config()

    def test_list_backends(self):
        names = list_backends()
        for expected in ("mock", "cpu", "openai", "vllm", "transformers"):
            self.assertIn(expected, names)

    def test_mock_impute_accuracy_tracks_skill(self):
        backend = get_backend("mock", self.cfg)
        backend._skill = 0.0
        table, _, _ = synthesize_table(seed=self.cfg.scm.seed,
                                       n_columns=self.cfg.scm.max_columns,
                                       n_rows=self.cfg.scm.n_rows)
        masked, cells = table.mask(mask_prob=self.cfg.train.mask_prob, seed=0)
        preds = backend.impute(masked)
        correct = sum(1 for (r, c, true) in cells if preds.get((r, c)) == true)
        self.assertEqual(correct, 0)  # skill 0 -> always wrong
        backend._skill = 1.0
        preds = backend.impute(masked)
        correct = sum(1 for (r, c, true) in cells if preds.get((r, c)) == true)
        self.assertEqual(correct, len(cells))  # skill 1 -> always correct

    def test_mock_train_step_monotonic(self):
        backend = get_backend("mock", self.cfg)
        prev = backend.skill
        for _ in range(10):
            backend.train_step()
            self.assertGreaterEqual(backend.skill, prev)
            prev = backend.skill
        self.assertEqual(backend.skill, 1.0)

    def test_cpu_impute_returns_all(self):
        backend = get_backend("cpu", self.cfg)
        table, _, _ = synthesize_table(seed=self.cfg.scm.seed,
                                       n_columns=self.cfg.scm.max_columns,
                                       n_rows=8)
        masked, cells = table.mask(mask_prob=0.3, seed=0)
        preds = backend.impute(masked)
        self.assertEqual(set(preds.keys()), set((r, c) for r, c, _ in cells))


if __name__ == "__main__":
    unittest.main()
