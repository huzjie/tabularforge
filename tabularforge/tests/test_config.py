"""Tests for config loading."""
import unittest
import tempfile
import os

from tabularforge.config import Config, load_config


class TestConfig(unittest.TestCase):
    def test_default(self):
        cfg = load_config()
        self.assertEqual(cfg.name, "tabularforge")
        self.assertGreater(cfg.model.dim, 0)

    def test_from_dict_partial(self):
        cfg = Config.from_dict({"backend": "cpu", "model": {"dim": 128}})
        self.assertEqual(cfg.backend, "cpu")
        self.assertEqual(cfg.model.dim, 128)

    def test_load_yaml(self):
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            f.write("backend: cpu\nmodel:\n  dim: 96\n")
            path = f.name
        try:
            cfg = load_config(path)
            self.assertEqual(cfg.backend, "cpu")
            self.assertEqual(cfg.model.dim, 96)
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()
