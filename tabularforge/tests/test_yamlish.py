"""Tests for the minimal YAML parser."""
import unittest

from tabularforge.utils.yamlish import parse


class TestYamlish(unittest.TestCase):
    def test_scalars(self):
        self.assertEqual(parse("a: 1\nb: hello\nc: true\n"), {"a": 1, "b": "hello", "c": True})

    def test_inline_comment(self):
        self.assertEqual(parse("name: mock  # comment\n"), {"name": "mock"})

    def test_block_list(self):
        self.assertEqual(parse("items:\n  - a\n  - b\n"), {"items": ["a", "b"]})

    def test_nested_map(self):
        self.assertEqual(parse("model:\n  dim: 64\n  x: 1\n"),
                         {"model": {"dim": 64, "x": 1}})

    def test_inline_list(self):
        self.assertEqual(parse("mrl: [8, 16, 32]\n"), {"mrl": [8, 16, 32]})


if __name__ == "__main__":
    unittest.main()
