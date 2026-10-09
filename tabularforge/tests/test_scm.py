"""Tests for the SCM synthetic-data engine."""
import unittest

from tabularforge.scm.dag import sample_dag, topological_order
from tabularforge.scm.synth import ScmSynthesizer, synthesize_table


class TestDag(unittest.TestCase):
    def test_acyclic_by_construction(self):
        parents = sample_dag(10, edge_prob=0.4, seed=7)
        # every edge points forward (i -> j with i < j), so natural order is topo
        for j, ps in enumerate(parents):
            for i in ps:
                self.assertLess(i, j)

    def test_topological_order(self):
        parents = [[], [0], [1], [0, 1], []]
        order = topological_order(parents)
        self.assertEqual(len(order), 5)


class TestSynth(unittest.TestCase):
    def test_generate_shape(self):
        syn = ScmSynthesizer(seed=1)
        rows, specs, parents = syn.generate(n_columns=6, n_rows=32)
        self.assertEqual(len(rows), 32)
        self.assertEqual(len(specs), 6)
        self.assertTrue(all(len(r) == 6 for r in rows))

    def test_deterministic(self):
        syn = ScmSynthesizer(seed=42)
        r1, _, _ = syn.generate(n_columns=5, n_rows=10)
        r2, _, _ = syn.generate(n_columns=5, n_rows=10)
        self.assertEqual(r1, r2)

    def test_synthesize_table(self):
        table, specs, parents = synthesize_table(seed=0, n_columns=5, n_rows=8)
        self.assertEqual(table.n_cols, 5)
        self.assertEqual(table.n_rows, 8)


if __name__ == "__main__":
    unittest.main()
