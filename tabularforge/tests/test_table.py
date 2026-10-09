"""Tests for the table data model."""
import unittest

from tabularforge.tabular.table import Table, Column, MISSING


class TestTable(unittest.TestCase):
    def test_mask_and_fill(self):
        cols = [Column("a", "numeric"), Column("b", "categorical", ["x", "y"])]
        rows = [[1.0, "x"], [2.0, "y"], [3.0, "x"]]
        t = Table(cols, rows)
        masked, cells = t.mask(mask_prob=0.5, seed=3)
        self.assertTrue(len(cells) > 0)
        for (r, c, _true) in cells:
            self.assertTrue(masked.is_missing(r, c))

    def test_missing_cells(self):
        cols = [Column("a", "numeric")]
        rows = [[1.0], [MISSING], [None]]
        t = Table(cols, rows)
        self.assertEqual(t.missing_cells(), [(1, 0), (2, 0)])

    def test_records_roundtrip(self):
        cols = [Column("a", "numeric"), Column("b", "categorical", ["x", "y"])]
        rows = [[1.0, "x"], [2.0, "y"]]
        t = Table(cols, rows)
        recs = t.to_records()
        t2 = Table.from_records(recs, cols)
        self.assertEqual(t2.rows, t.rows)


if __name__ == "__main__":
    unittest.main()
