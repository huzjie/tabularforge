"""SCM synthetic table synthesizer.

Generates labeled tables by walking a sampled DAG in topological order. Each
column has a dtype and a deterministic generator function of its parents (plus
noise). Because parents are known, every generated cell has a recoverable
ground truth, which makes the data ideal for pretraining a masked-cell model
and for benchmarking imputation.
"""
import math

from ..utils.stable import stable_float, stable_seed, stable_choice
from .dag import sample_dag

DTYPES = ("numeric", "categorical", "datetime", "text")


class ColumnSpec:
    """A single column: name, dtype, and the categorical domain (if any)."""

    def __init__(self, name, dtype="numeric", domain=None, decimals=1):
        self.name = name
        self.dtype = dtype
        self.domain = domain or []
        self.decimals = decimals

    def to_dict(self):
        return {"name": self.name, "dtype": self.dtype, "domain": list(self.domain)}


class ScmSynthesizer:
    """Generate synthetic tables from structural causal models."""

    def __init__(self, seed=42, edge_prob=0.35):
        self.seed = seed
        self.edge_prob = edge_prob

    def build_specs(self, n_columns):
        """Create ``n_columns`` ColumnSpecs with mixed dtypes and known parents."""
        parents = sample_dag(n_columns, self.edge_prob, seed=self.seed)
        specs = []
        for j in range(n_columns):
            dtype = ("categorical" if j % 3 == 1 else
                     "datetime" if j % 3 == 2 else "numeric")
            domain = None
            if dtype == "categorical":
                n_cat = 2 + (j % 4)
                domain = [f"cat{j}-{k}" for k in range(n_cat)]
            specs.append(ColumnSpec(f"col{j}", dtype, domain))
        return specs, parents

    def _parent_value(self, parent_val, spec):
        """Map a parent column value to a scalar contribution."""
        if spec.dtype == "categorical":
            # use the stable hash of the value as an index contribution
            return float(stable_seed(str(parent_val)) % 1000) / 1000.0
        try:
            return float(parent_val)
        except (TypeError, ValueError):
            return float(stable_seed(str(parent_val)) % 1000) / 1000.0

    def generate(self, n_columns=8, n_rows=256, seed=None):
        """Generate a table: list of rows (list of cell values) + specs + parents.

        Returns ``(rows, specs, parents)``.
        """
        seed = self.seed if seed is None else seed
        specs, parents = self.build_specs(n_columns)
        # re-seed the synth determinism per call
        base = seed * 1000003
        rows = []
        for r in range(n_rows):
            row = [None] * n_columns
            for j in range(n_columns):
                spec = specs[j]
                # aggregate causal parents
                agg = 0.0
                for pi in parents[j]:
                    agg += self._parent_value(row[pi], specs[pi])
                noise = stable_float(f"syn:{base}:{r}:{j}:noise", -1.0, 1.0)
                if spec.dtype == "numeric":
                    # linear causal function + noise
                    val = agg + noise
                    val = round(val, spec.decimals)
                    row[j] = val
                elif spec.dtype == "categorical":
                    # deterministic truth table driven by parents + noise
                    key = int(stable_seed(f"syn:{base}:{r}:{j}:cat") % 1000)
                    idx = (key + int(abs(agg) * 7)) % len(spec.domain)
                    row[j] = spec.domain[idx]
                elif spec.dtype == "datetime":
                    # day-of-year derived from parents, offset by noise
                    day = int(stable_seed(f"syn:{base}:{r}:{j}:dt") % 365) + int(abs(agg) * 30)
                    row[j] = f"2026-{(day // 30) % 12 + 1:02d}-{day % 28 + 1:02d}"
                else:  # text
                    row[j] = f"text-{int(stable_seed(f'syn:{base}:{r}:{j}:tx') % 5000):04d}"
            rows.append(row)
        return rows, specs, parents


def synthesize_table(seed=42, n_columns=8, n_rows=256):
    """Convenience wrapper: build a Table from a fresh SCM draw."""
    from ..tabular.table import Table, Column
    syn = ScmSynthesizer(seed=seed)
    rows, specs, parents = syn.generate(n_columns=n_columns, n_rows=n_rows)
    columns = [Column(s.name, s.dtype, s.domain) for s in specs]
    table = Table(columns, rows)
    return table, specs, parents
