"""Structural causal model (SCM) synthetic-data engine."""
from .dag import sample_dag, topological_order
from .synth import ScmSynthesizer, ColumnSpec
from .causal import discover_parents, feature_importance

__all__ = [
    "sample_dag", "topological_order",
    "ScmSynthesizer", "ColumnSpec",
    "discover_parents", "feature_importance",
]
