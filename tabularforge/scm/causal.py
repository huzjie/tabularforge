"""Causal parent discovery / feature importance.

Given a generated table (or any table) and a target column, score how much each
other column 'explains' the target by measuring how well the target can be
predicted from each candidate column alone (a cheap mutual-information proxy).
This is a bonus capability: the SCM gives us true parents, so we can validate
the discovery method against ground truth.
"""
import math

from ..utils.stable import stable_float


def _entropy(values):
    counts = {}
    for v in values:
        counts[v] = counts.get(v, 0) + 1
    n = len(values)
    h = 0.0
    for c in counts.values():
        p = c / n
        h -= p * math.log(p + 1e-9)
    return h


def feature_importance(rows, target_idx, candidate_indices):
    """Return a dict {col_idx: score} where score = normalized mutual info proxy.

    Higher means the candidate column is more predictive of the target.
    """
    target = [r[target_idx] for r in rows]
    h_target = _entropy(target)
    scores = {}
    for ci in candidate_indices:
        if ci == target_idx:
            continue
        col = [r[ci] for r in rows]
        # joint entropy of (col, target)
        pairs = list(zip(col, target))
        h_joint = _entropy(pairs)
        h_col = _entropy(col)
        mi = h_col + h_target - h_joint
        scores[ci] = mi / (h_target + 1e-9)
    return scores


def discover_parents(rows, target_idx, candidate_indices, top_k=None, threshold=0.1):
    """Discover likely causal parents of ``target_idx`` by MI ranking."""
    scores = feature_importance(rows, target_idx, candidate_indices)
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    out = [i for i, s in ranked if s >= threshold]
    if top_k is not None:
        out = out[:top_k]
    return out, scores
