"""DAG sampling utilities.

The core trick behind Kumo Tabular's pretraining data: instead of collecting
real tables, SAMPLE a directed acyclic graph of causal relations, then generate
tables where each column is a function of its parents. This yields an
effectively infinite, fully-labeled stream of tables with known ground truth.
"""
from ..utils.stable import stable_bool


def sample_dag(n_nodes, edge_prob=0.35, seed=0):
    """Sample a DAG over ``n_nodes`` nodes.

    Node ``j`` may depend on any earlier node ``i < j`` with probability
    ``edge_prob``. Restricting edges to ``i < j`` guarantees acyclicity by
    construction (a topological order exists: the natural index order).
    Returns a list of parent sets, ``parents[j] = [i, ...]``.
    """
    parents = [[] for _ in range(n_nodes)]
    for j in range(n_nodes):
        for i in range(j):
            if stable_bool(f"dag:{seed}:{i}->{j}", edge_prob):
                parents[j].append(i)
    return parents


def topological_order(parents):
    """Return a topological order of the nodes given parent sets.

    The natural order 0..n-1 is already topological for ``sample_dag`` output
    (edges only go forward), but we recompute it defensively for hand-built
    graphs.
    """
    n = len(parents)
    indeg = [0] * n
    children = [[] for _ in range(n)]
    for j in range(n):
        for i in parents[j]:
            indeg[j] += 1
            children[i].append(j)
    queue = [i for i in range(n) if indeg[i] == 0]
    order = []
    while queue:
        node = queue.pop(0)
        order.append(node)
        for c in children[node]:
            indeg[c] -= 1
            if indeg[c] == 0:
                queue.append(c)
    return order
