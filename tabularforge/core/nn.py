"""Pure-Python neural building blocks (operate on lists/floats).

Used by the cpu backend's TabFM forward pass. Everything is deterministic and
dependency-free so the whole framework runs anywhere Python 3.9+ runs.
"""
import math
import random


def _seed_vec(key, n, lo=-0.1, hi=0.1):
    from ..utils.stable import stable_vector
    return stable_vector(key, n, lo, hi)


def matvec(W, x):
    """W: (out, in), x: (in,) -> (out,)."""
    return [sum(w[j] * x[j] for j in range(len(x))) for w in W]


def _add(a, b):
    return [a[i] + b[i] for i in range(len(a))]


class Linear:
    """A dense layer with hash-based weights (deterministic, no RNG state)."""

    def __init__(self, in_dim, out_dim, key="lin", bias=True):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.key = key
        self.W = [_seed_vec(f"{key}:w:{i}", in_dim, -0.5, 0.5) for i in range(out_dim)]
        self.b = _seed_vec(f"{key}:b", out_dim, -0.1, 0.1) if bias else [0.0] * out_dim

    def __call__(self, x):
        out = matvec(self.W, x)
        if self.b:
            out = _add(out, self.b)
        return out


class MLP:
    """A small multi-layer perceptron with relu hidden activations."""

    def __init__(self, dims, key="mlp"):
        self.layers = []
        for i in range(len(dims) - 1):
            self.layers.append(Linear(dims[i], dims[i + 1], key=f"{key}:{i}"))

    def __call__(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = [max(0.0, v) for v in x]
        return x


class LayerNorm:
    """Layer norm over a 1-D vector; supports 2-D lists (rows of tokens)."""

    def __init__(self, dim, key="ln", eps=1e-5):
        self.dim = dim
        self.eps = eps
        self.g = _seed_vec(f"{key}:g", dim, 0.9, 1.1)
        self.b = _seed_vec(f"{key}:b", dim, -0.05, 0.05)

    def __call__(self, x):
        if x and isinstance(x[0], (list, tuple)):
            return [self.__call__(row) for row in x]
        mean = sum(x) / len(x)
        var = sum((v - mean) ** 2 for v in x) / len(x)
        std = math.sqrt(var + self.eps)
        return [self.g[i] * ((v - mean) / std) + self.b[i] for i, v in enumerate(x)]


class Embedding:
    """A hash-based embedding table: id -> vector (deterministic)."""

    def __init__(self, num, dim, key="emb"):
        self.num = num
        self.dim = dim
        self.key = key
        self.table = [_seed_vec(f"{key}:{i}", dim) for i in range(num)]

    def __call__(self, idx):
        return list(self.table[idx % self.num])
