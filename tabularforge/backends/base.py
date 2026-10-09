"""Backend abstraction + registry.

A backend is the smallest interface the framework needs to impute a table and
(optionally) train. The registry lets CLI / server / benchmark swap backends by
name without touching call sites.
"""
_BACKENDS = {}


def register_backend(name):
    def deco(cls):
        _BACKENDS[name] = cls
        return cls
    return deco


def get_backend(name, cfg):
    if name not in _BACKENDS:
        raise ValueError(f"unknown backend {name!r}; available={sorted(_BACKENDS)}")
    return _BACKENDS[name](cfg)


def list_backends():
    return sorted(_BACKENDS)


class Backend:
    """Base class. Subclasses implement ``impute`` and optionally ``train_step``."""

    name = "base"

    def __init__(self, cfg):
        self.cfg = cfg

    def impute(self, table):
        """Return {(row, col): value} for every missing cell."""
        raise NotImplementedError

    def train_step(self):
        """One training step; return a scalar metric (optional)."""
        raise NotImplementedError

    @property
    def skill(self):
        return getattr(self, "_skill", 0.0)
