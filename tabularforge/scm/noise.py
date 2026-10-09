"""Noise generators for the SCM engine."""
from ..utils.stable import stable_float, stable_choice


def gaussian(key, lo=-2.0, hi=2.0):
    """A cheap bounded 'noise' term (uniform, deterministic per key)."""
    return stable_float(key, lo, hi)


def categorical_noise(key, n_cats):
    return stable_choice(key, range(n_cats))
