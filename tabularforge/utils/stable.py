"""Deterministic high-entropy RNG helpers (md5-seeded).

Reusable trick: seed a `random.Random` from the first 8 bytes of an md5 digest
of the key. Naively flattening the 16-byte digest repeats only ~2 distinct
values; the first-8-bytes-as-seed approach gives a full 64-bit space quickly and
is far cheaper than per-coordinate md5 (which would slow the fast path down).
"""
import hashlib
import random


def stable_seed(key):
    return int.from_bytes(hashlib.md5(str(key).encode("utf-8")).digest()[:8], "big")


def stable_float(key, lo=0.0, hi=1.0):
    r = random.Random(stable_seed(key))
    return lo + (hi - lo) * r.random()


def stable_ints(key, n, lo, hi):
    r = random.Random(stable_seed(key))
    return [r.randint(lo, hi) for _ in range(n)]


def stable_vector(key, n, lo=-1.0, hi=1.0):
    r = random.Random(stable_seed(key))
    return [lo + (hi - lo) * r.random() for _ in range(n)]


def stable_choice(key, items):
    r = random.Random(stable_seed(key))
    return r.choice(list(items))


def stable_bool(key, p=0.5):
    return stable_float(key) < p
