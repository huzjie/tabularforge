"""Scoring helpers."""
import math


def accuracy(preds, truths):
    if not truths:
        return 1.0
    return sum(1 for p, t in zip(preds, truths) if p == t) / len(truths)


def rmse(preds, truths):
    if not truths:
        return 0.0
    errs = []
    for p, t in zip(preds, truths):
        try:
            errs.append((float(p) - float(t)) ** 2)
        except (TypeError, ValueError):
            errs.append(1.0)
    return math.sqrt(sum(errs) / len(errs))
