"""Regression metrics."""
import math


def mae(preds, truths):
    if not truths:
        return 0.0
    errs = [abs(float(p) - float(t)) for p, t in zip(preds, truths)]
    return sum(errs) / len(errs)


def mse(preds, truths):
    if not truths:
        return 0.0
    errs = [(float(p) - float(t)) ** 2 for p, t in zip(preds, truths)]
    return sum(errs) / len(errs)


def rmse(preds, truths):
    return math.sqrt(mse(preds, truths))


def r2(preds, truths):
    if not truths:
        return 0.0
    vals = [float(t) for t in truths]
    mean = sum(vals) / len(vals)
    ss_tot = sum((v - mean) ** 2 for v in vals)
    ss_res = sum((float(p) - float(t)) ** 2 for p, t in zip(preds, truths))
    return 1.0 - (ss_res / ss_tot) if ss_tot else 0.0


def mape(preds, truths):
    if not truths:
        return 0.0
    errs = []
    for p, t in zip(preds, truths):
        tv = float(t)
        if tv == 0:
            continue
        errs.append(abs((float(p) - tv) / tv))
    return (sum(errs) / len(errs)) if errs else 0.0
